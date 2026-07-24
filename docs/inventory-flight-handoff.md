# Inventory handoff: Transformer Arrow Flight v2

This document is the integration handoff for the Inventory client. The
language-neutral source of truth is [`contracts/flight/v2`](../contracts/flight/v2/README.md):
its JSON Schemas and golden fixtures take precedence over examples in prose.
Operational deployment is covered by the
[`Flight service runbook`](flight-operations.md).

## Compatibility boundary

- Contract name: `transformer-flight`.
- Contract version: `2`.
- Supported operations: `fit`, `predict`.
- Canonical devices: `cpu`, `cuda`, `auto`.
- Expected Inventory client: `arrow-flight-client@0.0.8`.
- Every RPC carries configured metadata `authorization: Bearer TOKEN`.
- Inventory configures client `maxSendMessageLength` and
  `maxReceiveMessageLength` to at least the `maxMessageBytes` value returned by
  capabilities. The v2 default interoperability target is 16 MiB; the initial
  RecordBatch target is about 8 MiB.
- V2 uses `DoAction`, streaming `DoPut`, `GetFlightInfo`, and streaming
  `DoGet`. It does **not** use `DoExchange` or `PollFlightInfo`.
- V2 is a single-Transformer-instance protocol. There is no shared-storage or
  multi-replica scheduling contract.
- V1 actions, descriptor paths, durable jobs and idempotency responses are not
  accepted. Inventory and Transformer must switch to v2 in one deployment
  boundary.

The Transformer test suite does not depend on an Inventory checkout. No real
Node-to-PyArrow interoperability run is implied by this document; Inventory
owns that consumer-side verification.

Inventory receives the service host, port and transport settings through its
deployment configuration; v2 has no endpoint-discovery action. Production
configuration must include the trusted server CA and server name expected by
the certificate, plus client certificate/key when mTLS is enabled. Endpoint
URI spelling is client-library specific, so the normative contract identifies
the authority and TLS settings rather than inventing a second URI format.

## Actions

`ListActions` advertises exactly these names:

| Action | Request schema | Purpose | Idempotency |
| --- | --- | --- | --- |
| `transformer.v2.capabilities` | `query.schema.json` | Versions, schemas, limits, devices and queue capacity | Read-only |
| `transformer.v2.health` | `query.schema.json` | Authenticated liveness/readiness and aggregate metrics | Read-only |
| `transformer.v2.job.create` | `create.schema.json` | Create an immutable fit or predict job | Required |
| `transformer.v2.job.seal` | `seal.schema.json` | Make the complete ordered input manifest immutable | Required |
| `transformer.v2.job.start` | `job-mutation.schema.json` | Move a sealed job to the durable queue | Required |
| `transformer.v2.job.status` | `status.schema.json` | Read durable state, progress and results | Read-only |
| `transformer.v2.job.cancel` | `job-mutation.schema.json` | Request transactional cancellation | Required |

Every action body and every successful action result is a small UTF-8 JSON
object containing:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "11111111-1111-4111-8111-111111111111"
}
```

Unknown fields, actions, contract names and versions are rejected. A failed
operation is a failed Flight RPC; the service never encodes an error in a
successful JSON result.

Mutating actions additionally require an `idempotencyKey`. The server hashes
compact, key-sorted parsed JSON after removing `requestId` and
`idempotencyKey`. Repeating the same key, action, authenticated subject and
canonical request returns the originally stored result. That replay result can
therefore contain the first request's `requestId`. Reusing the key for a
different canonical request is a conflict. Keep JSON value types stable across
retries; for example, do not rewrite an integer as a floating-point number.

### Capabilities and health

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "66666666-6666-4666-8666-666666666666"
}
```

Send that body with either `transformer.v2.capabilities` or
`transformer.v2.health`. Capabilities is the authoritative runtime source for
service/PyArrow/Torch versions, schema IDs, limits, CPU/CUDA availability,
physical-device count and dynamic queue capacities. Health returns `live`,
`ready`, `draining`, ledger status, separate runtime/recovery storage
free-space telemetry, CUDA availability/quarantine count and aggregate metrics. CUDA
being unavailable does not make `live` false and does not by itself make
`ready` false.

Exact examples:

- [`capabilities.request.json`](../contracts/flight/v2/fixtures/json/capabilities.request.json)
  and [`capabilities.result.json`](../contracts/flight/v2/fixtures/json/capabilities.result.json)
- [`health.request.json`](../contracts/flight/v2/fixtures/json/health.request.json)
  and [`health.result.json`](../contracts/flight/v2/fixtures/json/health.result.json)

### Create fit

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "11111111-1111-4111-8111-111111111111",
  "idempotencyKey": "inventory-fit-20260718-1",
  "operation": "fit",
  "device": "cpu",
  "modelLabel": "returns.daily",
  "modelConfig": {
    "seqLen": 20,
    "hidden": 256,
    "layers": 5,
    "dropout": 0.1,
    "nhead": 8,
    "mode": "relaxed"
  },
  "trainingConfig": {
    "lr": 0.0005,
    "batchSize": 256,
    "epochs": 25,
    "patience": 5,
    "lossStage": 4,
    "lossSchedule": "epoch",
    "stageSize": 5,
    "useAmp": false,
    "weightDecay": 0.00001,
    "monitor": "ret_mae_skill",
    "monitorMinImprovement": 0.0,
    "saveBestCheckpoint": true,
    "seed": 42,
    "deterministic": false
  }
}
```

`seqLen` is required. Other model/training values have the same validated
defaults as the existing CLI when omitted. Fit accepts a logical `modelLabel`,
never a checkpoint path or model reference. On success the label's
owner-scoped alias advances to the new immutable model generation.

See [`create-fit.request.json`](../contracts/flight/v2/fixtures/json/create-fit.request.json)
and [`create-fit.result.json`](../contracts/flight/v2/fixtures/json/create-fit.result.json).

### Create predict

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "22222222-2222-4222-8222-222222222222",
  "idempotencyKey": "inventory-predict-20260718-1",
  "operation": "predict",
  "device": "auto",
  "modelRef": "mdl_0123456789abcdef0123456789abcdef",
  "predictionColumn": "out"
}
```

Predict requires exactly one of `modelRef` or `modelAlias`. An alias is
resolved during create and the concrete immutable generation is returned as
`resolvedModelRef`; later alias changes cannot alter the job. Model and
preprocessing configuration is loaded from that generation. Predict does not
accept training/model overrides, paths or arbitrary argv.

See [`create-predict.request.json`](../contracts/flight/v2/fixtures/json/create-predict.request.json).

### Seal, start, status and cancel

Seal uses the authoritative server digests returned by committed uploads:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "55555555-5555-4555-8555-555555555555",
  "idempotencyKey": "inventory-seal-20260718-1",
  "jobId": "33333333-3333-4333-8333-333333333333",
  "manifest": [
    {
      "payloadId": "44444444-4444-4444-8444-444444444444",
      "ordinal": 0,
      "sha256": "eb454cc6a7bd34dc00d818d9e39282b359a309b151dbaa31a9f86e876f3d17bb"
    }
  ]
}
```

The manifest must match the complete committed input set, be sorted by
`ordinal`, and contain exactly contiguous ordinals `0..N-1`. Missing,
duplicate, additional or digest-mismatched entries are rejected. Inputs become
immutable after seal.

Start and cancel have the same body shape:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "99999999-9999-4999-8999-999999999999",
  "idempotencyKey": "inventory-start-20260718-1",
  "jobId": "33333333-3333-4333-8333-333333333333"
}
```

Use the action-specific idempotency key and action name. Status omits the key:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "requestId": "88888888-8888-4888-8888-888888888888",
  "jobId": "33333333-3333-4333-8333-333333333333"
}
```

Exact request/result pairs are in
[`fixtures/json`](../contracts/flight/v2/fixtures/json/). Poll status using its
`pollAfterMs`; terminal status uses `0`.

## Upload protocol

For every current logical Inventory payload, open exactly one streaming
`DoPut` with this path descriptor:

```text
pathDescriptor(
  "transformer", "v2", "jobs", jobId, "inputs", decimalOrdinal
)
```

After sending the Arrow schema, send exactly one application-metadata-only
message before any RecordBatch:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "jobId": "33333333-3333-4333-8333-333333333333",
  "payloadId": "44444444-4444-4444-8444-444444444444",
  "ordinal": 0,
  "schemaId": "inventory.sequence.fit.v1",
  "rows": 255
}
```

Use `inventory.sequence.predict.v1` for prediction. `rows` is the total rows in
all RecordBatches of this DoPut. The metadata-only layout is intentional and is
compatible with `arrow-flight-client@0.0.8`; it also represents typed empty
input by sending the schema and metadata but no RecordBatch.

RecordBatch boundaries are only transport chunking. **One DoPut is one
semantic `fit-stream`/`predict-stream` frame**, whether it carries zero, one or
many RecordBatches. RPC completion order has no semantic meaning;
`ordinal` defines frame order. Uploads may execute concurrently or out of
order, but the sealed manifest is ordered.

After start, the worker reads the sealed IPC files strictly by ordinal and
wraps each complete file in exactly one existing 8-byte big-endian
length-prefixed subprocess frame for prediction. Multiple RecordBatches in the
file never become multiple prediction calls. A predict job uses one
`predict-stream` subprocess and loads its resolved checkpoint once.

A fit worker instead gives its single `fit-stream` subprocess the already
validated durable input directory to read. Training order is
`epoch -> ordinal -> optimizer batches`: `epochs` is the job-wide epoch count,
and model, optimizer, loss scheduler, checkpoint selection and early stopping
all have one lifetime per job. Rows enter a bounded job-wide shuffle window in
ordinal order; shuffle windows and optimizer batches may cross payload
boundaries and therefore do not depend on transport partitioning. Only one
payload and the bounded window are materialized as tensors at a time, and each
payload is reopened on every epoch, so the complete dataset is never loaded
into memory. Changing Inventory's payload-size limit therefore does not change
batching, shuffle order or epoch-based training state.

The server writes all batches to one IPC file, validates each batch, computes
the digest, fsyncs the file, atomically publishes it, fsyncs the directory and
commits the ledger before returning exactly one `PutResult`. Its
`appMetadata` is:

```json
{
  "contract": "transformer-flight",
  "version": 2,
  "jobId": "33333333-3333-4333-8333-333333333333",
  "payloadId": "44444444-4444-4444-8444-444444444444",
  "ordinal": 0,
  "status": "committed",
  "rows": 255,
  "batches": 4,
  "bytes": 12345678,
  "sha256": "eb454cc6a7bd34dc00d818d9e39282b359a309b151dbaa31a9f86e876f3d17bb",
  "schemaFingerprint": "a5326a4ece5fcaf07e2f4022aa726f4fe9a29d66d2e5e6c227fb376ab90dd96c"
}
```

Treat `sha256` and `schemaFingerprint` as opaque authoritative server values.
The golden forms are
[`upload-fit.metadata.json`](../contracts/flight/v2/fixtures/json/upload-fit.metadata.json)
and [`put-result.metadata.json`](../contracts/flight/v2/fixtures/json/put-result.metadata.json).

### Upload retries and disconnects

- Disconnect before durable commit leaves no committed input. A startup pass
  removes any orphan temporary file.
- Disconnect after durable commit but before receiving `PutResult` leaves the
  input visible in `status.committedInputs`.
- First reconcile an uncertain upload through status. An exact retry must use
  the same owner, job, `payloadId`, `ordinal`, schema, rows, RecordBatch
  partitioning and values; it returns the original committed metadata.
- Reusing the ordinal or payload ID for different content is a conflict.
- Once the job is sealed, no upload retry or new input is accepted.

Preserving RecordBatch partitioning on an exact retry matters because the
server digest covers the reconstructed IPC file, not only logical cell values.

## Arrow schemas and value validation

Input column names and order are exact; additional columns are rejected.

```text
inventory.sequence.fit.v1
  src: list<float32|float64>       # exact width seqLen * featureDim
  tgt: list<float32|float64>       # exact width 6

inventory.sequence.predict.v1
  src: list<float32|float64>       # exact width seqLen * featureDim

transformer.prediction.v1
  <predictionColumn>: list<float32> # exact width 6
```

`list`, `large-list` and `fixed-size-list` are accepted for input. A
`fixed-size-list` is recommended because it carries width in the schema.
Variable-list width must remain identical across every row, batch and payload
in the job. The source width must be positive, divisible by immutable
`seqLen`, and match the resolved model `featureDim` for prediction.
Every payload in one job must also have the same schema fingerprint; do not
mix list representations, element precision or field nullability between
DoPut calls.

Validation rules:

- null list rows and null list elements are rejected;
- `src` may contain IEEE NaN but not positive/negative infinity;
- `tgt` is finite, has width 6, has non-negative volatility at index 4, and a
  hit probability in `[0, 1]` at index 5;
- every non-NaN input value must fit `float32`, including values transported as
  `float64`;
- prediction contains finite `float32` values, width 6, and exactly the input
  row count;
- typed empty input and output schemas are supported.

The prediction vector follows the existing CLI order:

```text
[meanR, sigmaR, logitTP, logitSL, volNext, logitHit]
```

Prediction input contains no `tgt`, and prediction output does not echo it.
Inventory retains its target column and performs any local comparison.

A typed empty prediction payload produces one typed empty output for the same
ordinal. A fit upload may itself be typed empty, but a fit job with no usable
training rows cannot publish a checkpoint because existing `fit-stream`
semantics require at least one non-empty training frame.

## State machine and device behavior

```text
UPLOADING -> SEALED -> QUEUED -> RUNNING
                                  |-> SUCCEEDED
                                  |-> FAILED
                                  |-> RETRYING -> RUNNING
                                  `-> CANCELLING -> CANCELLED
```

Cancel from `UPLOADING`, `SEALED`, `QUEUED` or `RETRYING` immediately produces
`CANCELLED`. Cancel from `RUNNING` first produces `CANCELLING`. Terminal states
are immutable. Every state mutation increments `revision`.

`cpu` always selects CPU. `auto` selects CUDA at start when available and CPU
otherwise. Explicit `cuda` is checked at create and again at start; it never
falls back to CPU. If CUDA disappears before start, start fails with
`DEVICE_UNAVAILABLE` while the job remains `SEALED`. An available but busy GPU
leaves the job `QUEUED`. CUDA capacity is discovered from a boot-scoped
physical-device inventory; each attempt is bound to one GPU, but physical IDs
are never exposed through the contract.

A confirmed GPU loss closes the failed attempt, quarantines that device until
the next Linux boot and moves the job to `RETRYING`. Another healthy GPU
may claim the next attempt. The running subprocess is never migrated between
devices. CUDA OOM and an ordinary subprocess failure remain terminal.

Fit inputs are stored persistently by Transformer. At each completed global
epoch the service registers a recovery checkpoint containing model, optimizer,
AMP scaler, loss/early-stopping progress, best-checkpoint selection and random
state. A service or host interruption moves the fit to `RETRYING`; the next
attempt resumes from the latest registered epoch, or epoch zero when none was
completed. An incomplete epoch is repeated. Missing, corrupt or incompatible
registered recovery data fails explicitly and never causes a silent restart.

`UPLOADING`, `SEALED`, `QUEUED` and `RETRYING` fit jobs survive a service
restart. Prediction inputs and attempt artifacts remain runtime data; an
interrupted prediction becomes `FAILED / EXECUTION_INTERRUPTED`.

## Results and model references

Every status result includes `jobId`, `operation`, `state`, `revision`, all
state timestamps, requested/selected device, the committed input manifest,
scalar progress, attempt number, nullable recovery metadata, nullable safe
error, result object and `pollAfterMs`. Fit recovery metadata reports the
latest completed generation, whether training was already complete, the
generation restored by the most recently claimed attempt, retry count and last
retry code. It contains neither a filesystem path nor a physical GPU ID.
Progress keys follow the existing metrics JSONL and may grow; clients should
preserve unknown scalar keys rather than treating them as a closed schema.
`revision` and terminal state, not progress text, are the authoritative
concurrency/result signals.

For prediction, successful status contains one output descriptor per committed
input, with the same ordinal and row count. Use the exact `descriptorPath` from
status with `GetFlightInfo`:

```text
pathDescriptor(
  "transformer", "v2", "jobs", jobId, "outputs", decimalOrdinal
)
```

The server returns a random opaque ticket with an expiry. Inventory must not
construct, parse, persist as a model identifier, or derive a filesystem path
from that ticket. Call streaming `DoGet` with the same bearer identity. Tickets
are owner/job/output scoped and checked on every DoGet. A wrong-owner, unknown
or expired ticket is rejected. DoGet streams batches and does not require
buffering the entire result.

For fit, successful status returns an immutable opaque `modelRef` plus safe
checkpoint metadata (digest, byte count, model/train/data configuration and
checkpoint-selection metadata). **Transformer owns the checkpoint file and
all filesystem paths. Inventory stores and sends only `modelRef` or a logical
alias.** No network response exposes a checkpoint path.

## Retry matrix

| Operation | Safe client behavior after lost response |
| --- | --- |
| capabilities / health / status | Retry with a new `requestId` |
| create / seal / start / cancel | Replay the same canonical request with the same action-specific `idempotencyKey` |
| DoPut | Read status; if absent, repeat the exact upload with the same payload ID, ordinal and transport partitioning |
| GetFlightInfo | Request a new ticket from the status-provided descriptor |
| DoGet | Retry GetFlightInfo/DoGet while the output is retained; never invent a ticket |
| `RETRYING` | Keep polling the same `jobId`; do not create a replacement job |
| terminal `FAILED` fit | Fix the reported cause and submit a new job; the failed job is immutable |

`status` is authoritative after any ambiguous transport outcome. In
particular, a replayed start result can still say `QUEUED` even if a later
status is already `RUNNING` or terminal.

## Errors

Stable contract/application codes are:

| Category | Codes |
| --- | --- |
| Protocol, access and control | `INVALID_ARGUMENT`, `UNAUTHENTICATED`, `PERMISSION_DENIED`, `NOT_FOUND`, `ALREADY_EXISTS`, `FAILED_PRECONDITION`, `RESOURCE_EXHAUSTED`, `CANCELLED`, `UNAVAILABLE`, `INTERNAL` |
| Device, execution and storage | `DEVICE_UNAVAILABLE`, `DEVICE_LOST`, `EXECUTION_INTERRUPTED`, `SUBPROCESS_FAILED`, `SUBPROCESS_HUNG`, `MALFORMED_OUTPUT`, `CUDA_OUT_OF_MEMORY`, `DISK_FULL` |
| Training recovery | `RECOVERY_CHECKPOINT_UNAVAILABLE`, `RECOVERY_CHECKPOINT_INCOMPATIBLE`, `RECOVERY_INPUT_UNAVAILABLE` |

Terminal status contains only a stable code and safe message. Authentication
is evaluated before application handlers. Jobs, model aliases/generations,
outputs and tickets are bound to the authenticated subject. Cross-owner job,
model and descriptor lookups are intentionally indistinguishable from missing
resources; presenting another subject's otherwise valid output ticket is
rejected with `PERMISSION_DENIED`.

PyArrow 24 cannot emit exact gRPC `ALREADY_EXISTS`, `FAILED_PRECONDITION`, or
`RESOURCE_EXHAUSTED` from a Python Flight server. The RPC still fails and its
safe text retains the stable application code, but the observed wire status is
the closest PyArrow mapping (`INVALID_ARGUMENT` for the first two and
`UNKNOWN` for Arrow capacity errors). Inventory must not mistake that binding
limitation for a successful result. See
[`flight-dependency-note.md`](flight-dependency-note.md).

## Inventory configuration matrix

| Profile | Endpoint transport | Client credentials | Message settings | Intended use |
| --- | --- | --- | --- | --- |
| Production TLS | TLS | Bearer token + trusted server CA | send/receive at least advertised `maxMessageBytes` | Production |
| Production mTLS | TLS | Bearer token + trusted server CA + client certificate/key | same | Production with client-certificate enforcement |
| Development | Plaintext loopback | Bearer token still required | same | Local test only |
| LAN | Plaintext non-loopback, explicitly enabled server-side | Bearer token still required | same | Temporary trusted-LAN migration only, not production security |

TLS/mTLS policy never selects a compute device. The job's canonical `device`
field is the only device request.

Recommended client settings for v2 defaults:

```text
configured metadata:       authorization = Bearer <secret>
target RecordBatch:         about 8 MiB
maxSendMessageLength:       >= 16777216
maxReceiveMessageLength:    >= 16777216
application timeout:        short for actions/uploads, polling for execution
```

Always use the live capabilities result if deployment values differ from the
defaults. The 16 MiB value is an interoperability target, not a hard PyArrow
server receive cap; application quotas remain authoritative.

## Minimal integration sequence

1. Call authenticated `ListActions`, capabilities and health; require v2 and
   the expected schema IDs.
2. Create the fit/predict job with a fresh idempotency key. Persist `jobId`,
   create result and, for predict, `resolvedModelRef`.
3. For each logical Inventory payload, allocate a stable payload UUID and
   ordinal, then perform one DoPut. Persist each complete PutResult.
4. On uncertainty, reconcile `committedInputs` through status before retrying.
5. Seal exactly the ordered PutResult manifest with a stable seal key.
6. Start with a stable start key and poll status no faster than `pollAfterMs`.
7. For predict, use each returned descriptor with GetFlightInfo and its opaque
   ticket with streaming DoGet. Match by ordinal, not completion order.
8. For fit, persist `modelRef`; do not request or infer its filesystem path.

The operator runbook contains a local service command and an authenticated
manual control-plane probe. Real Node-to-PyArrow and physical GPU gates must be
reported separately and must not be inferred from Python integration tests.
