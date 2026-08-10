# Inventory handoff: Transformer Arrow Flight v3

> Type: Reference. Consumer integration guide for the breaking v3 protocol.

The normative wire contract is
[`app/contracts/flight/v3`](../app/contracts/flight/v3/README.md). JSON Schemas
and golden fixtures in that directory take precedence over this guide. Service
deployment and recovery operations are documented in the
[`Flight runbook`](flight-operations.md); the architectural decision is
[`ADR 0005`](adr/0005-durable-streaming-flight-v3.md).

V3 has no v2 compatibility surface. Inventory must require exactly protocol
version 3 and must not fall back to v2 action names, descriptor paths or state
semantics.

## Transport and authentication

Use Arrow Flight `DoAction`, `DoPut`, `GetFlightInfo` and `DoGet`. V3 does not
use `DoExchange` or `PollFlightInfo`. Every RPC carries:

```text
authorization: Bearer a.<base64url>
```

The credential represents one owner subject. Jobs, model aliases, model
references, status, receipts, tickets and outputs are owner-scoped. Never log
the bearer credential or an opaque output ticket.

Every action document begins with:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "requestId": "UUID"
}
```

Mutations additionally require a stable action-specific `idempotencyKey`.
Repeating the same canonical request is safe; reusing a key for a different
request is rejected. A failed operation is a failed Flight RPC. Stable
application codes are included in safe error text and in terminal status.

## Actions

The server advertises exactly:

| Action | Purpose | Fenced mutation |
| --- | --- | --- |
| `transformer.v3.capabilities` | Versions, schemas, limits and capacity | No |
| `transformer.v3.health` | Authenticated liveness/readiness | No |
| `transformer.v3.job.create` | Create a client-identified fit or predict job | Initial ownership |
| `transformer.v3.job.acquire` | Transfer Inventory ownership and advance its fence | Compare-and-swap |
| `transformer.v3.job.status` | Read bounded job state and result summary | No |
| `transformer.v3.job.inputs.list` | Reconcile committed inputs by revision | No |
| `transformer.v3.job.input.close` | Commit EOF and the immutable input summary | Yes |
| `transformer.v3.job.outputs.list` | List terminal output receipts | No |
| `transformer.v3.job.cancel` | Cancel a non-terminal job | Yes |
| `transformer.v3.model.describe` | Resolve and describe an immutable model | No |

At startup, call `capabilities` and fail closed unless
`protocolVersions` equals `[3]`. Effective limits in that response are the
runtime authority; do not copy repository defaults into Inventory.

## Persist identity before create

Inventory generates and commits these values in its PostgreSQL transaction
before the first network request:

- `jobId`: stable remote identity for this logical job;
- `clientExecutionId`: identity of the current Inventory claim;
- create `idempotencyKey` and the immutable create document.

Fit create follows the
[`create-fit.request.json`](../app/contracts/flight/v3/fixtures/json/create-fit.request.json)
fixture. Predict create follows
[`create-predict.request.json`](../app/contracts/flight/v3/fixtures/json/create-predict.request.json).
Predict supplies exactly one `modelRef` or owner-scoped `modelAlias`. Transformer
atomically resolves an alias and returns the immutable `resolvedModelRef`.

The create result starts with:

```text
input.state     = OPEN
execution.state = WAITING_INPUT
fencingToken    = "1"
```

Persist the whole result, especially `resolvedModelRef`, ownership and upload
limits. If the response is lost, replay the same logical create. `jobId` is
client-generated and Transformer retains its compact identity after heavy job
data is retired, so no separate resolve call is needed.

## ML data contract

Every create carries an Inventory-owned semantic identity:

```json
{
  "dataContract": {
    "id": "inventory.learning-dataset",
    "version": 1,
    "dataContractSha256": "64 lowercase hex characters",
    "seqLen": 10,
    "featureDim": 891,
    "targetSchemaId": "inventory.target.v1"
  }
}
```

Inventory owns the canonical document behind this digest, including ordered
feature identities, target semantics, normalization, missing-value policy and
profile version. Transformer stores the identity but does not recreate those
semantics. Predict create must use the digest certified by the resolved model;
a mismatch fails before upload with `MODEL_SCHEMA_MISMATCH`.

## Cross-system fencing and takeover

Every upload, close and cancel carries the current pair:

```json
{
  "clientExecutionId": "UUID",
  "fencingToken": "7"
}
```

The token is a canonical positive decimal string, not a JSON integer. On
Inventory lease takeover, call `transformer.v3.job.acquire` with the previous
execution ID, expected token and the new execution ID. Transformer compares
the old pair atomically and returns the next token. Persist that response before
issuing mutations from the new claim.

A stale owner receives `STALE_FENCE`. The fence is checked before DoPut data is
accepted and again immediately before its durable receipt commit, so a request
that overlaps takeover cannot overwrite the new owner's input.

## Upload

One DoPut is one semantic payload. RecordBatch boundaries are transport
chunking only. Use:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "inputs", ordinal)
```

Write one application-metadata message before RecordBatches:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "jobId": "UUID",
  "clientExecutionId": "UUID",
  "fencingToken": "7",
  "payloadId": "UUID",
  "ordinal": 0,
  "schemaId": "inventory.sequence.fit.v2",
  "dataContractSha256": "64 lowercase hex characters",
  "rows": 1820
}
```

The exact physical schema is fixed:

```text
inventory.sequence.fit.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
  tgt: non-null FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
```

Out-of-order DoPut completion is allowed. The server returns one PutResult only
after the immutable artifact and PostgreSQL receipt are durable. Persist the
complete PutResult. `nextInputOrdinal` is the first missing ordinal;
`inputRevision` is a per-job monotonic commit revision; `queued=true` means this
commit caused automatic execution queueing.

The first non-empty contiguous prefix queues the job. A fit worker may therefore
be `RUNNING` while input remains `OPEN`. Arrival order and timing do not change
the worker's logical order: ordinal, then row within the payload.

## Reconcile inputs by revision

If a PutResult is lost, do not infer failure from the transport. Reconcile with
`transformer.v3.job.inputs.list`. The first page supplies `afterRevision` and
freezes the returned `snapshotRevision`. Continue with the same snapshot and
the returned cursor while:

```text
cursor < commitRevision <= snapshotRevision
```

After the traversal, persist its `snapshotRevision` as the next
`afterRevision`. A low ordinal committed late receives a higher revision and is
therefore visible in the next traversal. Page size is at most 100.

An absent receipt may be retried with the same `payloadId`, ordinal, schema,
rows and data. An exact committed duplicate returns the existing receipt;
different content for an occupied identity or ordinal is a conflict.

## Close input

Close is EOF, not a start command. Build the canonical digest over all server
receipts sorted by ordinal. The fields are:

```text
payloadId, ordinal, schemaId, dataContractSha256, rows, batches, bytes,
sha256, schemaFingerprint
```

Exclude `commitRevision`, timestamps and arrival order. Send only the summary:

```json
{
  "jobId": "UUID",
  "clientExecutionId": "UUID",
  "fencingToken": "7",
  "payloadCount": 5,
  "totalRows": 9100,
  "totalBytes": 324625520,
  "manifestSha256": "64 lowercase hex characters"
}
```

Transformer verifies contiguous ordinals `0..payloadCount-1`, totals, digest,
one physical Arrow schema and one data-contract digest before committing
`input.state=CLOSED`. New uploads are then rejected.

An empty fit close fails with `EMPTY_INPUT` and leaves input `OPEN`. Empty
predict is valid: zero payloads produce zero outputs. A typed-empty predict
payload produces one typed-empty output with the requested prediction column.

## State and polling

Status exposes two independent state axes:

```text
input.state:
  OPEN | CLOSED | ABORTED

execution.state:
  WAITING_INPUT | QUEUED | RUNNING | RETRYING | CANCELLING |
  SUCCEEDED | FAILED | CANCELLED
```

`OPEN + RUNNING` is normal. Terminal success requires closed input and atomic
publication. Poll no faster than `pollAfterMs`. Status is bounded and reports
counts rather than embedding all input/output receipts.

Before EOF, a failed fit attempt repeats incomplete epoch zero from its start;
durably committed inputs are retained. After EOF, recovery checkpoints are at
complete global-epoch boundaries. Inventory must tolerate `RETRYING` without
resending already committed payloads.

## Prediction output

Output discovery is available only after `execution.state=SUCCEEDED`:

1. Page through `transformer.v3.job.outputs.list`.
2. For each ordinal, call `GetFlightInfo` with:

   ```text
   pathDescriptor("transformer", "v3", "jobs", jobId, "outputs", ordinal)
   ```

3. Use the returned opaque ticket in `DoGet` before it expires.

The output schema is:

```text
transformer.prediction.v2
  <predictionColumn>: non-null FixedSizeList<Float32>[6]
```

Transformer may prepare attempt-local results while input is open, but no
partial output becomes visible. All output receipts and `SUCCEEDED` commit in
one terminal transaction.

## Models

`transformer.v3.model.describe` accepts one `modelRef` or owner-scoped
`modelAlias`. A published `modelRef` and generation are immutable and have no
automatic TTL. `predictionColumn` belongs to the prediction job, not the model.

Stable lifecycle failures are:

| Code | Meaning |
| --- | --- |
| `NOT_FOUND` | No owner-visible model identity |
| `MODEL_UNAVAILABLE` | Metadata exists but checkpoint is absent |
| `MODEL_CORRUPT` | Checkpoint size or digest is invalid |
| `MODEL_SCHEMA_MISMATCH` | Model is uncertified or data contract differs |

Models created before v3 are not implicitly compatible. They require explicit
offline certification or retraining.

## Retry decisions

| Lost or failed step | Inventory behavior |
| --- | --- |
| Create response | Replay the same logical create; `jobId` is already known |
| Acquire response | Replay the same acquire idempotency key |
| DoPut response | Reconcile input receipts, then retry the exact payload only if absent |
| Close response | Replay the same close idempotency key and summary |
| Cancel response | Replay the same cancel idempotency key and current fence |
| Status/list/model describe | Retry as a read-only request |
| GetFlightInfo/ticket expiry | Obtain a new ticket after terminal success |
| DoGet interruption | Obtain a fresh ticket and restart that output download |
| `STALE_FENCE` | Stop mutations from the old claim; acquire only through lease takeover |

`requestId` may change between transport attempts. Stable job, payload,
ownership and idempotency identities must not.

## Cutover checklist

1. Stop Inventory v2 workers and Transformer v2.
2. Back up PostgreSQL and apply Transformer migration `0004`.
3. Deploy Transformer advertising exactly `[3]` and verify authenticated
   capabilities and health.
4. Deploy Inventory requiring v3 with no fallback.
5. Verify create replay, ownership takeover, fenced DoPut, revision pagination,
   streaming fit, EOF close and terminal prediction download.
6. Retrain or explicitly certify any legacy model required by v3 traffic.

Do not run Inventory v2 and v3 against the same Transformer database during
the cutover.
