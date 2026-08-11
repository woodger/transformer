# ADR 0005: Durable streaming Flight v3

- Status: accepted
- Date: 2026-08-10
- Supersedes: Flight v2 lifecycle decisions in ADR 0003 and the worker v1
  process contract in ADR 0004

## Context

Flight v2 accepts a set of durable `DoPut` payloads, seals their complete
manifest and starts one worker only after the whole input is known. This closes
transport retry windows, but it leaves two system-level gaps for Inventory:

- a create response can be lost after Transformer commits a server-generated
  `jobId` and before Inventory persists it;
- an Inventory PostgreSQL lease does not fence a late mutation from the former
  owner at the Transformer boundary.

The sealed worker manifest also makes training strictly batch-oriented. It
cannot start epoch zero after the first durable payload while later payloads
continue to arrive. A durable streaming protocol must not bind a job to one
Flight connection: every `DoPut` is an independently committed application
payload, and a worker can be replaced without losing committed inputs.

The protocol is a breaking boundary. Keeping any v2 compatibility surface in
the v3 production runtime would make state, recovery and ownership semantics
ambiguous.

## Decision

Transformer Flight v3 is a durable application-level streaming job protocol.
It does not use `DoExchange`, and a job is not owned by one network connection.
Each successful `DoPut` durably publishes one immutable semantic payload and
commits its receipt in PostgreSQL before the server returns `PutResult`.

The target fit flow is:

```text
job.create
-> input OPEN / execution WAITING_INPUT
-> first non-empty contiguous committed input
-> execution QUEUED
-> worker RUNNING while input remains OPEN
-> input.close commits EOF and an immutable manifest
-> worker completes epoch 0
-> later epochs replay the complete durable dataset
-> SUCCEEDED atomically publishes modelRef
```

The implementation may be delivered as a sequence of reviewable changes in a
development branch. Production is upgraded at one breaking boundary:

```text
Inventory v2 + Transformer v2
             -> atomic cutover
Inventory v3 + Transformer v3
```

The v3 runtime has no v2 compatibility surface: no v2 dispatcher, action
names, descriptor paths, schema aliases, fallback or dual protocol support.

## Public state

Input and execution are independent axes:

```text
input.state:
  OPEN | CLOSED | ABORTED

execution.state:
  WAITING_INPUT
  QUEUED
  RUNNING
  RETRYING
  CANCELLING
  SUCCEEDED
  FAILED
  CANCELLED
```

The following coupling rules are normative:

- `OPEN` moves once to `CLOSED` or `ABORTED`;
- `CLOSED` and `ABORTED` reject new uploads;
- `SUCCEEDED` requires `CLOSED` and terminal publication checks that fact in
  the same PostgreSQL transaction;
- cancellation or a non-retryable failure while input is `OPEN` moves input to
  `ABORTED`;
- a retryable failure preserves `OPEN` or `CLOSED`;
- `OPEN + RUNNING` is valid and is the normal streaming state;
- the first committed non-empty contiguous prefix moves execution from
  `WAITING_INPUT` to `QUEUED` automatically;
- `job.input.close` commits EOF; it does not start execution;
- empty fit close is rejected with `EMPTY_INPUT` and input remains `OPEN`;
- empty predict is valid. Zero payloads produce zero outputs, while a
  typed-empty input payload produces the corresponding typed-empty output.

## Public actions and output access

The closed v3 action set is:

```text
transformer.v3.capabilities
transformer.v3.health
transformer.v3.job.create
transformer.v3.job.acquire
transformer.v3.job.status
transformer.v3.job.inputs.list
transformer.v3.job.input.close
transformer.v3.job.outputs.list
transformer.v3.job.cancel
transformer.v3.model.describe
```

`job.seal` and `job.start` do not exist. `GetFlightInfo` is permitted only
after `execution.state = SUCCEEDED`; closing input is necessary but is not a
sufficient publication condition. Prediction outputs may be prepared in an
attempt workspace while input is open, but all outputs become visible in one
terminal transaction. Partial successful results are never exposed.

## Stable job identity and tombstones

Inventory creates a UUID `jobId` and commits it locally before the first
network call. `job.create` carries:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "requestId": "uuid",
  "idempotencyKey": "inventory-job-create:...",
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "operation": "fit"
}
```

Transformer atomically stores the owner, canonical create-request hash,
initial external ownership and the idempotent result. The compact identity is
retained after large job artifacts and mutable job rows are retired. A repeat
with the same owner and request hash resolves to the retained identity; a
different create request cannot reuse the `jobId`. Responses for another owner
do not disclose whether the identity exists.

This retained identity closes both the lost-create-response window and future
identity reuse. A separate `job.resolve` action is not introduced.

## Cross-system fencing

External ownership consists of:

```json
{
  "clientExecutionId": "uuid",
  "fencingToken": "7"
}
```

`fencingToken` is a positive monotonic server-issued integer encoded as a
canonical decimal string. A takeover calls `job.acquire` with the previous
owner identity, expected token and a new `clientExecutionId`. Transformer
atomically compares the current pair, increments the token and returns it.
Exact idempotent replay, including replay after a lost acquire response,
returns the already committed new ownership without incrementing again.

The current external fence is checked:

- before accepting `DoPut` data;
- again in the PostgreSQL transaction immediately before committing a
  `DoPut` receipt;
- by `job.input.close` and `job.cancel`;
- by every other public mutation.

A `DoPut` which starts before takeover but loses the second check deletes only
its own temporary or unreferenced candidate artifact and returns
`STALE_FENCE`. Read-only status, input listing, output listing and model
description do not require a fence.

The existing UUID `attemptId` remains the internal worker equality fence.
Internal attempt ownership is never transferred, so no additional internal
`attemptFence` is added. The monotonic fence exists only at the
Inventory-to-Transformer boundary.

## Durable upload and ordering

The descriptor is:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "inputs", ordinal)
```

Metadata contains the external fence and semantic identity:

```json
{
  "contract": "transformer-flight",
  "version": 3,
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "fencingToken": "8",
  "payloadId": "uuid",
  "ordinal": 0,
  "schemaId": "inventory.sequence.fit.v2",
  "dataContractSha256": "hex",
  "rows": 1820
}
```

One `DoPut` is one semantic payload. RecordBatch boundaries are transport
chunking only. Out-of-order completion is allowed, but workers receive only
the contiguous ordinal prefix. Logical order is ordinal and then row within
the payload; arrival timing never changes it.

Each upload writes a unique immutable candidate path, for example:

```text
recovery/jobs/{jobId}/inputs/{ordinal}-{payloadId}-{uploadToken}.arrow
```

No upload overwrites one shared final path for an ordinal. PostgreSQL selects
the one winning artifact reference after checking ordinal, payload identity,
the current external fence, limits and job state. A losing candidate is an
orphan eligible for reconciliation; it can never replace the winner selected
by a later owner.

`PutResult` includes the receipt plus:

```text
inputRevision
nextInputOrdinal
queued
```

`inputRevision` is a per-job monotonic commit revision. `nextInputOrdinal` is
the first missing ordinal. Duplicate worker notifications and repeated
`PutResult` recovery must not cause one ordinal to be consumed twice.

## Input close and manifest digest

`job.input.close` carries summary values instead of the complete manifest:

```json
{
  "jobId": "uuid",
  "clientExecutionId": "uuid",
  "fencingToken": "8",
  "payloadCount": 5,
  "totalRows": 9100,
  "totalBytes": 324625520,
  "manifestSha256": "hex"
}
```

Transformer verifies ordinals `0..payloadCount-1`, no gaps, counts, rows,
bytes, one physical Arrow schema and one `dataContractSha256`. The digest is
SHA-256 over the contract-defined canonical ordered list of server receipts.
Receipts are sorted by ordinal. The digest excludes arrival order, timestamps,
`commitRevision`, queue/execution state and all wall-clock data, so it is
stable for the same committed dataset.

The successful close transaction records the summary and moves input to
`CLOSED`. A repeated identical close is idempotent; a different summary is a
conflict.

## Stable pagination

`job.inputs.list` and `job.outputs.list` are paginated. Input traversal is by
monotonic `commitRevision`, not ordinal. The first request fixes the current
`inputRevision` as `snapshotRevision`; every page uses:

```text
cursor < commitRevision <= snapshotRevision
```

The field meanings are:

- `afterRevision`: watermark of the completed previous traversal;
- `snapshotRevision`: inclusive upper bound frozen for this traversal;
- `cursor`: last read `commitRevision` within this traversal.

A new traversal starts with:

```text
afterRevision = previousSnapshotRevision
snapshotRevision = currentInputRevision
cursor = afterRevision
```

Consequently an input with a low ordinal committed late is assigned a higher
`commitRevision` and cannot be skipped. Page size is bounded by the contract;
status and terminal results do not embed unbounded input or output arrays.

## Streaming fit semantics

The logical data order is:

```text
ordinal -> row within payload -> bounded shuffle window -> optimizer batch
```

Payload and RecordBatch boundaries are not optimizer-batch, shuffle-window or
epoch boundaries. Epoch zero consumes the open contiguous stream. A full
shuffle window trains immediately; at the current input frontier the worker
waits for the next ordinal. EOF flushes the last incomplete shuffle window,
and only then completes epoch zero. Epochs one and later replay the complete
closed immutable dataset from durable storage.

The service performs complete physical and value validation before committing
an input artifact and its immutable receipt. A worker attempt verifies receipt
identity, byte count and SHA-256 before first use. Later reads of the same
receipt use fast replay: the worker still parses IPC and checks the exact
physical schema and row count, but does not repeat digest or value scans on
every epoch. Closed-input replay prepares at most one CPU batch ahead while the
current batch trains. This prefetch is not used for the open epoch zero, so the
durable control-channel and EOF ordering remain synchronous.

Changing payload partitioning or upload timing must not change the ML
trajectory. The go/no-go condition is:

```text
same ordered dataset
+ seed
+ deterministic=true configuration
+ same hardware/runtime
-> same row and shuffle order
-> same optimizer steps
-> same ML state after every epoch
-> semantically equivalent checkpoint
-> same final model
```

ML metrics and optimizer steps are compared; wall-clock telemetry such as
timestamps, elapsed time and latency is excluded. Checkpoint comparison occurs
after deserialization and covers model, optimizer, scaler, training state,
RNG state, shuffle state, early-stopping state and checkpoint selection. Raw
checkpoint-file SHA-256 is not an equivalence criterion because serialization
is not required to be canonical.

## Recovery and idle timeout

There is no safe checkpoint boundary inside an open global epoch. If a worker
fails before EOF, the entire incomplete epoch zero is repeated from its start.
Committed input artifacts remain durable. Model, optimizer and random state
are restored to the beginning of the incomplete epoch, so no optimizer step is
applied twice. Once input is closed, recovery remains at complete global-epoch
boundaries.

`inputIdleTimeout` protects a GPU from an abandoned open stream. It is active
only while the worker has explicitly confirmed that it is waiting for the
next contiguous ordinal. It is reset only when that contiguous frontier
advances. An out-of-order commit does not extend it. Successful ownership
takeover provides a bounded grace period but does not turn unrelated mutations
into input activity. This timeout is independent of the hard subprocess
execution and cancellation deadlines.

## ML data contract and Arrow schemas

Inventory owns the semantic data-contract document. Create carries its stable
identity:

```json
{
  "dataContract": {
    "id": "inventory.learning-dataset",
    "version": 1,
    "dataContractSha256": "hex",
    "seqLen": 10,
    "featureDim": 891,
    "targetSchemaId": "inventory.target.v1"
  }
}
```

The canonical Inventory document includes ordered feature identities, target
semantics, normalization, missing-value policy and profile version.
Transformer stores and returns the identity and hash, but does not recreate or
interpret Inventory feature semantics. The term `dataContractSha256` is used
everywhere. Predict must present the hash certified by the selected model;
otherwise create fails with `MODEL_SCHEMA_MISMATCH` before upload.

The breaking Arrow schemas are:

```text
inventory.sequence.fit.v2
  src: FixedSizeList<Float32>[seqLen * featureDim]
  tgt: FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: FixedSizeList<Float32>[seqLen * featureDim]

transformer.prediction.v2
  <predictionColumn>: FixedSizeList<Float32>[6]
```

The fixed types define dimensions even for typed-empty payloads.

## Model lifecycle

Published `modelRef` and generation are immutable and have no automatic TTL.
`modelAlias` remains owner-scoped and is resolved atomically to one
`resolvedModelRef` during `job.create`. `predictionColumn` is a job parameter,
not a model property.

`model.describe` returns immutable model identity, generation, checkpoint
digest, model configuration and certified data-contract identity. Stable model
errors are:

- unknown owner-visible identity: `NOT_FOUND`;
- model row exists but checkpoint is absent: `MODEL_UNAVAILABLE`;
- digest or checkpoint is invalid: `MODEL_CORRUPT`;
- incompatible or uncertified data contract: `MODEL_SCHEMA_MISMATCH`.

Models created before v3 are not implicitly compatible because they do not
have a certified `dataContractSha256`. They must be retrained or certified by
an explicit auditable offline migration. Certification does not silently
rewrite historical immutable model metadata.

## Worker process contract v2

Worker v2 receives an immutable base manifest containing job, attempt, model,
training and data-contract identity plus a snapshot of the already committed
contiguous inputs. New contiguous inputs and explicit EOF are sent over a
bounded service-to-worker control channel. The worker does not query
PostgreSQL and does not watch a directory.

Every control message identifies `jobId`, numeric `attempt`, `attemptId`, a
strictly increasing message sequence and input ordinal when applicable. The
worker acknowledges the highest contiguous ordinal. Repeated control messages
are safe: already accepted inputs are validated as exact duplicates and never
applied twice. After service or worker recovery a fresh attempt receives a new
snapshot built from PostgreSQL, including `inputClosed`; notification delivery
is an optimization, not a source of truth.

All progress, checkpoint, output and terminal events continue to carry
`attempt`, `attemptId` and a per-process sequence. `attemptId` is enough to
reject late messages because ownership is never transferred within an attempt.

## Required acceptance tests

The cutover is not accepted until automated tests cover:

- first committed non-empty payload queues a worker while input remains open;
- upload timing, out-of-order completion and payload partitioning preserve
  logical row order;
- optimizer batches and shuffle windows cross payload boundaries;
- input close completes epoch zero;
- crash before EOF repeats the incomplete epoch without double optimizer
  application;
- takeover rejects late `DoPut`, close and cancel from the previous owner;
- stale `DoPut` cannot overwrite the later owner's input artifact;
- lost create and acquire responses replay to the committed identity;
- close rejects a gap, incorrect totals or incorrect digest;
- model schema mismatch is rejected before upload;
- output and model publication are impossible before input is closed;
- `GetFlightInfo` is unavailable until execution succeeds;
- pagination returns a late low ordinal in the next revision traversal;
- an out-of-order commit does not extend input idle timeout;
- a tombstoned `jobId` cannot be reused;
- notification loss after input commit and after close is recovered from
  PostgreSQL;
- duplicate worker control messages do not consume data twice;
- deterministic closed-input and delayed-streaming runs meet the semantic ML
  equivalence condition above.

Race tests use explicit synchronization at reserve, artifact publication,
fence recheck, input commit, notification and terminal publication boundaries;
wall-clock sleeps are not the arbitration mechanism.

## Cutover

The operational cutover is:

1. stop Inventory workers and Transformer v2;
2. apply the breaking Transformer PostgreSQL migration;
3. remove v2 jobs, inputs, attempts, tickets, idempotency and recovery state;
4. preserve API access tokens;
5. retain legacy model records only as uncertified until explicit offline
   certification, or retrain them;
6. deploy only Flight v3 and worker v2;
7. advertise exactly `protocolVersions: [3]`;
8. start Inventory configured to require v3 without fallback;
9. verify health, create/acquire fencing, streaming fit and terminal output
   access before restoring traffic.

Filesystem cleanup is reconciliation after the database boundary, not part of
the PostgreSQL transaction. Old and failed candidate artifacts may remain as
orphans temporarily, but PostgreSQL never references a partial file.

## Consequences

- Training can overlap durable upload without tying recovery to a network
  session.
- External ownership becomes safe for multiple Inventory workers.
- State, pagination and model compatibility become explicit public contracts.
- Worker supervision gains a bounded bidirectional application protocol and
  must recover notification loss from PostgreSQL snapshots.
- The implementation and cutover are substantially more complex than v2, and
  semantic deterministic equivalence becomes the release gate.
