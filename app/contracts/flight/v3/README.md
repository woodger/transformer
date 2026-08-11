# Transformer Arrow Flight contract v3

This directory is the normative language-neutral wire contract between
Inventory and Transformer. JSON Schemas, Arrow schemas and golden fixtures are
versioned together. The durability and recovery rationale is recorded in
[ADR 0005](../../../../docs/adr/0005-durable-streaming-flight-v3.md).

## Envelope and authentication

Every action request and result is a UTF-8 JSON object containing:

```json
{"contract":"transformer-flight","version":3,"requestId":"UUID"}
```

All RPCs require `authorization: Bearer TOKEN`. Mutations also carry an
`idempotencyKey`. The canonical request hash is SHA-256 of compact key-sorted
JSON after removing `requestId` and `idempotencyKey`. An exact repeat returns
the committed result; reusing a key for a different request is rejected.

## Actions

The server advertises exactly:

- `transformer.v3.capabilities`
- `transformer.v3.health`
- `transformer.v3.job.create`
- `transformer.v3.job.acquire`
- `transformer.v3.job.status`
- `transformer.v3.job.inputs.list`
- `transformer.v3.job.input.close`
- `transformer.v3.job.outputs.list`
- `transformer.v3.job.cancel`
- `transformer.v3.model.describe`

The list above is the complete action surface. `job.input.close` is the EOF
operation. DoExchange and PollFlightInfo are not part of the contract.

The request and result schemas are closed: unrecognized fields are rejected.
`action-result.schema.json` is the closed union of all action results. Golden
documents are in `fixtures/json/`.

## State

V3 exposes two independent axes:

```text
input.state     OPEN | CLOSED | ABORTED
execution.state WAITING_INPUT | QUEUED | RUNNING | RETRYING |
                CANCELLING | SUCCEEDED | FAILED | CANCELLED
```

The first committed non-empty contiguous input prefix automatically queues the
job. A worker may run while input is open. `job.input.close` commits EOF and
the immutable receipt manifest; it is not a start command. Success and public
output access require closed input.

Empty fit close returns `EMPTY_INPUT` without closing input. Empty predict is
valid: no payloads produce no outputs, while one typed-empty payload produces
one typed-empty output.

## Stable identity and fencing

Inventory generates and persists `jobId` before create. Transformer retains a
compact owner-scoped job identity after heavy job data is retired, so a lost
create response is recovered by exact replay and the UUID cannot later be
reused for a different request.

Every public mutation is fenced by `clientExecutionId` and a server-issued
monotonic `fencingToken`, encoded as a decimal string. `job.acquire` atomically
compares the previous ownership and increments the token. A DoPut checks the
fence before receiving data and again in the input-commit transaction. A stale
upload can leave only its unique unreferenced candidate; it cannot replace the
winning input artifact.

## Upload

The input descriptor is:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "inputs", ordinal)
```

Metadata follows `upload-metadata.schema.json`. One DoPut is one semantic
payload; RecordBatch boundaries are transport chunks. The server durably
publishes an immutable candidate, commits its receipt and only then sends one
PutResult. PutResult reports `inputRevision`, the first missing
`nextInputOrdinal` and whether this commit caused automatic queueing.

Out-of-order commit is allowed. Worker delivery follows only the contiguous
ordinal prefix, so arrival timing does not change logical row order.

## Close and manifest digest

Close sends constant-size counts and `manifestSha256`, not a complete array.
The digest is SHA-256 of compact key-sorted JSON for the following server
receipt fields, sorted by ordinal:

```text
payloadId, ordinal, schemaId, dataContractSha256, rows, batches, bytes,
sha256, schemaFingerprint
```

`commitRevision`, timestamps, queue state and arrival order are excluded. The
server verifies contiguous ordinals, totals, one physical Arrow schema and one
`dataContractSha256` before committing `CLOSED`.

## Pagination

Input pages are ordered by per-job `commitRevision`. A first request supplies
`afterRevision` and fixes the returned `snapshotRevision`. Later pages reuse
that snapshot and satisfy:

```text
cursor < commitRevision <= snapshotRevision
```

After completing a traversal, the next one begins with its previous
`snapshotRevision` as `afterRevision`. A low ordinal committed late therefore
receives a new revision and cannot be lost. Page size is at most 100.

Status contains only bounded summaries. Input receipts and output descriptors
are returned by their list actions.

## Arrow schemas

The physical schemas are exact:

```text
inventory.sequence.fit.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
  tgt: non-null FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]

transformer.prediction.v2
  <predictionColumn>: non-null FixedSizeList<Float32>[6]
```

Null rows/elements and infinities are rejected. `src` may contain NaN; target
and prediction values must be finite. Target volatility at index 4 is
non-negative and hit probability at index 5 is in `[0, 1]`. Fixed list widths
make a zero-batch typed-empty payload unambiguous.

## ML data contract and models

Inventory owns the semantic dataset document. Create carries its identity,
`dataContractSha256`, `seqLen`, `featureDim` and target schema identity.
Transformer stores and returns these fields without reconstructing Inventory
feature semantics. Predict create is rejected with `MODEL_SCHEMA_MISMATCH`
before upload unless the selected immutable model is certified for the same
hash.

`modelAlias` is owner-scoped and is atomically resolved to
`resolvedModelRef` during create. `model.describe` exposes immutable generation,
checkpoint digest, model configuration and data-contract identity without a
server path. Published models have no automatic TTL. Stable failures are
`NOT_FOUND`, `MODEL_UNAVAILABLE`, `MODEL_CORRUPT` and
`MODEL_SCHEMA_MISMATCH`.

## Output access

Clients obtain output descriptors from `job.outputs.list`. GetFlightInfo uses:

```text
pathDescriptor("transformer", "v3", "jobs", jobId, "outputs", ordinal)
```

It is allowed only after `execution.state = SUCCEEDED`. Tickets remain random,
opaque, owner-scoped and expiring. Prediction artifacts may be staged while
input is open, but all outputs are published together with terminal success;
partial success is never visible.

## Streaming and recovery

Epoch zero consumes the open contiguous stream. Optimizer batches and bounded
shuffle windows cross both RecordBatch and payload boundaries. At the current
frontier the worker reports that it is waiting; EOF flushes the final window
and completes the epoch. Later epochs replay the closed durable dataset.

Failure before EOF restarts the incomplete epoch from its beginning. Recovery
checkpoints remain complete-global-epoch snapshots. `inputIdleTimeout` runs
only while a worker has confirmed that it waits for the next contiguous
ordinal, and an out-of-order commit does not extend it.

For identical ordered data, seed, deterministic configuration and
hardware/runtime, delayed streaming and fully closed input must produce the
same row/shuffle order, optimizer steps, per-epoch ML state, semantic
checkpoint and final model. Wall-clock telemetry and serialized file digest
are not equivalence criteria.
