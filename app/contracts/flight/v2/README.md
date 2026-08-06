# Transformer Arrow Flight contract v2

This directory is the normative, language-neutral contract between Inventory
and Transformer. Prose, JSON Schemas and golden fixtures are versioned together.

Consumer workflow and retry guidance is in
[`docs/inventory-flight-handoff.md`](../../../../docs/inventory-flight-handoff.md).
Endpoint, security and service lifecycle configuration is in
[`docs/flight-operations.md`](../../../../docs/flight-operations.md). These guides
do not override the schemas or fixtures in this directory.

## Envelope and authentication

Every action request and result is a UTF-8 JSON object containing:

```json
{"contract":"transformer-flight","version":2,"requestId":"UUID"}
```

All RPCs require `authorization: Bearer TOKEN` metadata. Mutating actions also
require `idempotencyKey`. The canonical request hash is SHA-256 of compact,
key-sorted JSON after removing `requestId` and `idempotencyKey`. An exact repeat
returns the original result; the same key with a different hash is a conflict.

## Actions

- `transformer.v2.capabilities`
- `transformer.v2.health`
- `transformer.v2.job.create`
- `transformer.v2.job.seal`
- `transformer.v2.job.start`
- `transformer.v2.job.status`
- `transformer.v2.job.cancel`

The service advertises exactly these names with `ListActions`. DoExchange and
PollFlightInfo are not part of v2.

Requests and results have distinct normative schemas. Query requests use
`query.schema.json`; create, seal, start/cancel and status requests use their
corresponding request schemas. Action results use the operation-specific
`*-result.schema.json` files. `action-result.schema.json` is only the closed
union of those seven result forms, not an extensible envelope. Golden request
and result documents are in `fixtures/json/`.

Create accepts `operation` (`fit` or `predict`) and `device` (`cpu`, `cuda` or
`auto`). Fit accepts a validated `modelLabel`, `modelConfig` and
`trainingConfig`. Predict accepts exactly one immutable `modelRef` or logical
`modelAlias`; an alias is resolved to a generation during create. No action
accepts paths, argv arrays or unrecognized fields.

## State and retry rules

```text
UPLOADING -> SEALED -> QUEUED -> RUNNING
                                  |-> SUCCEEDED
                                  |-> FAILED
                                  |-> RETRYING -> RUNNING
                                  `-> CANCELLING -> CANCELLED
```

Cancel from `UPLOADING`, `SEALED`, `QUEUED` or `RETRYING` goes directly to
`CANCELLED`. Terminal states are immutable and each mutation increments
`revision`. An interrupted fit moves to `RETRYING` and resumes at its latest
registered global-epoch checkpoint. A confirmed CUDA device loss moves either
operation to `RETRYING`; fit resumes and predict reruns from its retained
input. Other execution failures remain terminal.

## Upload

The input descriptor is:

```text
pathDescriptor("transformer", "v2", "jobs", jobId, "inputs", ordinal)
```

Application metadata is sent once, and may arrive as a metadata-only FlightData
message after the schema. The required fields are defined in
`schemas/upload-metadata.schema.json`. A zero-batch typed Arrow stream is valid.

One DoPut is one semantic frame. All RecordBatches are written to one Arrow IPC
file. Only after file fsync, atomic rename, directory fsync and ledger commit
does the server emit one PutResult. `PutResult.appMetadata` follows
`schemas/put-result.schema.json`.

Seal supplies a complete ordered manifest. Ordinals must be exactly `0..N-1`;
payload IDs, digests and the complete ledger input set must match. Exact seal is
idempotent and conflicting seal is rejected. V2 permits at most 400 payloads
per job so the complete manifest always fits the 64 KiB action-document limit.

## Arrow schemas

Fit input contains `src` and `tgt`. Predict input contains `src` only.
The `*.v1` Arrow schema IDs are retained because their data shapes did not
change; their version is independent of the Flight control-plane version.

- `src`: List, LargeList or FixedSizeList of float32/float64;
- `tgt`: List, LargeList or FixedSizeList of float32/float64, width 6;
- prediction: exactly the configured column, List<float32>, width 6.

Null lists/elements and infinities are rejected. `src` may contain NaN;
`tgt` and prediction values must be finite and fit float32. Target volatility
at index 4 is non-negative and hit probability at index 5 lies in `[0,1]`.
List width is stable across all rows/batches/payloads in a job. Typed empty
inputs and prediction outputs are supported.

## Outputs

Status supplies output descriptors. Clients never construct a ticket. They call
GetFlightInfo with:

```text
pathDescriptor("transformer", "v2", "jobs", jobId, "outputs", ordinal)
```

The returned ticket is random and opaque, scoped to subject/job/output, expires,
contains no path and is checked on every DoGet. DoGet yields IPC batches without
`read_all()`. A successful fit returns only immutable `modelRef` plus safe
checkpoint metadata; Transformer retains all filesystem ownership.
The status result schema whitelists that metadata and excludes checkpoint
paths, argv and other server-local details.

Fit status also exposes safe recovery metadata: the latest completed epoch,
retry count, previous retry reason and the generation used by the most
recently claimed attempt. It never exposes a recovery path or physical CUDA
identifier.

## Training recovery and CUDA scheduling

Fit inputs and internal recovery checkpoints are persistent server-owned
artifacts. A checkpoint is visible only after its file is durably replaced and
its metadata is atomically registered in PostgreSQL. Recovery occurs only at a
completed global-epoch boundary; an incomplete epoch is repeated.

Capabilities reports CUDA capacity from a boot-scoped physical device
inventory. Each CUDA attempt owns one device lease and its subprocess is bound
to that device. A confirmed lost device is quarantined until the next Linux
boot. The failed job is requeued and may continue on another healthy GPU; a fit
restores its latest registered epoch. A subprocess is never moved while it is
running.

## Transport and limits

Capabilities advertises application limits for message target size, batch,
logical payload, rows, payload count, total job bytes and dynamic queue
capacity. Health reports runtime and persistent recovery storage separately.
The initial client RecordBatch target is 8 MiB, the advertised message target is
16 MiB and a logical payload defaults to at most 512 MiB.

PyArrow 24's Python server binding does not expose the C++ gRPC builder hook, so
the 16 MiB value is a compatibility target rather than a server transport hard
limit. Logical payload/row/job limits are enforced by the application. See
`../../../../docs/flight-dependency-note.md` for the exact impact.

## Transport security

Bearer authentication is always required. TLS and mTLS are optional transport
settings; plaintext must be enabled explicitly. Explicit CUDA selection never
falls back to CPU regardless of transport security.
