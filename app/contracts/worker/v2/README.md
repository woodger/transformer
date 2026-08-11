# Transformer worker process contract v2

This directory is the normative internal contract between the Transformer
service and one short-lived ML worker attempt. It is independent of the public
Flight contract, although this revision maps the Flight v3 Arrow schema IDs.

## Invocation and identity

The service creates and durably closes one immutable startup manifest before
spawning without shell interpretation:

```text
transformer-worker run
  --contract-version=2
  --job-id=<uuid>
  --attempt=<positive-integer>
  --attempt-id=<uuid>
  --manifest=<service-controlled-path>
```

The worker validates argv, the complete manifest and each referenced artifact.
`attemptId` is the internal equality fence and is never transferred. Recovery
creates a new attempt and a new `attemptId`; no second internal fence exists.

Capabilities are inspected through the same executable:

```text
transformer-worker inspect --contract-version=2
```

## Channels

| Channel | Contract |
| --- | --- |
| startup | immutable `command-manifest.schema.json` |
| new input and EOF | bounded NDJSON `control-message.schema.json` on stdin |
| progress and lifecycle | bounded NDJSON `event.schema.json` on stdout |
| bulk data | immutable Arrow IPC artifacts referenced by manifests |
| diagnostics | stderr |
| terminal transport | process exit status |

The worker never queries PostgreSQL and never watches a directory. The startup
manifest contains the current contiguous input snapshot and `inputClosed`.
The service sends later contiguous inputs and explicit EOF through the control
channel. PostgreSQL remains authoritative; a replacement attempt receives a
fresh snapshot, so notification loss cannot lose committed data.

Control and event messages carry `jobId`, numeric `attempt`, `attemptId` and a
strict per-direction sequence. The worker tracks the next ordinal and emits
`input.ack` after accepting an exact input. Repeated messages are validated as
exact duplicates and do not make training or prediction consume data twice.
At the current open frontier it emits `input.waiting`; that event is the only
condition which activates the service input-idle timer.

## Streaming semantics

For fit, epoch zero reads the startup inputs and later control messages as one
ordered stream. RecordBatch and payload boundaries are not optimizer-batch,
shuffle-window or epoch boundaries. EOF flushes the final incomplete shuffle
window and completes epoch zero. Later epochs reread the complete immutable
input set.

The worker does not publish an epoch-zero recovery checkpoint before EOF. If it
fails while input is open, a new attempt repeats that incomplete epoch from its
beginning. After EOF, checkpoints remain complete-global-epoch snapshots.

Predict may build attempt-local outputs as inputs arrive, including typed-empty
outputs. Its result manifest is emitted only after EOF. The service publishes
all outputs in one terminal transaction.

## Artifact lifecycle and exit semantics

The worker writes only inside its attempt workspace. It closes and fsyncs every
artifact before referring to it in an event or result manifest. It cannot
publish public output, recovery generation, model generation or `modelRef`.

The service validates, durably publishes and then records each service-owned
artifact. A crash may leave an unreferenced staged file, never a PostgreSQL
record pointing to a partial file.

- `completed`, a valid result manifest and exit `0` are all required for
  publication;
- exit `0` without exactly one `completed` event is a protocol violation;
- `completed` followed by non-zero exit is a protocol violation;
- non-zero exit without a valid safe error is `SUBPROCESS_FAILED`;
- failure to stop by the service deadline is `SUBPROCESS_HUNG`;
- late messages are rejected by `attemptId` and permitted execution state.
