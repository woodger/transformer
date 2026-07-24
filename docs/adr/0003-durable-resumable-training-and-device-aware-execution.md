# ADR 0003: Durable resumable training and device-aware execution

- Status: accepted
- Date: 2026-07-24
- Supersedes: recovery, runtime-storage and CUDA scheduling decisions in ADR 0001

## Context

Flight v1 intentionally tied jobs and Arrow inputs to the ephemeral
`/tmp/transformer` storage epoch. A service or host restart therefore failed an
active fit even when many hours of training had already completed. The worker
also exposed one logical CUDA lane and did not bind a claimed attempt to a
specific physical GPU.

The training path already owns one optimizer, loss schedule, checkpoint
selector and early-stopping lifecycle across all payloads. That state can be
captured at a completed global-epoch boundary without loading the complete
dataset into memory.

## Decision

Flight v2 replaces v1. Runtime compatibility with v1 actions, descriptors and
job records is not provided.

Committed fit inputs and internal training-recovery checkpoints live in the
persistent project `recovery/` directory. PostgreSQL remains authoritative for
their visibility and for job, attempt and retry state. A file which has not
been registered by an atomic Ledger operation is an orphan. PostgreSQL does
not store Arrow payloads or checkpoint blobs.

The first recovery format captures state only after a complete global epoch.
It contains current model, optimizer, AMP scaler, training progress,
early-stopping state, best-checkpoint selection and random-generator state.
Resume therefore loses at most the incomplete epoch. Recovery checkpoints are
internal artifacts and never receive a `modelRef`; only a successful fit
publishes an immutable model below `models/`.

An interrupted or device-lost fit closes its active attempt and moves to
`RETRYING`. A later attempt starts from the latest valid checkpoint registered
in PostgreSQL, or from epoch zero when no checkpoint has ever been registered.
A registered checkpoint which is missing, corrupt or incompatible is an
explicit recovery failure rather than permission to silently restart training.

CUDA scheduling uses a boot-scoped inventory of physical devices. A CUDA
attempt receives one server-owned device lease and its subprocess is bound
through `CUDA_VISIBLE_DEVICES`. A confirmed device loss quarantines that
device for the remainder of the current boot. The running process is never
moved between devices: it is reaped and a new attempt is created. A fit remains
`RETRYING` when no healthy CUDA device remains and can continue after a reboot
rebuilds the inventory.

Retry is allowed only for service or host interruption and confirmed device
loss. Cancellation, invalid input, incompatible recovery data, CUDA
out-of-memory, ordinary subprocess failure, malformed output and disk
exhaustion are not retried automatically.

## Storage layout

```text
<project-root>/recovery/
  jobs/{jobId}/
    inputs/{ordinal}.arrow
    checkpoints/{generation}.pth

/tmp/transformer/
  service.lock
  storage-epoch
  cuda-quarantine.json
  spool/jobs/{jobId}/attempts/{attempt}/

<project-root>/models/
  {modelRef}/
    checkpoint.pth
    metadata.json
```

Persistent inputs are read directly. The operating-system page cache is the
initial RAM acceleration mechanism; v2 does not add a second tmpfs copy or a
cache-coherency protocol.

## Public contract

Flight v2 adds the `RETRYING` state, safe recovery progress, dynamic CUDA
capacity and quarantine counts. It does not expose filesystem paths, worker
arguments or physical GPU identifiers. Fit is always resumable and has no
client-controlled recovery switch.

## Non-goals

- runtime compatibility with Flight v1;
- mid-epoch or individual-batch resume;
- one job distributed over multiple GPUs;
- a generic storage or repository abstraction;
- CUDA initialization in the Flight service process;
- automatic CPU fallback for a CUDA attempt;
- storing payloads or checkpoints in PostgreSQL;
- returning a quarantined GPU to service before reboot.

## Consequences

- A complete loss of `/tmp/transformer` no longer invalidates resumable fit
  jobs whose persistent inputs and registered checkpoint remain valid.
- Recovery storage needs explicit capacity monitoring and cleanup.
- Checkpoint files are larger because optimizer, best-model and random state
  are part of recovery.
- Resuming on a different GPU preserves logical training state but does not
  promise bit-for-bit equality for nondeterministic CUDA kernels.
- The v2 cutover invalidates v1 runtime jobs, attempts, tickets and
  idempotency records while preserving published models, aliases and access
  tokens.
