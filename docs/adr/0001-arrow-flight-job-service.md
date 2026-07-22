# ADR 0001: Arrow Flight job service boundary

- Status: accepted
- Date: 2026-07-18

## Context

Inventory and Transformer must run on different physical servers. The existing
Transformer interface is a pair of local framed Arrow subprocess protocols:
`fit-stream` and `predict-stream`. Training mutates optimizer/model state, so a
network retry after execution has begun is not safe.

## Decision

Transformer owns a single-instance Arrow Flight v1 service, a SQLite/WAL job
ledger, a durable filesystem spool and a worker scheduler. Flight RPC handlers
only authenticate, validate, durably stage inputs and mutate job state. They do
not execute Torch. A worker launches the existing CLI with `shell=False` in a
separate process group.

One successful DoPut is one logical Inventory payload and is persisted as one
Arrow IPC file. RecordBatch boundaries within that DoPut are transport
chunking. For prediction, the worker wraps each persisted file in exactly one
legacy 8-byte length-prefixed frame, ordered by `ordinal`. For fit, the worker
passes the durable input directory to the CLI so it can reopen each ordinal in
every job-wide epoch.

Job state is authoritative in SQLite. Files become visible only after durable
filesystem publication followed by a ledger commit. Interrupted `RUNNING` jobs
fail with `EXECUTION_INTERRUPTED`; they are never automatically retried.

Transformer owns checkpoint files. A successful fit publishes an immutable,
opaque `modelRef` and optionally advances an owner-scoped logical alias.
Network requests never contain filesystem paths or arbitrary CLI arguments.

TLS and job device selection are independent. Explicit `cuda` is checked at
create and start and never falls back to CPU. Plaintext must be enabled
explicitly.

V1 is deliberately single-instance: it has no multi-replica scheduler or
shared storage. It does not use DoExchange or PollFlightInfo.

## Consequences

- Upload/start retries are safe through canonical idempotency records.
- A lost response can be replayed without replaying Torch execution.
- Prediction uses one subprocess and one model load for all sealed inputs.
- Fit uses the durable spool as an epoch-replayable dataset: each job epoch
  visits every non-empty input by ordinal under one optimizer, loss schedule,
  checkpoint selector and early-stopping instance.
- CUDA jobs use a FIFO lane of capacity one; CPU capacity is configurable.
- SQLite and the spool must be on durable local storage and protected by a
  process-level state directory lock.
- PyArrow 24's Python server binding cannot express every desired gRPC status
  or configure a server receive-message limit. The v1 dependency note records
  the exact mapping and the application-level quota fallback.
