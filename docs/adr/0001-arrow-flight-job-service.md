# ADR 0001: Arrow Flight job service boundary

- Status: accepted
- Date: 2026-07-18

## Context

Inventory and Transformer must run on different physical servers. The existing
Transformer interface is a pair of local framed Arrow subprocess protocols:
`fit-stream` and `predict-stream`. Training mutates optimizer/model state, so a
network retry after execution has begun is not safe.

## Decision

Transformer owns a single-instance Arrow Flight v1 service, a PostgreSQL job
ledger, an ephemeral filesystem spool and a worker scheduler. Flight RPC
handlers only authenticate, validate, stage inputs and mutate job state. They
do not execute Torch. A worker launches the existing CLI with `shell=False` in
a separate process group.

One successful DoPut is one logical Inventory payload and is persisted as one
Arrow IPC file. RecordBatch boundaries within that DoPut are transport
chunking. For prediction, the worker wraps each persisted file in exactly one
legacy 8-byte length-prefixed frame, ordered by `ordinal`. For fit, the worker
passes the durable input directory to the CLI so it can reopen each ordinal in
every job-wide epoch.

PostgreSQL is authoritative for control-plane state, API access tokens and
published-model metadata. Runtime Arrow payloads, attempt output and logs live
under `/tmp/transformer`. A storage epoch binds these database rows to one
runtime filesystem generation. If that generation is lost, all jobs and their
runtime metadata are discarded instead of being resumed against missing data.
Interrupted `RUNNING` jobs fail with `EXECUTION_INTERRUPTED`; they are never
automatically retried.

Transformer owns checkpoint files. Only a successful fit atomically publishes
an immutable checkpoint and metadata below the persistent project `models/`
directory, records its opaque `modelRef` in PostgreSQL and optionally advances
an owner-scoped logical alias. Network requests never contain filesystem paths
or arbitrary CLI arguments.

API access tokens are issued and revoked through the Transformer CLI. The
Flight process loads active token digests into RAM and refreshes that cache via
PostgreSQL `LISTEN/NOTIFY`; request authentication does not query PostgreSQL.
The worker queue is likewise maintained in process memory after startup rather
than implemented as periodic database polling.

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
- PostgreSQL is the single durable source of truth; there is no second local
  database to coordinate or back up.
- Loss of `/tmp/transformer` invalidates every job, input, output ticket and
  idempotency record associated with that runtime generation. Work, including
  long-running fit, starts again as a new job.
- Published models and API tokens survive runtime loss. A `modelRef` exists
  only after its checkpoint has reached the persistent model directory and the
  corresponding PostgreSQL transaction commits.
- One process owns a runtime directory through a process-level lock. V1 remains
  single-instance even though PostgreSQL is remote.
- PyArrow 24's Python server binding cannot express every desired gRPC status
  or configure a server receive-message limit. The v1 dependency note records
  the exact mapping and the application-level quota fallback.
