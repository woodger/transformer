# ADR 0004: Clean Architecture process boundaries

- Status: accepted
- Date: 2026-08-06
- Supersedes: package-boundary decisions in ADR 0002

## Context

Transformer contains three independently started processes: the Arrow Flight
service, a short-lived ML worker for one execution attempt, and the
administrative CLI. The earlier Clean Architecture-light refactoring separated
durable upload, execution planning, subprocess supervision, artifact
publication, state policies and PostgreSQL use-case slices, but kept all of
them below the technical `app.flight` package.

Flight is an inbound transport, not the owner of job lifecycle, PostgreSQL,
artifact storage, scheduling or ML execution. The service and ML worker also
have a real process boundary: importing worker implementation into the service
would initialize or couple it to PyArrow/Torch concerns which the service does
not own.

## Decision

Transformer adopts Clean Architecture independently for each runtime process.

```text
service/bootstrap -> inbound Flight + outbound adapters + application/domain
worker/bootstrap  -> worker application + Arrow/Torch/checkpoint runtime
admin/bootstrap   -> CLI + required application use cases and PostgreSQL
```

There is no global bootstrap importing every implementation. The top-level CLI
dispatcher selects one bootstrap lazily.

The service dependency rule is:

```text
Flight adapter -> service application -> service domain
outbound adapters -> application-owned capability ports
bootstrap -> all service implementations
```

Application ports are named after capabilities: `JobRepository`,
`ArtifactPublisher`, `ExecutionPlanBuilder`, `AttemptProcess`,
`WorkerExecutor`, `WorkerCapabilities` and `DeviceLeaseManager`. They are not
named after PostgreSQL, filesystems or CUDA. A running process handle is not
exposed because `AttemptProcess` owns the child from spawn through final reap.
No generic repository, unit-of-work facade or DI container is introduced.

The service does not import worker implementation. Both processes may depend
on the neutral, independently versioned `app.contracts.worker.v1`. The public
Flight v2 contract and internal worker v1 contract evolve independently.

## Attempt identity and ownership

Every PostgreSQL claim increments the public numeric attempt and creates a new
UUID `attemptId` in the same transaction. One attempt starts at most one worker
subprocess, and ownership is never transferred while its `attemptId` remains
active. Service recovery closes an interrupted attempt instead of reattaching
its process. A permitted retry receives a new ordinal and `attemptId`.

`attemptId` is therefore the equality fence. Application mutations atomically
verify the job, active attempt identity and the exact states permitted for that
mutation. A second random token would duplicate this lifecycle. A monotonic
fence epoch is deferred unless a future design allows ownership transfer
without creating a new attempt.

## Process and artifact protocol

One worker process handles one execution attempt. A service-owned immutable
JSON manifest carries configuration and managed paths. Arrow IPC is used only
for bulk input/output artifacts; bounded versioned JSON events carry lifecycle
and progress; stderr carries diagnostics; process exit is only a terminal
transport fact.

The worker writes only to its attempt workspace. It never publishes a public
output, recovery generation, immutable model generation or `modelRef`.
Publication is staged:

```text
worker temporary write
-> close/fsync
-> atomic rename inside attempt workspace
-> event with size and digest
-> service validation
-> service-owned immutable publication and fsync
-> PostgreSQL reference/state transaction
```

PostgreSQL cannot atomically include filesystem or subprocess state. A crash
may leave an unreferenced staged or immutable artifact. Reconciliation removes
such orphans. PostgreSQL must never reference an incomplete artifact.

## Ownership

| Resource | Owner |
| --- | --- |
| job state, revision and idempotency result | PostgreSQL/service |
| active attempt ordinal and `attemptId` | PostgreSQL/service |
| process handle and pipes | worker-process adapter |
| committed Arrow inputs | service artifact storage |
| temporary checkpoint and result files | attempt workspace/worker |
| durable recovery checkpoint publication | service |
| immutable output/model publication | service |
| training and inference | worker |
| public Flight response | service inbound adapter |

## Non-goals

- changing the public Flight v2 behavior as part of package migration;
- importing or initializing Torch/CUDA in the service process;
- a permanent worker pool before profiling demonstrates a need;
- worker access to PostgreSQL, bearer authentication or public job lifecycle;
- a transaction abstraction pretending PostgreSQL includes filesystem state;
- ports around Torch tensors or other abstractions without a real consumer;
- dual-write or parallel authoritative implementations during migration.

## Consequences

- each process has a bounded import graph and owns only its resources;
- worker crashes and CUDA memory are isolated to one attempt;
- service restart never trusts late events from an old equality fence;
- artifact publication and reconciliation become explicit application
  lifecycle rather than hidden filesystem side effects;
- the migration must preserve PostgreSQL race arbitration and perform one
  deployable boundary change at a time.
