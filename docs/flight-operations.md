# Transformer Arrow Flight service: runbook v2

This runbook covers the single-instance Transformer Flight service. Consumer
wire details are in the
[`Inventory handoff`](inventory-flight-handoff.md), normative schemas and
fixtures are in [`app/contracts/flight/v2`](../app/contracts/flight/v2/README.md), and
the original service boundary is recorded in
[`ADR 0001`](adr/0001-arrow-flight-job-service.md), and durable resumable
training plus device-aware execution are fixed by
[`ADR 0003`](adr/0003-durable-resumable-training-and-device-aware-execution.md).

## Runtime requirements

- Python 3.11 on Linux.
- PyTorch, NumPy and PyArrow for training and Flight.
- SQLAlchemy 2, Psycopg 3, Alembic and python-dotenv for PostgreSQL access.
- PostgreSQL reachable on the private network.
- A persistent project `models/` directory for successfully published models.
- A persistent project `recovery/` directory for fit inputs and internal
  global-epoch checkpoints.
- A RAM-backed `/tmp` large enough for prediction inputs and active attempt
  artifacts.
- A visible Linux `/proc`, libc `prctl(PR_SET_PDEATHSIG)` support and permission
  to signal worker-owned process groups.
- For CUDA scheduling, `nvidia-smi` with stable GPU UUID output. Each visible
  GPU must be usable by the service user.

Production Python package versions are fixed only in
[`requirements.txt`](../requirements.txt). The environment is created on the
target host according to
[`deployment/systemd.md`](deployment/systemd.md). Commands in this runbook are
executed from `/opt/transformer` through
`./.venv/bin/python`.

```bash
cd /opt/transformer
```

Each training or prediction subprocess starts in an isolated process group.
Startup recovery compares the recorded PID, process group, boot ID and process
start ticks before signalling an interrupted group. If identity cannot be
proved safely, startup fails instead of risking a signal to a reused PID.

## Storage and source-of-truth boundaries

PostgreSQL is the single durable source of truth for:

- jobs, attempts and state transitions;
- input/output metadata, idempotency records and output tickets;
- registered training-recovery generations and retry history;
- published-model metadata and owner-scoped model aliases;
- API access tokens;
- the current runtime storage epoch.

The runtime filesystem is intentionally ephemeral:

```text
/tmp/transformer/
  service.lock
  storage-epoch
  cuda-quarantine.json
  spool/
    jobs/{jobId}/
      inputs/{ordinal}.arrow        # prediction only
      attempts/{attempt}/
        metrics.jsonl
        stdout.log
        stderr.log
        outputs/{ordinal}.arrow
        checkpoint.pth
```

Fit inputs and recoverable training state are persistent but remain internal:

```text
<project-root>/recovery/
  jobs/{jobId}/
    inputs/{ordinal}.arrow
    checkpoints/{completedEpoch}.pth
```

Only successfully trained models receive a persistent public identity:

```text
<project-root>/models/
  {modelRef}/
    checkpoint.pth
    metadata.json
```

Files are staged beside their destination, fsynced, atomically renamed and
followed by a directory fsync. A recovery checkpoint becomes visible only
after the file is durable and its generation is registered in PostgreSQL.
Model publication copies a successful attempt checkpoint into `models/` first
and commits its metadata to PostgreSQL only after the filesystem publication
succeeds. Failed and interrupted attempts never create a model generation.

One process owns both runtime and recovery directories through non-blocking
`service.lock` files. V2 remains single-instance: PostgreSQL does not turn the
in-memory worker queue or local stores into a multi-replica scheduler.

### Loss of `/tmp`

`storage-epoch` identifies the current runtime filesystem generation. If
`/tmp/transformer` is lost, the next process creates a new epoch. Transformer
removes prediction jobs, their inputs/outputs/tickets and linked idempotency
records because those artifacts cannot be reconstructed. A non-terminal fit
whose inputs are in `recovery/` remains authoritative: an interrupted attempt
becomes `RETRYING` and resumes from its latest registered completed epoch.

Published models, persistent fit inputs/checkpoints and API access tokens are
not tied to the runtime epoch. Loss of `recovery/` is different: a registered
input or checkpoint which is absent or corrupt produces an explicit recovery
error; Transformer never silently restarts the fit from epoch zero. PostgreSQL
stores metadata only, so neither filesystem can be reconstructed from the
database.

## PostgreSQL configuration and migrations

Database settings are read from `<project-root>/.env`. Values already present
in the process environment take precedence. The required settings are:

```dotenv
POSTGRES_HOST=10.20.30.10
POSTGRES_PORT=5432
POSTGRES_DB=transformer
POSTGRES_USER=transformer
POSTGRES_PASSWORD=replace-with-a-secret
```

`POSTGRES_PORT` defaults to `5432` when omitted. The database is expected to
remain on the private network. Credentials must not be committed to the
repository or included in logs.

Transformer uses the `transformer` PostgreSQL schema. The service never applies
migrations at startup. Inspect and update it explicitly:

```bash
./.venv/bin/python ./app/main.py db migrations status
./.venv/bin/python ./app/main.py db migrations apply
```

`status` is read-only. `apply` upgrades to the current Alembic head. To revert
exactly the latest applied revision:

```bash
./.venv/bin/python ./app/main.py db migrations rollback
```

The service and token-management commands refuse to start against a missing or
outdated schema and direct the operator to `db migrations apply`.

Revision `0002` is the deliberate breaking Flight v2 cutover. Applying it
removes v1 jobs, attempts, input/output tickets and idempotency records. It
preserves published model generations, aliases and access tokens. Stop the v1
service before applying the revision and start only v2 code afterwards; mixed
v1/v2 operation is unsupported.

PostgreSQL stores control-plane state, not Arrow payloads and not a local cache.
Transactions are short. Worker dispatch uses an in-process FIFO initialized
from queued rows at startup, so idle workers do not poll the database.

## API access tokens

Bearer authentication is required for every Flight RPC, including actions,
DoPut, GetFlightInfo and DoGet. Issue a token for a local service identity:

```bash
./.venv/bin/python ./app/main.py auth tokens issue --subject=inventory-production
```

The command prints the token ID, subject and newly generated credential. The
credential has the form `a.<base64url>` and is stored in PostgreSQL exactly in
that form. Deliver it through the deployment's secret channel; do not place it
in command history, logs or the repository.

List metadata without revealing credentials:

```bash
./.venv/bin/python ./app/main.py auth tokens list
```

Revoke by token ID:

```bash
./.venv/bin/python ./app/main.py auth tokens revoke 35dc6236-cfb9-4ac7-80db-320db21ef463
```

The Flight process builds an immutable SHA-256 digest index of active tokens in
RAM. Authentication computes the supplied credential digest and consults that
index; it does not query PostgreSQL on the RPC path. PostgreSQL
`LISTEN/NOTIFY` triggers a full cache refresh after issue or revoke. A listener
reconnect also reloads the complete active set, so PostgreSQL remains the only
source of truth.

## Flight service configuration

Service configuration precedence, from lowest to highest, is:

1. settings in `app/config.py` and remaining built-in defaults;
2. supported `TRANSFORMER_*` environment variables;
3. explicitly supplied `flight serve` options.

The CLI exposes only endpoint and transport overrides:

```text
--host
--port
--allow-plaintext
--tls-cert-file
--tls-key-file
--tls-ca-file
--tls-require-client-cert
```

The following service settings are configured in `app/config.py`, not through
the environment:

| Python setting | Default | Notes |
| --- | --- | --- |
| `HOST_DEFAULT` | `127.0.0.1` | Flight listen host |
| `PORT_DEFAULT` | `8815` | Flight listen port; `0` is accepted for tests |
| `ALLOW_PLAINTEXT` | `true` | Allow serving without TLS |
| `CPU_WORKERS` | `2` | Concurrent CPU worker lanes |
| `RETENTION_SECONDS` | `604800` | Terminal-job retention |

The runtime directory is derived with
`os.path.join(tempfile.gettempdir(), PROJECT_NAME)`. It resolves to
`/tmp/transformer` in the target systemd environment.

The corresponding `TRANSFORMER_*` environment variables are not read.
TLS and mTLS have no persistent configuration defaults: they are enabled only
by explicitly supplying certificate options to `flight serve`.
Flight v2 derives `cudaCapacity` from the healthy physical GPUs discovered at
startup; it is not an application setting.

Certificate and key must be configured together. `tls-require-client-cert`
also requires a CA file. Plaintext transport is accepted only when explicitly
enabled; bearer authentication remains mandatory in every transport mode.

### Quotas and interoperability targets

| Environment variable | Default | Notes |
| --- | --- | --- |
| `TRANSFORMER_MAX_MESSAGE_BYTES` | `16777216` | Client interoperability target |
| `TRANSFORMER_TARGET_BATCH_BYTES` | `8388608` | Recommended producer RecordBatch size |
| `TRANSFORMER_MAX_BATCH_BYTES` | `16777216` | Application RecordBatch limit |
| `TRANSFORMER_MAX_PAYLOAD_BYTES` | `536870912` | Logical DoPut and persisted IPC-file limit |
| `TRANSFORMER_MAX_ROWS_PER_PAYLOAD` | `2000000` | Rows in one DoPut |
| `TRANSFORMER_MAX_PAYLOADS_PER_JOB` | `400` | Logical payloads in one job |
| `TRANSFORMER_MAX_JOB_BYTES` | `68719476736` | Total committed input bytes per job |
| `TRANSFORMER_MAX_ACTIVE_JOBS_PER_SUBJECT` | `32` | Non-terminal jobs per subject |

The validated ordering is
`targetBatchBytes <= maxBatchBytes <= maxMessageBytes <= maxPayloadBytes`.
Inventory should discover effective values through capabilities instead of
copying defaults.

### Lifecycle policy

| Environment variable | Default | Notes |
| --- | --- | --- |
| `TRANSFORMER_TICKET_TTL_SECONDS` | `600` | Opaque DoGet ticket lifetime |
| `TRANSFORMER_CANCEL_GRACE_SECONDS` | `10.0` | SIGTERM grace before SIGKILL |
| `TRANSFORMER_SHUTDOWN_DRAIN_SECONDS` | `30.0` | Worker drain before forced cancellation |
| `TRANSFORMER_SUBPROCESS_TIMEOUT_SECONDS` | `86400.0` | Hard CLI execution deadline |
| `TRANSFORMER_MAINTENANCE_INTERVAL_SECONDS` | `60` | Maintenance interval |

All quotas, capacities and intervals must be positive. Only the service port
may be zero.

## Manual foreground start

Production startup is defined only in
[`deployment/systemd.md`](deployment/systemd.md). For foreground diagnostics,
a local plaintext process can be started with:

```bash
./.venv/bin/python ./app/main.py flight serve \
  --allow-plaintext \
  --host=127.0.0.1 \
  --port=8815
```

For a TLS endpoint:

```bash
./.venv/bin/python ./app/main.py flight serve \
  --host=0.0.0.0 \
  --port=8815 \
  --tls-cert-file=/run/secrets/transformer/tls.crt \
  --tls-key-file=/run/secrets/transformer/tls.key
```

Add `--tls-ca-file` and `--tls-require-client-cert` when client certificates
are required. Ensure the server certificate SAN matches the address used by
Inventory.

The process writes structured JSON logs to stderr. A deployment supervisor
must forward SIGTERM, allow at least `shutdownDrainSeconds +
cancelGraceSeconds` before an external SIGKILL, and never start two processes
against the same runtime directory. The target Fedora systemd unit, runtime
directory policy and operator procedure are documented in
[`deployment/systemd.md`](deployment/systemd.md).

## Startup and recovery

Startup:

1. locks the runtime and recovery directories;
2. verifies that the PostgreSQL schema is at the current Alembic head;
3. identifies and safely terminates exact surviving worker process groups;
4. removes crash-left temporary files after all surviving workers are reaped;
5. reconciles a changed runtime epoch, discarding jobs whose required runtime
   artifacts no longer exist;
6. moves interrupted persistent fits to `RETRYING`, marks interrupted
   predictions `FAILED / EXECUTION_INTERRUPTED`, and finishes interrupted
   `CANCELLING` jobs as `CANCELLED`;
7. removes incomplete upload reservations and unpublished/orphan artifacts;
8. reconciles persistent model directories against PostgreSQL metadata;
9. inventories usable physical CUDA devices without initializing CUDA in the
   Flight process;
10. loads the API token cache and starts its notification listener;
11. loads `QUEUED` and `RETRYING` jobs into the in-memory device queues and
    starts workers plus maintenance;
12. begins Flight RPC serving.

A resumed fit restores the latest PostgreSQL-registered global-epoch
checkpoint. If none exists, it restarts from epoch zero using the same
persistent inputs and immutable job configuration. The incomplete epoch, if
any, is deliberately repeated.

## Cancellation and shutdown

Job cancellation behavior:

- `UPLOADING`, `SEALED`, `QUEUED`, `RETRYING` become `CANCELLED`
  transactionally;
- `RUNNING` becomes `CANCELLING`, then the complete worker process group is
  sent SIGTERM and, after the configured grace period, SIGKILL if necessary;
- no prediction output or model is published after cancellation wins the final
  transaction race;
- terminal state and published artifacts win only when final publication
  committed first.

On SIGINT/SIGTERM the process closes the queue-claim boundary, marks itself
draining, stops accepting RPC work, waits for running work and cancels any
remaining worker groups. Maintenance and the token listener stop before the
PostgreSQL connection pool closes and the runtime lock is released.

## Retention and storage failures

Maintenance periodically deletes expired output tickets and eligible terminal
jobs. A job is retained while it has a live ticket, a recent linked idempotency
record or immutable model provenance. Recovery inputs/checkpoints are removed
after a fit becomes terminal; published model directories are not deleted by
job retention, and v2 has no network action for model deletion.

The service does not apply a configured free-space admission watermark.
Health reports current free bytes for runtime and recovery storage without
deriving readiness from them. Actual filesystem exhaustion uses stable
`DISK_FULL` and does not publish partial artifacts.

## Health and observability

The service has no separate unauthenticated HTTP health endpoint. Call the
authenticated `transformer.v2.health` Flight action.

- `live=true` means the process can answer the action.
- `ready=true` requires a non-draining service and a successful PostgreSQL
  health check.
- CUDA availability, physical-device count and quarantine count are reported
  independently.

Service logs are one JSON object per line on stderr. They cover service
lifecycle, RPC/action completion, job transitions, input commits, worker
execution, publication, runtime resets, recovery and maintenance. Bearer
credentials, authorization headers, output tickets, filesystem paths and
worker argv are not logged.

The health response also exposes bounded in-process aggregate metrics. Metrics
reset on service restart and are not a durable accounting source; PostgreSQL
timestamps remain available for incident analysis while the corresponding job
belongs to the active runtime generation.

Recommended alerts include:

- `ready=false` or PostgreSQL unavailable;
- filesystem exhaustion reported as `DISK_FULL`;
- persistent recovery/model growth outside forecast;
- worker-lane errors, CUDA OOM, quarantined devices or repeated subprocess
  failure/retry;
- interrupted-process recovery or a runtime storage epoch reset;
- long queue wait relative to configured CPU/CUDA capacity.

## Stable errors and PyArrow limitations

The service fails an RPC instead of returning an error result. Stable codes are
included in safe error text and terminal status. Raw tracebacks, filesystem
paths, credentials and subprocess stderr must not reach clients.

PyArrow 24 has two confirmed binding limitations:

1. Python `FlightServerBase` cannot emit exact gRPC `ALREADY_EXISTS`,
   `FAILED_PRECONDITION` or `RESOURCE_EXHAUSTED`; the service preserves its
   stable application code in safe text.
2. Python `FlightServerBase` cannot configure a hard server receive-message
   limit. Per-batch, logical-payload, row and job limits remain
   application-enforced.

Details are recorded in
[`flight-dependency-note.md`](flight-dependency-note.md).

## Known v2 limits

- One Transformer service instance with one local runtime store and one
  persistent recovery store.
- No replica scheduling or automatic failover.
- No `DoExchange` and no `PollFlightInfo`.
- At most 400 logical payloads per job so the seal manifest remains within the
  action-document limit.
- One DoPut is one semantic frame; RecordBatch chunking never changes training
  epochs or prediction invocation boundaries.
- One predict job loads one checkpoint once and emits one output per input
  ordinal.
- Fit recovery is only at a completed global-epoch boundary; an incomplete
  epoch is repeated.
- One fit attempt uses one GPU; a single job is not distributed across GPUs.
- A confirmed lost GPU is quarantined until the next Linux boot; an ordinary
  service restart does not return it to the pool.
- Loss of runtime storage invalidates prediction work but not a fit with intact
  persistent recovery artifacts.
- Transformer owns checkpoints; clients receive only opaque `modelRef` values.
- Output tickets are short-lived and are not model references.
- Plaintext availability is controlled by `ALLOW_PLAINTEXT` in
  `app/config.py`; `--allow-plaintext` can enable it for one process.
- Node-to-PyArrow interoperability and physical CUDA behavior require separate
  target-environment validation.
