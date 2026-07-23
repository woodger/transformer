# Transformer Arrow Flight v1 service runbook

This runbook covers the single-instance Transformer Flight service. Consumer
wire details are in the
[`Inventory handoff`](inventory-flight-handoff.md), normative schemas and
fixtures are in [`contracts/flight/v1`](../contracts/flight/v1/README.md), and
the service boundary is recorded in
[`ADR 0001`](adr/0001-arrow-flight-job-service.md).

## Runtime requirements

- Python 3.11 on Linux.
- PyTorch, NumPy and PyArrow for training and Flight.
- SQLAlchemy 2, Psycopg 3, Alembic and python-dotenv for PostgreSQL access.
- PostgreSQL reachable on the private network.
- A persistent project `models/` directory for successfully published models.
- A RAM-backed `/tmp` large enough for all active Arrow payloads and attempts.
- A visible Linux `/proc`, libc `prctl(PR_SET_PDEATHSIG)` support and permission
  to signal worker-owned process groups.

Each training or prediction subprocess starts in an isolated process group.
Startup recovery compares the recorded PID, process group, boot ID and process
start ticks before signalling an interrupted group. If identity cannot be
proved safely, startup fails instead of risking a signal to a reused PID.

Install the direct Python dependencies into the environment used by the
service:

```bash
python3.11 -m pip install torch numpy pyarrow \
  SQLAlchemy 'psycopg[binary]' alembic python-dotenv
python3.11 -m pip check
```

Select the CPU or CUDA PyTorch wheel appropriate for the deployment host.
CUDA availability is a runtime capability; a CPU-only installation is valid.

## Storage and source-of-truth boundaries

PostgreSQL is the single durable source of truth for:

- jobs, attempts and state transitions;
- input/output metadata, idempotency records and output tickets;
- published-model metadata and owner-scoped model aliases;
- API access tokens;
- the current runtime storage epoch.

The default runtime filesystem is intentionally ephemeral:

```text
/tmp/transformer/
  service.lock
  storage-epoch
  spool/
    jobs/{jobId}/
      inputs/{ordinal}.arrow
      attempts/{attempt}/
        metrics.jsonl
        stdout.log
        stderr.log
        outputs/{ordinal}.arrow
        checkpoint.pth
```

Only successfully trained models are published persistently:

```text
<project-root>/models/
  {modelRef}/
    checkpoint.pth
    metadata.json
```

Files are staged beside their destination, fsynced, atomically renamed and
followed by a directory fsync. Model publication copies a successful attempt
checkpoint into `models/` first and commits its metadata to PostgreSQL only
after the filesystem publication succeeds. Failed and interrupted attempts
never create a model generation.

One process owns a runtime directory through its non-blocking `service.lock`.
V1 remains single-instance: PostgreSQL does not turn the in-memory worker queue
or local runtime spool into a multi-replica scheduler.

### Loss of `/tmp`

`storage-epoch` identifies the current runtime filesystem generation. If
`/tmp/transformer` is lost, the next process creates a new epoch. Transformer
then rejects the old runtime generation and removes every job plus its attempts,
inputs, prediction outputs, tickets and idempotency records from PostgreSQL.
It does not attempt to resume or reconstruct any work. A fit that had already
run for 22 hours must be submitted and trained again as a new job.

Published models and API access tokens are not tied to the runtime epoch and
remain available. Their files and metadata are persistent. This is the only
intentional split in durability; there is no duplicated local database.

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
python3.11 ./app/main.py db migrations status
python3.11 ./app/main.py db migrations apply
```

`status` is read-only. `apply` upgrades to the current Alembic head. To revert
exactly the latest applied revision:

```bash
python3.11 ./app/main.py db migrations rollback
```

The service and token-management commands refuse to start against a missing or
outdated schema and direct the operator to `db migrations apply`.

PostgreSQL stores control-plane state, not Arrow payloads and not a local cache.
Transactions are short. Worker dispatch uses an in-process FIFO initialized
from queued rows at startup, so idle workers do not poll the database.

## API access tokens

Bearer authentication is required for every Flight RPC, including actions,
DoPut, GetFlightInfo and DoGet. Issue a token for a local service identity:

```bash
python3.11 ./app/main.py auth tokens issue --subject=inventory-production
```

The command prints the token ID, subject and newly generated credential. The
credential has the form `a.<base64url>` and is stored in PostgreSQL exactly in
that form. Deliver it through the deployment's secret channel; do not place it
in command history, logs or the repository.

List metadata without revealing credentials:

```bash
python3.11 ./app/main.py auth tokens list
```

Revoke by token ID:

```bash
python3.11 ./app/main.py auth tokens revoke 35dc6236-cfb9-4ac7-80db-320db21ef463
```

The Flight process builds an immutable SHA-256 digest index of active tokens in
RAM. Authentication computes the supplied credential digest and consults that
index; it does not query PostgreSQL on the RPC path. PostgreSQL
`LISTEN/NOTIFY` triggers a full cache refresh after issue or revoke. A listener
reconnect also reloads the complete active set, so PostgreSQL remains the only
source of truth.

## Flight service configuration

Service configuration precedence, from lowest to highest, is:

1. built-in defaults;
2. `TRANSFORMER_*` environment variables;
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

### Endpoint and transport

| Environment variable | Default | Notes |
| --- | --- | --- |
| `TRANSFORMER_RUNTIME_DIR` | `/tmp/transformer` | Ephemeral runtime spool and process lock |
| `TRANSFORMER_HOST` | `127.0.0.1` | Flight listen host |
| `TRANSFORMER_PORT` | `8815` | Flight listen port; `0` is accepted for tests |
| `TRANSFORMER_ALLOW_PLAINTEXT` | `false` | Required when TLS is absent |
| `TRANSFORMER_TLS_CERT_FILE` | unset | PEM server certificate; configure with key |
| `TRANSFORMER_TLS_KEY_FILE` | unset | PEM private key; configure with certificate |
| `TRANSFORMER_TLS_CA_FILE` | unset | Client CA for mTLS |
| `TRANSFORMER_TLS_REQUIRE_CLIENT_CERT` | `false` | Requires TLS and a client CA |

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
| `TRANSFORMER_DISK_MIN_FREE_BYTES` | `1073741824` | Runtime-spool admission watermark |

The validated ordering is
`targetBatchBytes <= maxBatchBytes <= maxMessageBytes <= maxPayloadBytes`.
Inventory should discover effective values through capabilities instead of
copying defaults.

### Worker and retention policy

| Environment variable | Default | Notes |
| --- | --- | --- |
| `TRANSFORMER_CPU_CAPACITY` | `2` | Concurrent CPU lanes |
| `TRANSFORMER_CUDA_CAPACITY` | `1` | V1 requires exactly one FIFO CUDA lane |
| `TRANSFORMER_TICKET_TTL_SECONDS` | `600` | Opaque DoGet ticket lifetime |
| `TRANSFORMER_CANCEL_GRACE_SECONDS` | `10.0` | SIGTERM grace before SIGKILL |
| `TRANSFORMER_SHUTDOWN_DRAIN_SECONDS` | `30.0` | Worker drain before forced cancellation |
| `TRANSFORMER_SUBPROCESS_TIMEOUT_SECONDS` | `86400.0` | Hard CLI execution deadline |
| `TRANSFORMER_RETENTION_SECONDS` | `604800` | Terminal-job retention |
| `TRANSFORMER_MAINTENANCE_INTERVAL_SECONDS` | `60` | Maintenance interval |

All quotas, capacities and intervals must be positive. Only the service port
may be zero.

## Start the service

Apply migrations and issue the Inventory token before the first start. A local
plaintext process can then be started with:

```bash
python3.11 ./app/main.py flight serve \
  --allow-plaintext \
  --host=127.0.0.1 \
  --port=8815
```

For a TLS endpoint:

```bash
python3.11 ./app/main.py flight serve \
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

With the same runtime storage epoch, startup:

1. locks the runtime directory and removes crash-left temporary files;
2. verifies that the PostgreSQL schema is at the current Alembic head;
3. identifies and safely terminates exact surviving worker process groups;
4. marks interrupted `RUNNING` jobs `FAILED / EXECUTION_INTERRUPTED` and
   interrupted `CANCELLING` jobs `CANCELLED`;
5. removes incomplete upload reservations and unpublished attempt artifacts;
6. reconciles persistent model directories against PostgreSQL metadata;
7. loads `QUEUED` jobs into the in-memory device queues;
8. loads the API token cache and starts its notification listener;
9. starts workers, maintenance and Flight RPC serving.

`UPLOADING`, `SEALED` and `QUEUED` jobs can survive an ordinary service restart
only while their runtime epoch and files remain present. A running fit is never
automatically retried because its optimizer and model state may already have
mutated.

If the epoch changes, the runtime-loss policy runs before ordinary job
recovery: every old job is discarded, including queued and terminal jobs.
Published models remain because their files and database records live outside
the runtime generation.

## Cancellation and shutdown

Job cancellation behavior:

- `UPLOADING`, `SEALED`, `QUEUED` become `CANCELLED` transactionally;
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

## Retention and disk policy

Maintenance periodically deletes expired output tickets and eligible terminal
jobs. A job is retained while it has a live ticket, a recent linked idempotency
record or immutable model provenance. Published model directories are not
deleted by job retention, and v1 has no network action for model deletion.

Create, upload and worker admission check the runtime filesystem free-space
watermark. Health reports `ready=false` below that watermark. Runtime disk-full
failures use stable `DISK_FULL` and do not publish partial artifacts. Monitor
both `/tmp/transformer` capacity and persistent model-directory growth.

## Health and observability

The service has no separate unauthenticated HTTP health endpoint. Call the
authenticated `transformer.v1.health` Flight action.

- `live=true` means the process can answer the action.
- `ready=true` requires a non-draining service, a successful PostgreSQL health
  check and enough free runtime storage.
- CUDA availability is reported independently.

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
- runtime free space below the watermark;
- persistent model growth outside forecast;
- worker-lane errors, CUDA OOM, device unavailability or repeated subprocess
  failure;
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
   limit. Per-batch, logical-payload, row, job and storage limits remain
   application-enforced.

Details are recorded in
[`flight-dependency-note.md`](flight-dependency-note.md).

## Known v1 limits

- One Transformer service instance and one local runtime directory.
- No replica scheduling or automatic failover.
- No `DoExchange` and no `PollFlightInfo`.
- At most 400 logical payloads per job so the seal manifest remains within the
  action-document limit.
- One DoPut is one semantic frame; RecordBatch chunking never changes training
  epochs or prediction invocation boundaries.
- One predict job loads one checkpoint once and emits one output per input
  ordinal.
- Running fit is never automatically retried.
- Loss of runtime storage invalidates all jobs in that storage epoch.
- Transformer owns checkpoints; clients receive only opaque `modelRef` values.
- Output tickets are short-lived and are not model references.
- Plaintext requires explicit `--allow-plaintext`.
- Node-to-PyArrow interoperability and physical CUDA behavior require separate
  target-environment validation.
