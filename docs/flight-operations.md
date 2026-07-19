# Transformer Arrow Flight v1 service runbook

This runbook covers the single-instance Transformer Flight service. Consumer
wire details are in the
[`Inventory handoff`](inventory-flight-handoff.md), the normative schemas and
fixtures are in [`contracts/flight/v1`](../contracts/flight/v1/README.md), and
the architectural boundary is recorded in
[`ADR 0001`](adr/0001-arrow-flight-job-service.md).

## Runtime and host requirements

- Python 3.11 on Linux.
- Direct dependencies from [`requirements.txt`](../requirements.txt):
  `numpy==2.4.5`, `pyarrow==24.0.0`, and `torch==2.12.0`.
- A durable local filesystem for the complete state directory. SQLite, input
  spool and model files must remain on the same Transformer host.
- A visible Linux `/proc` in the service PID namespace, including
  `/proc/sys/kernel/random/boot_id` and readable `/proc/<pid>/stat` entries.
- libc `prctl(PR_SET_PDEATHSIG)` support and permission to signal worker-owned
  process groups.

The Linux requirements are safety mechanisms, not optional diagnostics. Each
legacy CLI subprocess starts in an isolated session/process group behind an
internal supervisor. The supervisor uses `prctl(PR_SET_PDEATHSIG)` to kill the
group if its Flight-service parent dies. Startup recovery compares durable PID,
process-group, boot ID and process start ticks through `/proc` before signalling
an interrupted group. If identity cannot be established safely, startup fails
instead of risking a signal to a reused PID.

The Torch requirement deliberately pins the public version without a local
wheel build tag. Deployment chooses the CPU or CUDA wheel from its configured
PyTorch package index; it must not change the pinned `2.12.0` API version.
Record the selected wheel/build and validate it on the target host. A CPU-only
build is valid. Physical CUDA availability is a runtime capability, not an
installation assumption.

## Service topology and persistent state

Run exactly one service process for one state directory. A non-blocking file
lock at `<stateDir>/service.lock` rejects a second owner. V1 has no
multi-replica scheduler, shared storage or failover protocol.

Default layout:

```text
state/
  service.lock
  jobs.sqlite3
  jobs.sqlite3-wal
  jobs.sqlite3-shm
  spool/
    jobs/{jobId}/
      inputs/{ordinal}.arrow
      attempts/{attempt}/
        metrics.jsonl
        stdout.log
        stderr.log
        outputs/{ordinal}.arrow
  models/{modelRef}/
    checkpoint.pth
    metadata.json
```

SQLite uses WAL, full synchronous durability and foreign keys. Files are
staged beside their destination, fsynced, atomically renamed and followed by a
directory fsync. Network requests never provide these paths. Restrict state
directory permissions: it contains checkpoints, training metrics and worker
logs.

Use a filesystem with durable POSIX rename/fsync behavior. Do not mount the
same state directory into multiple service instances. For backup or migration,
stop the service cleanly and copy the entire state directory as one consistent
unit; copying only SQLite or only the spool cannot preserve ledger/artifact
atomicity.

## Configuration loading

Configuration precedence, from lowest to highest, is:

1. dataclass defaults;
2. JSON file passed with `--config`;
3. `TRANSFORMER_FLIGHT_*` environment variables;
4. explicitly supplied service CLI options.

JSON accepts the camelCase names below and their snake_case Python forms.
Environment names use uppercase snake case. Unknown fields, invalid numeric
types (including booleans used as integers) and inconsistent limits fail at
startup.

### Endpoint and security

| JSON field | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `stateDir` | `TRANSFORMER_FLIGHT_STATE_DIR` | `<project-root>/state` | Persistent state root |
| `bindHost` | `TRANSFORMER_FLIGHT_BIND_HOST` | `127.0.0.1` | Private-safe loopback default |
| `port` | `TRANSFORMER_FLIGHT_PORT` | `8815` | `0` is accepted for tests/dynamic binding |
| `profile` | `TRANSFORMER_FLIGHT_PROFILE` | `production` | `production`, `development`, or `lan` |
| `allowPlaintext` | `TRANSFORMER_FLIGHT_ALLOW_PLAINTEXT` | `false` | Must be explicit when TLS is absent |
| `tlsCertFile` | `TRANSFORMER_FLIGHT_TLS_CERT_FILE` | unset | PEM server certificate; configure with key |
| `tlsKeyFile` | `TRANSFORMER_FLIGHT_TLS_KEY_FILE` | unset | PEM private key; configure with certificate |
| `tlsCaFile` | `TRANSFORMER_FLIGHT_TLS_CA_FILE` | unset | Client CA for mTLS |
| `tlsRequireClientCert` | `TRANSFORMER_FLIGHT_TLS_REQUIRE_CLIENT_CERT` | `false` | Requires TLS and `tlsCaFile` |
| `bearerTokensFile` | `TRANSFORMER_FLIGHT_BEARER_TOKENS_FILE` | unset | Required secret token-to-subject JSON |

### Quotas and interoperability targets

| JSON field | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `maxMessageBytes` | `TRANSFORMER_FLIGHT_MAX_MESSAGE_BYTES` | `16777216` | Client interoperability target; see known limitation below |
| `targetBatchBytes` | `TRANSFORMER_FLIGHT_TARGET_BATCH_BYTES` | `8388608` | Recommended producer RecordBatch size |
| `maxBatchBytes` | `TRANSFORMER_FLIGHT_MAX_BATCH_BYTES` | `16777216` | Application `RecordBatch.nbytes` limit |
| `maxPayloadBytes` | `TRANSFORMER_FLIGHT_MAX_PAYLOAD_BYTES` | `536870912` | Logical DoPut and persisted IPC file limit |
| `maxRowsPerPayload` | `TRANSFORMER_FLIGHT_MAX_ROWS_PER_PAYLOAD` | `2000000` | Total rows in one DoPut |
| `maxPayloadsPerJob` | `TRANSFORMER_FLIGHT_MAX_PAYLOADS_PER_JOB` | `400` | Cannot exceed 400; seal must fit 64 KiB |
| `maxJobBytes` | `TRANSFORMER_FLIGHT_MAX_JOB_BYTES` | `68719476736` | Total committed input bytes per job |
| `maxActiveJobsPerSubject` | `TRANSFORMER_FLIGHT_MAX_ACTIVE_JOBS_PER_SUBJECT` | `32` | Non-terminal jobs per bearer subject |
| `diskMinFreeBytes` | `TRANSFORMER_FLIGHT_DISK_MIN_FREE_BYTES` | `1073741824` | Admission/readiness free-space watermark |

The validated ordering is
`targetBatchBytes <= maxBatchBytes <= maxMessageBytes <= maxPayloadBytes`.
Inventory should discover the effective values through capabilities rather
than copying defaults.

### Worker, timeout and retention policy

| JSON field | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `cpuCapacity` | `TRANSFORMER_FLIGHT_CPU_CAPACITY` | `2` | Concurrent CPU worker lanes |
| `cudaCapacity` | `TRANSFORMER_FLIGHT_CUDA_CAPACITY` | `1` | V1 requires exactly one FIFO CUDA lane |
| `queuePollMs` | `TRANSFORMER_FLIGHT_QUEUE_POLL_MS` | `100` | Durable queue polling interval |
| `ticketTtlSeconds` | `TRANSFORMER_FLIGHT_TICKET_TTL_SECONDS` | `600` | Opaque DoGet ticket lifetime |
| `cancelGraceSeconds` | `TRANSFORMER_FLIGHT_CANCEL_GRACE_SECONDS` | `10.0` | SIGTERM grace before SIGKILL |
| `shutdownDrainSeconds` | `TRANSFORMER_FLIGHT_SHUTDOWN_DRAIN_SECONDS` | `30.0` | Worker drain before forced cancellation |
| `subprocessTimeoutSeconds` | `TRANSFORMER_FLIGHT_SUBPROCESS_TIMEOUT_SECONDS` | `86400.0` | Hard legacy CLI execution deadline |
| `retentionSeconds` | `TRANSFORMER_FLIGHT_RETENTION_SECONDS` | `604800` | Terminal job retention, default seven days |
| `maintenanceIntervalSeconds` | `TRANSFORMER_FLIGHT_MAINTENANCE_INTERVAL_SECONDS` | `60` | Ticket/retention maintenance period |

Every numeric quota/capacity/interval must be positive, except the service port
may be zero. Cancellation, drain and subprocess deadlines must be positive
finite values.

The CLI exposes endpoint/security overrides only:

```text
--config
--state-dir
--bind-host
--port
--profile
--allow-plaintext
--tls-cert-file
--tls-key-file
--tls-ca-file
--tls-require-client-cert
--bearer-tokens-file
```

Use JSON or environment configuration for quotas and worker policy.

## Authentication and transport profiles

Bearer authentication is mandatory even with mTLS. The token file is a secret
JSON object in either form:

```json
{
  "a-long-random-token": "inventory-production"
}
```

or:

```json
{
  "tokens": {
    "a-long-random-token": "inventory-production"
  }
}
```

Tokens are 1-4096 printable non-whitespace ASCII characters. Subjects are
1-256 characters without control characters. The service reads the mapping at
startup; restart to rotate it. Put the file in the platform secret mount, not
the repository, and restrict it to the service account.

All methods require exactly one `authorization: Bearer TOKEN` metadata value,
including `ListActions`, every action, DoPut, GetFlightInfo and DoGet. Missing
or invalid authentication is rejected by middleware before application code.
The subject owns its jobs, model aliases/generations, outputs and tickets.

| Profile | Required server settings | Binding policy | Security use |
| --- | --- | --- | --- |
| Production TLS | `profile=production`, certificate, key, bearer file | Loopback or non-loopback | Production |
| Production mTLS | Production TLS + CA + `tlsRequireClientCert=true` | Loopback or non-loopback | Production with client certificates |
| Development plaintext | `profile=development`, `allowPlaintext=true` | Loopback only | Local development |
| LAN plaintext | `profile=lan`, `allowPlaintext=true` | Non-loopback allowed | Temporary trusted LAN only |

Production without TLS fails configuration validation. Non-loopback plaintext
requires the explicit `lan` profile. Plaintext is a temporary development/LAN
profile, not a production security profile. Certificates, private keys, client
CA and bearer credentials are read from external files and are never stored in
the repository. Restart to rotate TLS material.

TLS/mTLS configuration and job device selection are independent. Enabling,
disabling or changing TLS never changes `cpu`/`cuda`/`auto` behavior. Explicit
`cuda` never falls back to CPU.

## Production configuration example

Store this outside the repository, for example as
`/etc/transformer/flight.json`:

```json
{
  "stateDir": "/var/lib/transformer-flight",
  "bindHost": "10.20.30.40",
  "port": 8815,
  "profile": "production",
  "tlsCertFile": "/run/secrets/transformer/tls.crt",
  "tlsKeyFile": "/run/secrets/transformer/tls.key",
  "tlsCaFile": "/run/secrets/transformer/client-ca.crt",
  "tlsRequireClientCert": true,
  "bearerTokensFile": "/run/secrets/transformer/bearers.json",
  "cpuCapacity": 2,
  "cudaCapacity": 1,
  "diskMinFreeBytes": 10737418240,
  "retentionSeconds": 604800
}
```

If mTLS is not required, omit `tlsCaFile` and leave
`tlsRequireClientCert=false`; bearer authentication remains required. Ensure
the certificate SAN matches the address Inventory uses.

## Install and start

Create a Python 3.11 environment and install the pinned direct runtime
dependencies through the deployment's selected PyTorch package index:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then run the single service entrypoint from the project root:

```bash
.venv/bin/python ./app/main.py serve-flight \
  --config=/etc/transformer/flight.json
```

The process writes structured JSON logs to stderr and serves Flight on the
configured host/port. The deployment supervisor should:

- run one process as a dedicated unprivileged account;
- mount the state directory and secret files persistently;
- keep the service and its workers in the same Linux PID namespace with
  `/proc` mounted;
- forward SIGTERM and allow at least
  `shutdownDrainSeconds + cancelGraceSeconds` before external SIGKILL;
- restart on non-zero startup/serve failure, but never run two owners of one
  state directory;
- use the authenticated health action for readiness/liveness policy.

No second deployment system is required by the project; use the environment's
existing process supervisor.

## Local development server and manual probe

The following plaintext configuration is intentionally loopback-only and must
not be reused for production:

```bash
install -d -m 0700 /tmp/transformer-flight-state
umask 077
printf '%s\n' '{"dev-token":"inventory-local"}' \
  > /tmp/transformer-flight-bearers.json

.venv/bin/python ./app/main.py serve-flight \
  --state-dir=/tmp/transformer-flight-state \
  --bind-host=127.0.0.1 \
  --port=8815 \
  --profile=development \
  --allow-plaintext \
  --bearer-tokens-file=/tmp/transformer-flight-bearers.json
```

In another terminal, list actions and call capabilities/health:

```bash
.venv/bin/python - <<'PY'
import json
import uuid

import pyarrow.flight as flight

client = flight.FlightClient(("127.0.0.1", 8815))
options = flight.FlightCallOptions(headers=[
    (b"authorization", b"Bearer dev-token"),
])

print([item.type for item in client.list_actions(options=options)])
for name in ("transformer.v1.capabilities", "transformer.v1.health"):
    body = json.dumps({
        "contract": "transformer-flight",
        "version": 1,
        "requestId": str(uuid.uuid4()),
    }, separators=(",", ":")).encode()
    result = list(client.do_action(
        flight.Action(name, body), options=options
    ))
    print(name, json.loads(result[0].body.to_pybytes()))
PY
```

Expected minimum checks:

- exactly seven v1 actions are listed;
- capabilities reports protocol version 1 and the three expected schema IDs;
- health reports `live=true`; `ready` reflects ledger/disk/draining state;
- `cuda.available=false` is acceptable and must not make liveness false;
- the same call without authorization fails unauthenticated.

For an end-to-end CPU smoke check using real legacy subprocesses, run the
project integration tests rather than constructing server filesystem state by
hand:

```bash
.venv/bin/python -m pytest -q \
  tests/test_flight_fit_integration.py \
  tests/test_flight_prediction_integration.py
```

This is a Python-side check. It does not establish Node-to-PyArrow or physical
GPU interoperability.

## Health, readiness and device policy

The service has no separate unauthenticated HTTP health endpoint. Call
`transformer.v1.health` over Flight with bearer metadata.

- `live=true` means the service process can answer the action.
- `ready=true` requires: not draining, SQLite health check succeeds, and free
  space is at or above `diskMinFreeBytes`.
- CUDA unavailability is reported independently as `cuda.available=false`.
- Capabilities includes CUDA device count/runtime, CPU/CUDA queue capacity and
  current library versions.

For job selection:

- `cpu` selects CPU;
- `auto` selects CUDA when available at start, otherwise CPU;
- `cuda` fails at create when unavailable and is checked again at start;
- loss of explicit CUDA before start leaves the job `SEALED` with
  `DEVICE_UNAVAILABLE` rather than queueing it on CPU;
- a busy but available GPU means `QUEUED`; the CUDA lane remains FIFO with
  capacity one.

Physical CUDA may be absent on a valid CPU deployment. Validate a real CUDA
wheel, device, OOM handling and driver-disconnect behavior on the target GPU
host before claiming the GPU gate.

## Startup and recovery

Startup executes recovery before accepting RPCs:

1. create/lock the state directory;
2. delete crash-left sibling `.tmp` artifacts;
3. initialize/check the SQLite schema and WAL settings;
4. inspect durable active-attempt process identities and terminate only an
   exact surviving process group (SIGTERM, grace, then SIGKILL);
5. convert interrupted `RUNNING` to `FAILED / EXECUTION_INTERRUPTED` and
   interrupted `CANCELLING` to `CANCELLED`;
6. clear incomplete upload reservations and remove unpublished attempt
   checkpoints/outputs and other orphan artifacts;
7. preserve committed inputs and all durable jobs;
8. start worker lanes and periodic maintenance.

`UPLOADING`, `SEALED` and `QUEUED` survive restart. A preserved queued job can
be claimed after startup. Terminal states never change. No interrupted
`RUNNING` job is automatically retried; this is critical for fit because the
optimizer may already have mutated model state.

If startup cannot prove a recorded live process group's identity through
boot-ID/start-tick/session checks, it stops with a recovery error rather than
starting alongside a potentially mutating orphan.

## Cancellation and shutdown

Job cancellation behavior:

- `UPLOADING`, `SEALED`, `QUEUED` -> `CANCELLED` transactionally;
- `RUNNING` -> `CANCELLING`, then signal the complete worker process group;
- send SIGTERM, wait `cancelGraceSeconds`, then SIGKILL if necessary;
- if cancellation commits before final artifact publication, no output or
  checkpoint is published;
- terminal state/artifacts win only when final publication committed first.

On SIGINT/SIGTERM the application first closes the worker queue-claim boundary,
then marks itself draining, making readiness false and rejecting new create or
start requests, and shuts down the Flight server. A start already in flight may
finish as `QUEUED`, but cannot be claimed after that boundary and therefore
survives restart. The service waits up to `shutdownDrainSeconds` for work that
was already `RUNNING`; remaining workers are cancelled as process groups.
Maintenance and worker threads finish before SQLite closes and the state lock
is released.

An external supervisor must not send its own SIGKILL before the configured
drain plus cancellation grace unless host failure makes durability impossible.

## Retention and disk policy

Maintenance runs every `maintenanceIntervalSeconds`:

- expired output tickets are deleted;
- terminal jobs older than `retentionSeconds` are eligible for deletion;
- a job is retained while it has a non-expired ticket or a recent linked
  idempotency record;
- jobs that produced model generations are retained as immutable provenance;
- published model directories are not deleted by job retention;
- ledger rows are removed before their job directory; a crash-left directory
  is removed by next-start reconciliation.

There is no v1 network action to delete models. Plan storage for immutable
checkpoint growth and define any future model lifecycle as a separately
versioned contract.

Create/upload/worker admission checks the free-space watermark. Health sets
`ready=false` when it is crossed. Disk-full failures use stable `DISK_FULL` and
must not publish partial artifacts. Monitor both absolute free space and growth
of job/model directories; the watermark is protection, not a cleanup target.

## Observability

Service logs are one JSON object per line on stderr. Event families include:

- `flight.service.*`: startup recovery counts, drain, signal and stop;
- `flight.rpc.completed`: Flight method, status and latency;
- `flight.action.completed`: action name, request ID, optional job ID and
  latency;
- `flight.job.created` / `flight.job.transition`: request/job and state change;
- `flight.input.committed`: job/payload/ordinal plus rows, batches and bytes;
- `flight.worker.started`: job, attempt, selected device and worker PID;
- `flight.worker.finished` / `flight.worker.failed`: queue/run duration and
  stable result, including CUDA-unavailable/OOM classification;
- prediction-output, checkpoint-publication and DoGet events: ordinal, rows,
  batches and bytes without server filesystem paths;
- `flight.recovery.process_group`: recovery outcome without unsafe signalling;
- `flight.maintenance.*`: tickets/jobs/directories processed.

Bearer tokens, authorization headers, tickets, checkpoint paths and worker
argv are not logged. Job/payload/request IDs are correlation fields in logs;
they are not metric labels.

The authenticated health result exposes in-process aggregate metrics:

- RPC counts by bounded method/status and total RPC latency;
- upload and prediction/download bytes/rows/batches;
- created/sealed/queued/started/succeeded/failed/cancelled job counts and
  bounded state-transition counters;
- total worker queue wait/run time and lane-error counts;
- checkpoint bytes, ticket/download counts and CUDA unavailability/OOM counts;
- readiness and CUDA-availability gauges;
- disk total/used/free/watermark gauges and values in the health document.

Metrics reset on service restart and are not a durable accounting source.
Durable status timestamps allow an operator to derive queue wait
(`startedAt - queuedAt`) and execution duration (`finishedAt - startedAt`) per
incident without creating unbounded metric labels. Export JSON logs and poll
health through the deployment's existing observability stack.

Recommended alerts:

- `ready=false` or ledger unavailable;
- disk watermark exceeded or model storage growth outside forecast;
- `workerLaneErrors`, `CUDA_OUT_OF_MEMORY`, `DEVICE_UNAVAILABLE`, or repeated
  subprocess failures/hangs;
- restart recovery of any interrupted job/process group;
- long queue wait relative to configured CPU/CUDA capacity;
- state-directory lock failure or unsafe process-recovery error.

## Stable errors and PyArrow 24 limitations

The service fails an RPC instead of returning an error result. Stable codes are
included in safe error text and terminal status. Do not expose raw tracebacks,
filesystem paths, credentials or subprocess stderr to a client.

PyArrow 24 has two confirmed binding limitations:

1. Python FlightServerBase cannot emit exact gRPC `ALREADY_EXISTS`,
   `FAILED_PRECONDITION`, or `RESOURCE_EXHAUSTED`. The service uses a failing
   closest mapping and preserves the stable application code in safe text.
2. Python FlightServerBase cannot configure a hard gRPC server receive-message
   limit. The advertised 16 MiB is an interoperability target. Per-batch,
   logical-payload, row, total-job and disk limits are enforced after transport
   receives a chunk.

The full component/version, reproduction, expected/actual behavior, migration
impact and proposed upstream API are recorded in
[`flight-dependency-note.md`](flight-dependency-note.md). These are PyArrow
binding gaps, not known defects in `arrow-flight-client@0.0.8` or Inventory.
No private Cython/grpc shim is used.

## Known v1 limits

- One Transformer service instance and one local state directory.
- No shared storage, replica scheduling or automatic failover.
- No `DoExchange` and no `PollFlightInfo`.
- Action documents and DoPut metadata are at most 64 KiB.
- At most 400 logical payloads per job so the complete seal manifest remains
  below the action-document limit.
- One DoPut is always one semantic legacy frame; internal RecordBatch count
  never changes fit/predict invocation boundaries.
- One predict job loads one checkpoint in one `predict-stream` subprocess and
  emits one output per input ordinal.
- Running fit is never automatically retried after interruption.
- Transformer owns checkpoints; clients receive only opaque `modelRef`.
- Output tickets are short-lived, opaque and not model references.
- Plaintext exists only for explicit development/LAN operation and is not a
  production security profile.
- Direct dependencies are version-pinned, but the environment must select and
  record the intended CPU/CUDA Torch wheel and its transitive resolution.
- Node-to-PyArrow interoperability and physical CUDA behavior require separate
  target-environment validation; Python tests do not prove either gate.
