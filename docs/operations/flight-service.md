# Transformer Arrow Flight service operations

> Тип: операционное руководство. Запуск, shutdown, storage, and recovery of
> the current Flight v15 service.

Wire semantics are defined by [Flight v15](../../app/contracts/flight/v15/README.md).
This document covers service operation, not Consumer JSON details.

## Runtime requirements

- project `.venv` and the pinned Python dependencies;
- PostgreSQL reachable from the service host;
- persistent configured runtime, models, recovery, and telemetry directories;
- a Linux process environment capable of terminating service-owned Worker
  process groups;
- `nvidia-smi` only when GPU scheduling is required;
- OpenSearch only when Training Telemetry Query should materialize reports.

OpenSearch outage does not block fit, predict, model publication, Model Catalog,
or shutdown. It makes telemetry query unavailable or unavailable-after-terminal
according to its query contract.

## Start and stop

Apply migrations before start. Use the production systemd unit where deployed:

```bash
sudo systemctl start transformer
systemctl status transformer
journalctl -u transformer -f
sudo systemctl stop transformer
```

The process obtains exclusive locks for its configured runtime and recovery
directories. Two service instances must use distinct managed storage roots;
one instance must never clean another instance's runtime artifacts.

At startup the service verifies the database revision, terminates only safely
identified orphan Worker process groups, reconciles interrupted jobs, and
removes unreferenced files only inside its locked managed roots. It does not
scan arbitrary filesystem paths or OpenSearch for authority.

## Storage ownership

PostgreSQL is authoritative for job lifecycle, idempotency, owners, attempts,
published model metadata, API tokens, and telemetry outbox state. Managed
filesystem storage contains durable inputs, attempt/recovery artifacts,
published checkpoints, and temporary telemetry files. OpenSearch is a
best-effort projection.

Published model directories contain only provider-managed checkpoint artifacts.
Catalog/detail resolves their metadata from PostgreSQL; neither filesystem scan
nor telemetry can create a visible model generation.

## Recovery and cleanup

Recovery resumes only from a registered checkpoint whose job configuration,
input manifest, semantic identities, and progress fences match exactly. A
corrupt or incompatible artifact fails; it is never silently remapped.

Maintenance removes terminal job artifacts according to configured retention
and executes requested model deletion. Startup reconciliation removes only
unreferenced artifacts in the same service roots. Do not manually delete
PostgreSQL rows or managed model directories to force cleanup; use `models
delete` or the documented clean-cut migration.

## Flight v15 release

Migration 0027 deletes prior boundary state. Stop all service instances, wait
for jobs to become terminal, apply it, replace OpenSearch metrics indices with
v7, then deploy the v15 service. Old models, checkpoints, recovery state, and
telemetry cannot be used afterwards. See
[database migrations](database-migrations.md) and
[OpenSearch deployment](../deployment/opensearch.md).

## Health and troubleshooting

Use `transformer.v15.health` for the authenticated provider health surface and
`transformer.v15.capabilities` for current device/upload/query availability.
Use service logs and database state for operational diagnosis. Never put bearer
credentials, database passwords, or raw checkpoint paths into shared logs or
support messages.
