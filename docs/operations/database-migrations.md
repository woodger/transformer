# Управление схемой PostgreSQL

> Тип: операционное руководство. Проверка и применение schema migrations
> Transformer.

Flight service never runs Alembic automatically. Stop the service before an
upgrade that changes state used by a running process. Use the same database
configuration as the service and take an operator-managed backup before any
destructive migration.

## Commands

```bash
./.venv/bin/python app/main.py db migrations status
./.venv/bin/python app/main.py db migrations apply
./.venv/bin/python app/main.py db migrations rollback
```

`status` is read-only. `apply` upgrades to the checkout's Alembic head and
prints status afterwards. `rollback` is not a substitute for backup: some
revisions deliberately reject downgrade.

## Flight v15 clean cut

Revision `0027_public_contract_simplification` is destructive. It is required
for the Flight v15 / Semantic v3 boundary and has no downgrade.

Before applying it:

1. Stop every Transformer service instance sharing this PostgreSQL schema.
2. Ensure every job is terminal. The migration refuses `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING`, or `CANCELLING` jobs rather than deleting
   active work.
3. Decide whether an external backup is required. Old models and telemetry are
   intentionally not retained by this release.
4. Apply the migration once.

The migration removes old jobs, attempts, inputs/outputs through database
cascades, idempotency records, models, deletion records, recovery metadata,
and database-backed telemetry/outbox records. On the next v15 service start,
managed-storage reconciliation removes unreferenced job/model/telemetry
directories under that service's configured roots.

OpenSearch is outside PostgreSQL and is not changed by this migration. Follow
[the OpenSearch deployment guide](../deployment/opensearch.md) to replace
metrics v6 indices with v7 templates and empty v7 indices before new training.

Do not start a pre-v15 executable after applying revision 0027. It has no
compatible database reader and must not recreate legacy state.
