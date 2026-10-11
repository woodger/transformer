# Управление схемой PostgreSQL

> Тип: операционное руководство. Проверка и применение schema migrations
> Transformer.

Flight service никогда не запускает Alembic автоматически. Остановите сервис
перед upgrade, меняющим state, используемый работающим process. Используйте ту
же configuration database, что и service, и выполните управляемый оператором
backup до любой destructive migration.

## Commands

```bash
./.venv/bin/python app/main.py db migrations status
./.venv/bin/python app/main.py db migrations apply
./.venv/bin/python app/main.py db migrations rollback
```

`status` доступен только для чтения. `apply` обновляет до Alembic head текущего
checkout и затем печатает status. `rollback` не заменяет backup: некоторые
revisions намеренно отклоняют downgrade.

## Схема public

Текущий runtime и команды `auth`, `models` и `db` используют `public`.
На новой базе migrations создают таблицы и `alembic_version` в этой схеме.
У роли должны быть права `USAGE` и `CREATE` на `public`.

Для существующей базы с таблицами в `transformer` выполните однократный перенос
при остановленных instances service до первого запуска нового runtime или
`db migrations apply`. Целевая схема `public` должна существовать и не содержать
конфликтующих имён таблиц, индексов, типов или последовательностей.

Следующий SQL переносит набор таблиц revision `0031` без изменения данных или
номера revision. Индексы, ограничения и принадлежащая `models` identity sequence
перемещаются вместе с таблицами; независимая очередь требует отдельного
`ALTER SEQUENCE`. Удаление пустой схемы использует `RESTRICT`, поэтому
неучтённые объекты остановят транзакцию.

```sql
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

ALTER TABLE transformer.job_identities SET SCHEMA public;
ALTER TABLE transformer.jobs SET SCHEMA public;
ALTER TABLE transformer.job_attempts SET SCHEMA public;
ALTER TABLE transformer.job_inputs SET SCHEMA public;
ALTER TABLE transformer.input_uploads SET SCHEMA public;
ALTER TABLE transformer.job_outputs SET SCHEMA public;
ALTER TABLE transformer.models SET SCHEMA public;
ALTER TABLE transformer.deleted_models SET SCHEMA public;
ALTER TABLE transformer.idempotency_records SET SCHEMA public;
ALTER TABLE transformer.api_access_tokens SET SCHEMA public;
ALTER TABLE transformer.output_tickets SET SCHEMA public;
ALTER TABLE transformer.runtime_state SET SCHEMA public;
ALTER TABLE transformer.training_metrics_artifacts SET SCHEMA public;
ALTER TABLE transformer.fit_run_summary_artifacts SET SCHEMA public;
ALTER TABLE transformer.metrics_outbox SET SCHEMA public;
ALTER TABLE transformer.training_metric_intervals SET SCHEMA public;
ALTER TABLE transformer.training_recovery_checkpoints SET SCHEMA public;
ALTER TABLE transformer.alembic_version SET SCHEMA public;
ALTER SEQUENCE transformer.job_queue_sequence_seq SET SCHEMA public;

DROP SCHEMA transformer RESTRICT;
COMMIT;
```

После переноса сравните данные и значения sequences, проверьте
`db migrations status` и доступ к прежним models, jobs и API tokens. Запускайте
сервис с кодом, использующим `public`. Простая смена schema до переноса
`alembic_version` заставит Alembic считать базу немигрированной; повторный
upgrade вместо переноса создаст другой набор таблиц.

## Semantic v7 clean cut и текущий Flight v24

Revision `0031_bernoulli_entropy_penalty` разрушительна и не имеет downgrade.
Она изменяет только границу durable state: D1 включает revision языка даже для
objective без регуляризатора. Старые jobs, generations, checkpoint/recovery,
idempotency records и database telemetry не имеют reader Semantic v7 /
checkpoint v14 и удаляются. Таблицы и поля PostgreSQL не меняются.

Перед её применением:

1. Остановите все instances service Transformer для этой schema PostgreSQL.
2. Дождитесь terminal outcome каждого job. Migration отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING` и `CANCELLING`.
3. Сохраните внешний backup, если нужны прежние models, jobs или telemetry.
   Этот release намеренно не переносит их на новый contract.
4. Примените migration один раз и выполните documented replacement индексов
   OpenSearch metrics v12 на v13 до запуска Flight v24. Не запускайте после
   этого старый executable, не поддерживающий Semantic v7 / checkpoint v14.

Metrics v13 связывает observations с checkpoint v14; runtime не читает прежние
point/run formats v12. Replacement индексов является отдельной операцией
из `docs/deployment/opensearch.md`, migration PostgreSQL не удаляет OpenSearch
documents. Startup reconciliation очищает только
уже не имеющие references managed artifacts согласно действующему lifecycle.

Target Head Diagnostics v5 сохраняет прежний artifact и configuration v3;
confidence и entropy penalties не встраиваются в diagnostics artifact и не
меняют его формат. Их declarations хранятся в objective checkpoint/registry
metadata, а observations — в auxiliary losses и gradient interactions.
