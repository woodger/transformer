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
