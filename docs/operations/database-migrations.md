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

## Semantic v6 clean cut и текущий Flight v23

Revision `0030_bernoulli_confidence_penalty` разрушительна и не имеет downgrade.
Она изменяет только границу durable state: D1 включает revision языка даже для
objective без регуляризатора. Старые jobs, generations, checkpoint/recovery,
idempotency records и database telemetry не имеют reader Semantic v6 /
checkpoint v13 и удаляются. Таблицы и поля PostgreSQL не меняются.

Перед её применением:

1. Остановите все instances service Transformer для этой schema PostgreSQL.
2. Дождитесь terminal outcome каждого job. Migration отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING` и `CANCELLING`.
3. Сохраните внешний backup, если нужны прежние models, jobs или telemetry.
   Этот release намеренно не переносит их на новый contract.
4. Примените migration один раз и выполните documented replacement индексов
   OpenSearch metrics v11 на v12 до запуска Flight v23. Не запускайте после
   этого старый executable, не поддерживающий Semantic v6 / checkpoint v13.

Metrics v12 связывает observations с checkpoint v13; runtime не читает прежние
point/run formats v11. Replacement индексов является отдельной операцией
из `docs/deployment/opensearch.md`, migration PostgreSQL не удаляет OpenSearch
documents. Startup reconciliation очищает только
уже не имеющие references managed artifacts согласно действующему lifecycle.

Target Head Diagnostics v5 сохраняет прежний artifact и configuration v3;
confidence penalty не встраивается в diagnostics artifact и не меняет его
формат. Его declaration хранится в objective checkpoint/registry metadata,
а observations самого penalty — в auxiliary losses и gradient interactions.
