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

## Flight v15 clean cut

Revision `0027_public_contract_simplification` разрушительна. Она необходима
для границы Flight v15 / Semantic v3 и не имеет downgrade.

Перед её применением:

1. Остановите каждый instance service Transformer, использующий эту schema
   PostgreSQL.
2. Убедитесь, что каждое job terminal. Migration отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING`, or `CANCELLING` jobs rather than deleting
   active work.
3. Решите, нужен ли внешний backup. Старые models и telemetry намеренно не
   сохраняются этим release.
4. Примените migration один раз.

Migration удаляет старые jobs, attempts, inputs/outputs через database
cascades, records idempotency, models, records deletion, metadata recovery и
records telemetry/outbox в database. При следующем старте service v15
reconciliation managed storage удаляет directories job/model/telemetry без
references под настроенными roots этого service.

OpenSearch находится вне PostgreSQL и эта migration его не меняет. Следуйте
[руководству deployment OpenSearch](../deployment/opensearch.md), чтобы заменить
индексы metrics v6 templates v7 и пустыми indices v7 до нового обучения.

Не запускайте executable до v15 после применения revision 0027. У него нет
совместимого reader database, и он не должен создавать legacy state вновь.
