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

## Базовый Semantic v4 clean cut и Flight v18

Revision `0028_positive_class_weighted_binary_bce` разрушительна и не имеет
downgrade. Она удаляет jobs, опубликованные generations, recovery/checkpoint
metadata, idempotency records и database telemetry, для которых отсутствует
reader Semantic v4 / Flight v18.

Перед её применением:

1. Остановите каждый instance service Transformer, использующий эту schema
   PostgreSQL.
2. Убедитесь, что каждое job terminal. Migration отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING` и `CANCELLING`, а не удаляет active work.
3. Решите, нужен ли внешний backup. Предыдущие models и telemetry намеренно не
   сохраняются этим release.
4. Примените migration один раз.
5. До нового обучения Flight v18 выполните замену индексов OpenSearch v8 на v9 по
   [руководству deployment OpenSearch](../deployment/opensearch.md).

Не запускайте executable, не поддерживающий Semantic v4, после применения
revision 0028. У него нет совместимого reader database, и он не должен
создавать legacy state вновь.

Flight v18 не добавляет migration PostgreSQL: его target head diagnostics
хранятся в checkpoint metadata и registry metadata опубликованной модели.
