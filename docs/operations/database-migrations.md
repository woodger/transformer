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

## Semantic v5 clean cut и текущий Flight v22

Revision `0029_encoder_normalization_order` разрушительна и не имеет downgrade.
Она удаляет jobs, опубликованные generations, recovery/checkpoint metadata,
idempotency records и database telemetry, для которых отсутствует reader
Semantic v5 / Flight v22.

Перед её применением:

1. Остановите каждый instance service Transformer, использующий эту schema
   PostgreSQL.
2. Убедитесь, что каждое job terminal. Migration отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING` и `CANCELLING`, а не удаляет active work.
3. Решите, нужен ли внешний backup. Предыдущие models и database telemetry
   намеренно не сохраняются этим release. Уже существующие documents Metrics
   v10 в OpenSearch migration не удаляет, но без registry generation они не
   становятся доступными через query.
4. Примените migration один раз, затем выполните documented destructive
   replacement индексов OpenSearch metrics v10 на v11 до запуска Flight v22.

Не запускайте executable, не поддерживающий Semantic v5, после применения
revision 0029. У него нет совместимого reader database, и он не должен
создавать legacy state вновь.

Target head diagnostics хранятся в checkpoint metadata и registry metadata
опубликованной модели.
