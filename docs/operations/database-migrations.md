# Миграции PostgreSQL

> Тип: операционное руководство. Проверка версии базы, применение миграций и перенос таблиц.

Миграции подготавливают базу для текущей версии Transformer. Сначала проверьте
её состояние, затем примените необходимые изменения. Сервис не запускает
Alembic автоматически.

Перед обновлением, которое меняет используемые сервисом данные, остановите
сервис. Команды должны обращаться к той же базе. До миграции с удалением
данных оператор сохраняет резервную копию.

## Команды

```bash
./.venv/bin/python app/main.py db migrations status
./.venv/bin/python app/main.py db migrations apply
./.venv/bin/python app/main.py db migrations rollback
```

`status` только читает состояние. `apply` обновляет базу до последней миграции
текущего кода и печатает результат. `rollback` откатывает последнюю миграцию,
если она допускает откат. Некоторые миграции его запрещают, поэтому эта
команда не заменяет резервную копию.

## Схема public

Сервис и команды `auth`, `models` и `db` используют `public`.
На новой базе миграции создают таблицы и `alembic_version` в этой схеме.
У роли должны быть права `USAGE` и `CREATE` на `public`.

Для существующей базы с таблицами в `transformer` выполните однократный перенос
при остановленных экземплярах сервиса до первого запуска нового кода или
`db migrations apply`. Целевая схема `public` должна существовать и не содержать
конфликтующих имён таблиц, индексов, типов или последовательностей.

Следующий SQL переносит таблицы версии `0031` без изменения данных или
номера миграции. Индексы, ограничения и последовательность identity-столбца `models`
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

После переноса сравните данные и значения последовательностей, проверьте
`db migrations status` и доступ к прежним моделям, заданиям и токенам. Запускайте
сервис с кодом, использующим `public`. Смена схемы в настройках до переноса
`alembic_version` заставит Alembic считать базу немигрированной; повторное
применение миграций вместо переноса создаст другой набор таблиц.

## Обновление формата моделей

Миграция `0031_bernoulli_entropy_penalty` удаляет несовместимые данные и не
допускает откат. Версия языка входит в идентификатор D1 даже для функции потерь
без регуляризатора. Поэтому прежние задания, модели, контрольные точки,
состояние восстановления, записи идемпотентности и метрики БД несовместимы
с Semantic v7 и контрольными точками v14 и удаляются. Структура таблиц
и полей PostgreSQL не меняется.

Перед её применением:

1. Остановите все экземпляры Transformer, использующие эту схему PostgreSQL.
2. Дождитесь завершения каждого задания. Миграция отклоняет `WAITING_INPUT`,
   `QUEUED`, `RUNNING`, `RETRYING` и `CANCELLING`.
3. Сохраните внешнюю резервную копию, если нужны прежние модели, задания или
   метрики. Эта версия не переносит их в новый формат.
4. Примените миграцию один раз и замените индексы метрик OpenSearch v12 на v13
   по инструкции до запуска нового сервиса. После этого используйте код,
   поддерживающий Semantic v7 и контрольные точки v14.

Метрики v13 связаны с контрольными точками v14; сервис не читает прежние
форматы метрик v12. Замена индексов — отдельная операция из
[инструкции OpenSearch](../deployment/opensearch.md). Миграция PostgreSQL
не удаляет документы OpenSearch. При старте сервис очищает только свои файлы,
на которые больше нет ссылок, по обычным правилам обслуживания.

Диагностика выходных головок v5 сохраняет прежний формат файла и конфигурацию
v3. Confidence penalty и entropy penalty не меняют этот формат. Их объявления
хранятся в определении функции потерь в контрольной точке и метаданных модели,
а результаты — во вспомогательных потерях и взаимодействиях градиентов.
