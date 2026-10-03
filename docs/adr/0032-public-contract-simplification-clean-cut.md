# ADR 0032: упрощение публичной границы clean cut

- Статус: Принято
- Дата решения: 2026-09-16

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

Subject-specific vocabulary устранил универсальный discriminator, но shared
documents всё ещё передавали Consumer-у provider-owned architecture,
операционные fences, повторённую geometry и derived totals. Это связывало
Inventory configuration с внутренней формой Transformer и дублировало данные,
которые provider может однозначно materialize после owner-scoped create.

Совместимое изменение невозможно: JSON representation входит в D1, хранится в
checkpoint/recovery metadata и возвращается Catalog/Telemetry projections.

## Решение

Принять destructive clean cut:

- Semantic v3 принимает Consumer-owned target/objective intent и
  `modelTuning`; Transformer выпускает opaque model-definition identity;
- Flight v15 разделяет fit и predict create, выдаёт job ID и mutation lease,
  а input close выводит totals из durable receipts;
- Worker v14, checkpoint/recovery v8, metrics v7, Model Catalog Query v3 и
  Training Telemetry Query v3 являются единственным active package;
- migration `0027` удаляет несовместимые durable jobs, models, telemetry и
  idempotency records; legacy reader, conversion и warm start не
  предоставляются.

Математика objective, typed bindings, owner scope, idempotency, Arrow data
plane и logical reconstruction `indexedFeatureBlocks` не меняются. Новая
representation не является digest-equivalent прежней; model definition digest
выпускает Transformer, а Consumer сохраняет его как opaque identity.

## Рассмотренные альтернативы

- Оставить прежний public document и скрыть его за Inventory adapter-ом.
  Отклонено: provider wire details продолжили бы проникать в Consumer domain.
- Поддерживать v14 и v15 одновременно. Отклонено: это сохраняет legacy
  contracts, models и recovery fencing.
- Конвертировать старые models/checkpoints. Отклонено: semantic migration и
  representation-independent hashing не были определены и не нужны.

## Последствия

- Consumer обновляется согласованно до единственного Flight v15 action surface.
- Любая существующая published generation и её telemetry удаляются при
  применении migration; обучение начинается заново.
- Current documentation и contract fixtures описывают только v3/v15 package;
  прежние source files и tests удалены из runtime tree.

## Текущая документация

- [Семантический контракт v5](../../app/contracts/semantic/v5/README.md)
- [Контракт Arrow Flight v22](../../app/contracts/flight/v22/README.md)
- [Процессный контракт Worker v20](../../app/contracts/worker/v20/README.md)
- [Контракт checkpoint/recovery v12](../../app/contracts/checkpoint/v12/README.md)
- [Запрос каталога моделей v7](../../app/contracts/model_catalog/v7/README.md)
- [Запрос телеметрии обучения v4](../../app/contracts/training_telemetry/v4/README.md)
- [Миграции PostgreSQL](../operations/database-migrations.md)
