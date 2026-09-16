# ADR 0031: предметный vocabulary shared contracts

- Статус: Заменено [ADR 0032](0032-public-contract-simplification-clean-cut.md)
- Дата решения: 2026-09-13

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

В первой consumer-neutral форме одно свойство `kind` одновременно служило
discriminator-ом для математических primitives, references, private resources,
initialization и operational documents. Это смешивало разные namespaces в
shared contracts и вынуждало Consumer разбирать provider vocabulary.

Semantic v1 canonical JSON входит в D1 preimages. Его нельзя изменить без
изменения immutable model identity, checkpoint metadata и public queries.

## Решение

Принять clean cut на subject-specific vocabulary: Semantic v2, Flight v14,
Worker v13, checkpoint/recovery v7, metrics v6, Model Catalog Query v2 и
Training Telemetry Query v2.

Public documents используют предметные поля и scalar primitives: `constraint`,
`targetIdentity`, `resourceIdentity`, `resourceClass`, `source`, `encoding` и
`backend`. Requested initialization, checkpoint-resolved initialization и
catalog summary остаются тремя разными documents. Общего replacement-поля
`type` не вводится.

D1 остаётся representation-specific: v1 и v2 не digest-equivalent. Старые
generations, checkpoints, recovery state и telemetry удаляются; models
обучаются заново. Compatibility reader, semantic conversion и weight migration
не предоставляются.

Математика objective, typed roles, gradient semantics, physical Arrow schema
IDs и logical reconstruction `indexedFeatureBlocks` сохраняются.

## Рассмотренные альтернативы

- Оставить `kind` и скрыть его Consumer adapter-ом. Отклонено: provider
  vocabulary всё равно остаётся частью materialized shared document.
- Механически заменить `kind` на `type`. Отклонено: общий discriminator
  сохраняет ту же неоднозначность.
- Ввести representation-independent semantic hashing. Отклонено: потребовало
  бы отдельного языка нормализации и доказуемой migration policy.
- Поддерживать v1 и v2 одновременно. Отклонено: это продлевает legacy runtime
  и делает clean-cut fencing неоднозначным.

## Последствия

- Provider и Consumer обновляются согласованно до единственного Flight v14
  action surface.
- Revision `0026` удаляет durable state прежнего vocabulary; связанные
  filesystem artifacts убирает reconciliation, а external OpenSearch telemetry
  очищается deployment procedure.
- Старые documents остаются историческими fixtures и source files, но не
  принимаются active runtime.

## Текущая документация

- [Семантический контракт v2](../../app/contracts/semantic/v2/README.md)
- [Контракт Arrow Flight v14](../../app/contracts/flight/v14/README.md)
- [Процессный контракт Worker v13](../../app/contracts/worker/v13/README.md)
- [Контракт checkpoint/recovery v7](../../app/contracts/checkpoint/v7/README.md)
- [Запрос каталога моделей v2](../../app/contracts/model_catalog/v2/README.md)
- [Запрос телеметрии обучения v2](../../app/contracts/training_telemetry/v2/README.md)
- [Архитектура Transformer](../architecture.md)
