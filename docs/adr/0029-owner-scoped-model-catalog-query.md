# ADR 0029: owner-scoped Model Catalog Query

- Status: Accepted
- Decision date: 2026-09-06

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

`model.describe` позволял прочитать metadata только заранее известной model
generation. Consumer не мог обнаружить доступные модели без обращения к
административному CLI, filesystem или best-effort OpenSearch telemetry. Эти
источники не являются owner-scoped registry truth и не дают bounded traversal
при одновременной публикации и удалении.

## Решение

Предоставить отдельный read-only Model Catalog Query с независимой immutable
revision. Registry опубликованных моделей остаётся единственным источником
membership, а owner scope всегда выводится из аутентифицированного subject.
Первая revision содержит bounded `list` и single `detail`; mutation, filters,
batch detail и server-side comparison в неё не входят.

List использует deterministic keyset order и owner-scoped high-water boundary.
Service-issued cursor integrity-protected, привязан к owner, revision, page
size, traversal state и абсолютному non-sliding expiration. Concurrent
publication после boundary исключается, а concurrent deletion может дать
оговорённый пропуск без строгого snapshot и tombstones.

List читает только registry metadata. Detail дополнительно валидирует canonical
contracts и digests и выполняет bounded full SHA-256 verification checkpoint.
Unknown, foreign и deleted model references имеют security-equivalent outcome.
Telemetry не определяет существование модели и не становится неявным query
backend.

Первичная transport-публикация query revision 1 выполняется clean-cut Flight
v12. Дальнейший lifecycle query language не обязан совпадать с версией job
workflow.

## Рассмотренные альтернативы

- Использовать OpenSearch metrics как каталог. Отклонено: telemetry best effort
  и имеет независимый lifecycle.
- Сканировать filesystem либо публиковать PostgreSQL/admin semantics.
  Отклонено: это раскрывает storage boundary и не обеспечивает owner-scoped
  application contract.
- Возвращать полный detail каждой строкой list. Отклонено из-за лишнего I/O и
  неограниченной стоимости страницы.
- Ввести строгий revisioned snapshot с tombstones. Отклонено как избыточное для
  согласованной UI consistency model.
- Выполнять сравнение моделей на стороне Transformer. Отклонено: семантика
  сравнения принадлежит Consumer-у.

## Последствия

- Consumer обнаруживает immutable generations без зависимости от telemetry и
  сохраняет exact `modelRef` для последующих операций.
- Pagination и detail verification имеют явные пределы, expiration и
  structured outcomes.
- Registry получает монотонную publication boundary и ограничение размера
  новых checkpoint, пригодных для полной detail verification.
- Удаление между list и detail безопасно завершается `MODEL_NOT_FOUND`; Consumer
  обновляет traversal.
- Batch detail, filters и bounded run/telemetry query требуют отдельных
  последующих решений.

## Текущая документация

- [Model Catalog Query v1](../../app/contracts/model_catalog/v1/README.md)
- [Контракт Arrow Flight v12](../../app/contracts/flight/v12/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Архитектура Transformer](../architecture.md)
- [Управление опубликованными моделями](../operations/published-models.md)
