# ADR 0030: owner-scoped запрос telemetry обучения

- Статус: Принято
- Дата решения: 2026-09-07

> Историческая запись решения; не является актуальным описанием системы. См.
> [указатель ADR](index.md).

## Контекст

Model Catalog показывает immutable model generation, но не предоставляет
training losses, target errors, optimizer health и gradient interactions.
Consumer не должен читать provider-owned OpenSearch indices или определять
существование модели по best-effort telemetry.

## Решение

Предоставить independently versioned read-only Training Telemetry Query.
Каждый request сначала разрешает exact `modelRef` через owner-scoped
registry и только затем читает provider-owned metrics projection. Unknown,
foreign и deleted generations неразличимы; telemetry не определяет
registry membership.

Query возвращает только полный проверенный epoch report, а optional
gradient interactions читает отдельным bounded action. Наблюдения
описывают training-проход эпохи до optimizer updates, а не повторную
оценку published checkpoint. Query не проверяет checkpoint и не добавляет
compatibility digest.

Первичная transport-активация revision 1 выполняется clean-cut Flight
v13. PostgreSQL schema, Worker/checkpoint formats и metrics v5 projection не
меняются.

## Рассмотренные альтернативы

- Дать Consumer-у доступ к OpenSearch. Отклонено: это раскрывает
  storage topology, обходит owner scope и связывает Consumer с metrics format.
- Вложить telemetry в Model Catalog detail. Отклонено: catalog truth и
  best-effort projection имеют разные availability и lifecycle.
- Возвращать gradient pairs в main report. Отклонено из-за
  потенциально квадратичного ответа.
- Повторно оценивать published checkpoint. Отклонено: это другая
  ML operation, а не query уже собранных training observations.

## Последствия

- Inventory получает browser-safe normalized report без знания OpenSearch
  indices, documents и credentials.
- Отсутствие, задержка или повреждение telemetry имеют собственные
  outcomes и не инвалидируют published model.
- Pagination cursor остаётся owner/model/report-bound; удаление model
  generation имеет приоритет над continuation.
- Активация меняет closed Flight action surface и требует синхронного
  обновления Consumer adapter.

## Текущая документация

- [Запрос телеметрии обучения v4](../../app/contracts/training_telemetry/v4/README.md)
- [Контракт Arrow Flight v22](../../app/contracts/flight/v22/README.md)
- [Интеграция с Flight](../flight-integration.md)
- [Архитектура Transformer](../architecture.md)
- [Политика metrics и OpenSearch](../policy/metrics-policy.md)
