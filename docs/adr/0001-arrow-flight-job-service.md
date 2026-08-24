# ADR 0001: граница Arrow Flight job service

- Status: Accepted
- Decision date: 2026-07-18

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Inventory и Transformer должны выполняться на разных физических серверах.
Существовавшие локальные framed Arrow commands рассчитаны на один subprocess,
а обучение изменяет model и optimizer state. Простой сетевой retry после
начала такого вызова мог повторить вычисление или публикацию с неизвестным
результатом.

Требовалась серверная граница, которая принимает Arrow data, владеет identity
и lifecycle длительной работы и отделяет transport retry от ML execution.

## Решение

Transformer владеет single-instance Arrow Flight job service. Service хранит
авторитетное control-plane state в PostgreSQL, принимает immutable semantic
payloads, планирует execution attempts и публикует только завершённые
artifacts. Torch выполняется вне RPC handler в отдельном worker process.

Клиент оперирует domain identities и idempotent operations. Network requests
не передают filesystem paths или произвольные worker arguments. Transformer
остаётся владельцем checkpoints и возвращает opaque model identity.

Решение фиксирует service boundary, но не конкретную версию Flight protocol,
схему PostgreSQL, storage layout или authentication implementation.

## Рассмотренные альтернативы

- Вызывать локальный training CLI как stateless remote command. Отклонено,
  потому что retry и потерянный response неразрешимы после начала mutation.
- Открыть framed subprocess protocol напрямую через сеть. Отклонено из-за
  отсутствия durable job identity, fencing и recovery boundary.
- Выполнять Torch внутри Flight RPC handler. Отклонено из-за слабой process
  isolation, cancellation и resource ownership.
- Разделить PostgreSQL или filesystem напрямую с Inventory. Отклонено, потому
  что это размывает provider boundary и durable ownership.

## Последствия

- Transformer становится stateful infrastructure service, а не удалённой
  оболочкой над CLI.
- Idempotency, recovery и artifact publication являются частью service
  lifecycle.
- Worker failure и CUDA runtime изолированы от Flight transport process.
- Single-instance boundary требует явного пересмотра перед введением replicas
  или общего scheduler/storage.

## Текущая документация

- [Архитектурная политика](../policy/architecture.md)
- [Контракт Arrow Flight v5](../../app/contracts/flight/v5/README.md)
- [Операционное руководство Flight](../flight-operations.md)
