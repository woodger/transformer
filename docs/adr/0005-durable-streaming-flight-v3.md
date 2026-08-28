# ADR 0005: durable streaming lifecycle и fencing

- Status: Accepted
- Decision date: 2026-08-10

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Batch-oriented protocol ожидал полный manifest до запуска worker и оставлял
две распределённые race conditions: потерянный create response и позднюю
mutation прежнего владельца после takeover в Inventory. Он также не позволял
начать первую epoch, пока Consumer продолжал durable upload.

Привязка job к одному Flight connection сделала бы disconnect равнозначным
потере прикладного состояния, хотя committed inputs уже могли быть надёжно
сохранены.

## Решение

Flight job использует application-level durable streaming protocol, а не
connection-scoped stream. Каждый успешный DoPut публикует один immutable
semantic payload и фиксирует receipt до ответа. RecordBatch boundaries
остаются transport chunking.

Consumer создаёт и сохраняет `jobId` до network create. Idempotency record
восстанавливает потерянный response. Каждая публичная mutation защищается
монотонным cross-system fencing token; проверка повторяется в transaction
commit после потенциально долгого upload.

Input и execution имеют независимые state axes. Первый непрерывный непустой
prefix может поставить worker в queue при открытом input, а явное close
фиксирует EOF и immutable manifest. Job и committed data переживают замену
network connection и worker attempt. `DoExchange` не используется как
lifecycle boundary.

## Рассмотренные альтернативы

- Ждать полного sealed manifest до execution. Отклонено, потому что upload и
  первая epoch не перекрываются.
- Связать job с одним `DoExchange`. Отклонено из-за слабой durability при
  disconnect и сложного retry.
- Выдавать `jobId` только сервером и добавлять resolve operation. Отклонено в
  пользу client-generated identity и idempotent create.
- Полагаться только на lease Inventory. Отклонено: provider обязан проверять
  stale owner на собственной mutation boundary.
- Поддерживать старый и новый lifecycle одновременно. Отклонено из-за
  неоднозначных state, recovery и fencing semantics.

## Последствия

- Upload, execution и reconnect могут идти независимо без потери committed
  inputs.
- Stale Consumer не может зафиксировать позднюю mutation после takeover.
- Streaming epoch требует deterministic data order и явного EOF behavior.
- Protocol и worker supervision сложнее batch execution и требуют recovery из
  durable snapshots, а не доверия notification delivery.

## Текущая документация

- [Контракт Arrow Flight v6](../../app/contracts/flight/v6/README.md)
- [Интеграция Consumer-ов](../consumer-flight-integration.md)
- [Операционное руководство Flight](../operations/flight-service.md)
- [Training runtime](../training-runtime.md)
