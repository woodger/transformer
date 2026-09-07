# ADR 0020: local opaque API access tokens

- Status: Accepted
- Decision date: 2026-08-24

> Historical decision record; not a current system reference. See
> [ADR index](index.md).

## Контекст

Transformer предоставляет service-to-service API одному контролируемому
Consumer-у. Нет interactive users, third-party clients, delegated permissions,
нескольких audiences, federation или независимого identity lifecycle.
PostgreSQL уже обязателен как durable authority service state.

Использование внешнего OAuth authorization server добавило отдельные identity,
availability и administration planes без прикладной потребности. Полный cache
активных tokens через PostgreSQL notifications, в свою очередь, требовал
listener connection, trigger и reload lifecycle.

## Решение

Transformer самостоятельно выпускает opaque Bearer credentials. Token не
содержит claims и не является JWT. Все credentials отображаются на стабильный
service owner; несколько одновременно действующих tokens обеспечивают
rotation, а не разные permissions.

PostgreSQL хранит только криптографический digest и lifecycle metadata.
Отдельный случайный token ID служит management handle и не позволяет пройти
authentication. Expiration фиксируется при issue и не является sliding.
Revoke физически удаляет row; встроенная revoke audit history отсутствует.

Verification остаётся stateful и database-backed, но использует bounded
process-local positive cache-aside. Cache не является durable authority, не
preload-ит полный набор и не использует `LISTEN/NOTIFY`. Bounded cache TTL
является одновременно явной верхней границей revoke latency. Database failure
на revalidation возвращает availability error, а не превращается в invalid
credential.

Bearer является replayable secret. При передаче через недоверенную сеть TLS с
проверкой server identity является частью security boundary. mTLS может
дополнять transport, но не заменяет Bearer lifecycle.

## Рассмотренные альтернативы

- OAuth 2.0/OIDC provider. Подходит для users, federation, scopes и delegated
  access, но создаёт лишний control plane для текущей service boundary.
- Self-contained JWT. Отклонён из-за signing-key lifecycle, embedded claims и
  сложного immediate revoke.
- PostgreSQL lookup на каждый RPC. Проще по consistency, но добавляет database
  round trip каждому вызову.
- Полный cache через `LISTEN/NOTIFY`. Отклонён из-за listener/trigger/reload
  lifecycle и зависимости memory footprint от всех active tokens.
- Raw Bearer storage или token file. Отклонены из-за последствий read-only
  утечки и второго persistence contract рядом с PostgreSQL.

## Последствия

- Transformer владеет generation, rotation, expiration и revoke credentials.
- PostgreSQL требуется при cache miss, но не на каждом RPC.
- Revoke имеет bounded, а не мгновенную process-local propagation.
- Hard delete минимизирует metadata и исключает встроенную историю отзывов.
- Появление users, scopes, delegated access, federation или обязательного
  centralized credential audit требует нового архитектурного решения.

## Текущая документация

- [Аутентификация Arrow Flight](../authentication.md)
- [Управление API access tokens](../operations/api-access-tokens.md)
- [Контракт Arrow Flight v13](../../app/contracts/flight/v13/README.md)
