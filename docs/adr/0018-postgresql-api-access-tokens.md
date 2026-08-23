# ADR 0018: PostgreSQL-backed API access tokens

## Статус

Принято.

## Контекст

До migration `0014` Transformer самостоятельно выдавал бессрочные bearer
credentials, хранил их в PostgreSQL и обновлял in-process authentication cache
через `LISTEN/NOTIFY`. Публичный административный контракт состоял из команд
`auth tokens issue|list|revoke`, а точный subject credential являлся
`owner_subject` для jobs, models, aliases, tickets и outputs.

ADR 0017 заменил этот механизм на Ory Hydra introspection и отдельное
управление OAuth clients. Эта модель признана ошибочной для Transformer:
управляемой сущностью Consumer-а снова должен быть непосредственно
долгоживущий API token, а PostgreSQL должен оставаться единственным
долговечным источником identity и service state.

Историческая реализация сохраняла исходный bearer credential в таблице. Для
восстановления публичного контракта это не требуется и создаёт лишний риск при
чтении или утечке базы данных.

## Решение

Transformer принимает только собственный credential формата
`a.<base64url>`. При `issue` генерируются 64 случайных байта, UUID token ID и
фиксированный owner subject `inventory`. Credential действует без срока
истечения до явного `revoke`. Один экземпляр Transformer и его PostgreSQL
database обслуживают эту единственную service identity.

Административный CLI:

```text
auth tokens issue
auth tokens list
auth tokens revoke <TOKEN_ID>
```

`issue` один раз печатает `Token ID`, `Subject` и `Token`. `list` показывает
ID, subject, creation time и состояние без credential. `revoke` принимает UUID,
фиксирует `revoked_at` и является идемпотентным.

PostgreSQL хранит только SHA-256 digest credential, token ID, subject,
`created_at` и `revoked_at`. Случайный bearer содержит 512 бит энтропии,
поэтому digest не требует password-oriented slow hashing. Исходный credential
не сохраняется в ORM, PostgreSQL, cache keys, logs или errors.

Flight process при старте загружает активные digest records в неизменяемый
in-process index. Каждый новый RPC вычисляет digest переданного bearer и
получает точный subject без запроса к PostgreSQL. Trigger отправляет
notification после issue/revoke; listener полностью перечитывает активный
набор. После reconnect listener также выполняет полную загрузку. Новый RPC с
неизвестным или отозванным credential получает `UNAUTHENTICATED`. Уже
авторизованный streaming RPC продолжает работу до обычной границы завершения.

Migration `0014` и ADR 0017 остаются в истории применённых изменений.
Migration `0015` создаёт новую пустую digest-only таблицу и notification
objects. Migration `0016` нормализует физическую schema, если database уже
отмечена как `0015`, но сохранила историческую raw-колонку: существующие
credentials переводятся в digests без изменения identity или revoke status.
Credentials, удалённые migration `0014`, не восстанавливаются.

Ory Hydra adapters, `HYDRA_ENDPOINT`, `auth clients` и OAuth-specific runtime
configuration удаляются. Flight v5 wire schemas, worker contract и persisted
owner-scoped application state не меняются.

## Последствия

- Consumer хранит один долгоживущий bearer и передаёт его в каждом Flight RPC;
  отдельный token exchange отсутствует.
- Компрометация bearer предоставляет доступ до явного отзыва, поэтому
  credential должен находиться только в secret storage Consumer-а.
- Все выпущенные tokens имеют subject `inventory` и видят один owner-scoped
  state, что позволяет ротацию без переименования owner-а.
- PostgreSQL и token listener обязательны при старте Flight process.
- Выпуск и отзыв не требуют перезапуска Flight service.
