# ADR 0018: PostgreSQL-backed API access tokens

## Статус

Принято как implementation record. Архитектурный класс решения и целевая
cache/TLS boundary уточнены [ADR 0020](0020-local-opaque-api-access-tokens.md);
решение о полном token index через PostgreSQL `LISTEN/NOTIFY` заменено.

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
фиксированный owner subject `inventory`. Credential действует три календарных
месяца с момента выпуска или до явного `revoke`. Один экземпляр Transformer и
его PostgreSQL database обслуживают эту единственную service identity.

Административный CLI:

```text
auth tokens issue
auth tokens list
auth tokens revoke <TOKEN_ID>
```

`issue` один раз печатает `Token ID`, `Expires` и `Token`. `list` показывает
ID, состояние `Active` или `Expired`, creation time и expiration time без
credential. `revoke` принимает UUID и физически удаляет token row. Неизвестный
ID и повторный revoke возвращают `not found`.

PostgreSQL хранит только SHA-256 digest credential, token ID, subject,
`created_at` и `expires_at`. Случайный bearer содержит 512 бит энтропии, поэтому
digest не требует password-oriented slow hashing. Исходный credential не
сохраняется в ORM, PostgreSQL, cache keys, logs или errors.

Flight process при старте загружает активные digest records в неизменяемый
in-process index. Каждый новый RPC вычисляет digest переданного bearer и
проверяет `expires_at`, после чего получает точный subject без запроса к
PostgreSQL. Trigger отправляет notification после issue/revoke; listener
полностью перечитывает активный набор. После reconnect listener также выполняет
полную загрузку. Новый RPC с неизвестным, отозванным или истёкшим credential
получает `UNAUTHENTICATED`. Уже авторизованный streaming RPC продолжает работу
до обычной границы завершения.

Migration `0014` и ADR 0017 остаются в истории применённых изменений.
Migration `0015` создаёт новую пустую digest-only таблицу и notification
objects. Migration `0016` нормализует физическую schema, если database уже
отмечена как `0015`, но сохранила историческую raw-колонку: существующие
credentials переводятся в digests без изменения identity или revoke status.
Migration `0017` удаляет все существующие бессрочные token rows и добавляет
обязательный `expires_at`. Credentials, удалённые migrations `0014` и `0017`,
не восстанавливаются. Migration `0018` удаляет существующие revoked rows и
колонку `revoked_at`; эти metadata также не восстанавливаются.

Ory Hydra adapters, `HYDRA_ENDPOINT`, `auth clients` и OAuth-specific runtime
configuration удаляются. Flight v5 wire schemas, worker contract и persisted
owner-scoped application state не меняются.

## Последствия

- Consumer хранит bearer, передаёт его в каждом Flight RPC и выполняет ротацию
  до `expires_at`; отдельный token exchange отсутствует.
- Компрометация bearer предоставляет доступ до явного отзыва или expiration,
  поэтому credential должен находиться только в secret storage Consumer-а.
- Все выпущенные tokens имеют subject `inventory` и видят один owner-scoped
  state, что позволяет ротацию без переименования owner-а.
- После revoke token metadata не сохраняются; встроенная audit history отзывов
  отсутствует. Expired token остаётся в PostgreSQL до явного revoke.
- PostgreSQL и token listener обязательны при старте Flight process.
- Выпуск и отзыв не требуют перезапуска Flight service.
