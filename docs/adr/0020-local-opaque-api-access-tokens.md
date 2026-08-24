# ADR 0020: local opaque API access tokens

- Статус: принято
- Дата: 2026-08-24
- Уточняет архитектурный класс решения ADR 0018
- Заменяет решение ADR 0018 о полном token index через PostgreSQL
  `LISTEN/NOTIFY`

## Контекст

Transformer предоставляет service-to-service API одному контролируемому
Consumer-у. В этой системе нет интерактивных пользователей, сторонних clients,
делегирования полномочий, нескольких audiences или федерации identity.
Авторизация отвечает на один вопрос: может ли предъявленный credential вызвать
Transformer от имени стабильного owner subject.

PostgreSQL уже является обязательным долговечным источником service state.
Credential выпускает и отзывает оператор Transformer, а Consumer хранит его в
собственном secret storage. Несколько одновременно действующих credentials
нужны для ротации одной identity, а не для создания отдельных пользователей.

ADR 0017 вводил внешний OAuth authorization server, client credentials,
introspection, audiences и scopes. Для текущей границы это создало отдельные
identity, availability и administration planes без прикладной потребности.
ADR 0018 вернул PostgreSQL-backed API tokens и зафиксировал текущие CLI,
формат credential, срок действия и migrations. Однако его полный in-memory
index с отдельным PostgreSQL `LISTEN/NOTIFY` lifecycle связывает authentication
с notification trigger, постоянным listener connection и числом всех активных
tokens.

Нужно закрепить не только текущую реализацию token rows, но и выбранный класс
решения вместе с границами его применимости, security properties и
deployment consequences.

## Класс системы

Local opaque access tokens применимы, пока одновременно выполняются следующие
условия:

- provider и Consumer находятся под единым операционным контролем;
- число Consumers мало и их service identities известны заранее;
- полномочия не делегируются пользователям или сторонним applications;
- нет требований к federation, SSO, consent, audiences или scopes;
- PostgreSQL уже обязателен для доступности provider-а;
- operator-managed issue, rotation и revoke являются достаточным lifecycle.

`local` означает, что Transformer сам выпускает и проверяет credential внутри
своей deployment boundary. Это не означает token file, привязку к одному host
или передачу credential через локальный filesystem.

Если появляется хотя бы одна из следующих потребностей, решение должно быть
пересмотрено в пользу поддерживаемого identity provider и стандартного
authorization protocol:

- интерактивные пользователи или third-party clients;
- разные permissions, audiences или scopes;
- delegated access или refresh tokens;
- federation нескольких services или административных доменов;
- независимый identity lifecycle вне PostgreSQL Transformer;
- обязательная централизованная audit history authentication credentials.

## Решение

### Credential и identity

Transformer использует локально выпущенный opaque Bearer credential. Token не
содержит embedded claims и не является JWT. Его значение предоставляет только
доказательство владения; subject, lifecycle и authorization state находятся в
PostgreSQL.

Отсутствуют отдельные users, passwords, OAuth clients, scopes, audiences,
refresh tokens и token exchange. Все текущие tokens отображаются на стабильный
owner subject `inventory`; одновременный выпуск нескольких tokens обеспечивает
безопасную ротацию без изменения owner-scoped jobs, models и aliases.

Точный bearer не является management identity. При выпуске создаётся отдельный
случайный `token_id`, который используется командами list/revoke и может
безопасно присутствовать в административном output. Знание `token_id` не
позволяет аутентифицироваться.

Текущий внешний формат `a.<base64url>`, entropy, административные команды и
трёхмесячный срок действия остаются зафиксированы ADR 0018 и operations
contract.

### Persistence и lifecycle

PostgreSQL хранит только SHA-256 digest Bearer credential и его metadata:
`token_id`, owner subject, `created_at` и `expires_at`. Исходный Bearer
показывается один раз при issue и не сохраняется в database, logs, errors или
cache entries.

SHA-256 используется не как password hash, а как необратимый lookup key для
credential с криптографически случайной высокой entropy. Медленный
password-oriented KDF не добавляет полезной защиты для такого token и не
используется.

Expiration фиксируется при issue. Успешные обращения, cache hits и ротация
других tokens не продлевают `expires_at`; sliding expiration отсутствует.
Expired row остаётся видимой оператору до явного revoke.

Revoke физически удаляет token row. Revoked state, tombstone и встроенная audit
history не сохраняются. Поэтому неизвестный и ранее удалённый `token_id`
неразличимы и одинаково возвращают `not found`.

### Stateful verification и cache

Проверка является stateful и database-backed. Flight process использует
process-local cache-aside с ограниченной capacity и ограниченным временем жизни
entries:

```text
Bearer
  -> SHA-256 digest
  -> bounded process-local cache
     -> hit: subject + fixed expires_at validation
     -> miss: exact digest lookup in PostgreSQL
```

Cache key является digest, а не исходным Bearer. Положительная entry никогда не
живёт дольше собственного `expires_at` и установленного максимального cache
TTL. Eviction меняет только latency следующей проверки и не меняет
authorization semantics.

PostgreSQL `LISTEN/NOTIFY`, notification trigger, полный preload всех активных
tokens и отдельный listener connection не используются. Новый token доступен
на первом cache miss. После физического revoke уже закэшированное положительное
решение может оставаться действительным не дольше максимального cache TTL;
немедленный distributed revoke не является свойством этого класса решения.

При cache miss или истечении entry PostgreSQL обязателен для нового решения.
Database failure не превращается в `UNAUTHENTICATED`: запрос получает
availability error. Ещё действующая положительная entry может использоваться
до своей обычной границы TTL и `expires_at`. Перезапуск процесса полностью
очищает cache.

Текущие implementation constants ограничивают cache 1024 положительными
entries и задают TTL 15 секунд. Следовательно, максимальная revoke latency
равна 15 секундам. Эти bounds проверяются тестами и не меняют fixed token
expiration.

### Security boundary и TLS

Bearer является replayable secret: любой, кто получил его точное значение,
может пользоваться полномочиями token до expiration, revoke или окончания
действующей cache entry. Digest-only storage защищает database dump от
немедленного использования credentials, но не защищает token в памяти
Consumer-а, command output или network transport.

Transport mode определяется только certificate options: полная пара
`--tls-cert-file`/`--tls-key-file` включает TLS, а отсутствие обоих options
означает plaintext. Отдельного plaintext switch нет. При передаче Bearer через
недоверенную сеть deployment обязан использовать TLS с проверкой identity
server-а; plaintext сам по себе не обеспечивает confidentiality. mTLS может
дополнительно защищать канал, но не заменяет Bearer lifecycle и owner identity.

Credentials не помещаются в repository, shell history, command arguments,
structured logs или exception text. Operator передаёт новый Bearer Consumer-у
через принятый secret channel и отзывает прежний только после подтверждённой
ротации.

## Альтернативы

### OAuth 2.0/OIDC provider

Подходит при federation, delegated authorization, users, scopes и нескольких
services. Для текущего класса системы создаёт лишние identity и availability
planes, administration credentials и network introspection. Испытанный вариант
и причины отказа сохранены в ADR 0017.

### Self-contained JWT

Убирает database lookup на verification path, но переносит authorization state
в claims, вводит signing key lifecycle и делает revoke зависимым от denylist
или короткого expiration. Эти свойства сложнее требуемой stateful модели.

### PostgreSQL lookup на каждый RPC

Имеет минимальную revoke latency и простую authoritative semantics, но добавляет
database round trip к каждому Flight RPC. Bounded cache-aside сохраняет
database authority и ограничивает нагрузку ценой явно ограниченной revoke
latency.

### Полный cache через `LISTEN/NOTIFY`

Даёт быстрое распространение issue/revoke, но требует полного набора активных
tokens в каждом процессе, notification trigger, постоянный listener connection
и reconnect/reload lifecycle. Для малого числа Consumers cache-aside проще, а
bounded revoke latency принимается явно.

### Хранение исходного Bearer или token file

Raw credential в database превращает read-only утечку в немедленный доступ.
Token file создаёт второй persistence и deployment contract рядом с
PostgreSQL. Оба варианта отклонены.

## Граница ADR

ADR закрепляет класс authentication решения, authority, credential model,
expiration/revoke semantics, cache consistency и channel security. Точный
format CLI output, SQL migration steps, размеры cache, cache TTL и расположение
Python modules остаются implementation/operations details при сохранении
указанных инвариантов.

ADR 0018 остаётся историей текущего CLI, token format, PostgreSQL schema и
migrations. Его решение о полном `LISTEN/NOTIFY` index заменено этим ADR.

## Состояние реализации

Digest-only persistence, отдельный `token_id`, fixed expiration, hard delete,
отсутствие OAuth claims и bounded cache-aside реализованы. Migration `0019`
удаляет token notification trigger и функцию. Runtime не выполняет preload, не
создаёт listener connection и не использует PostgreSQL `LISTEN/NOTIFY`.
Transport выбирается только наличием полной пары certificate/key options.

## Последствия

- Transformer остаётся владельцем credential lifecycle и обязан безопасно
  реализовывать generation, storage, rotation, expiration и revoke.
- PostgreSQL остаётся единственным долговечным authority authentication state;
  отдельный identity service не требуется.
- Issue виден на первом cache miss, но revoke имеет документированную верхнюю
  границу задержки 15 секунд.
- Hard delete минимизирует metadata, но исключает встроенную revoke audit
  history.
- Database availability требуется на cache miss; cache не становится вторым
  долговечным источником истины.
- Компрометация Bearer неотличима от легитимного Consumer-а, поэтому TLS и
  external secret storage являются обязательными частями security boundary.
- Появление users, scopes, delegated access или federation требует нового ADR,
  а не постепенного расширения локального token format.
