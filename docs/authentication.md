# Аутентификация Arrow Flight

> Тип: справочник. Текущая credential model, проверка токенов и security
> boundary Transformer Arrow Flight service.

Нормативное требование к заголовку каждого RPC находится в
[`app/contracts/flight/v5`](../app/contracts/flight/v5/README.md), а команды
выпуска, ротации и отзыва — в
[`операционном руководстве`](./flight-operations.md#токены-доступа-api).

## Граница системы

Transformer использует local opaque Bearer API access tokens для одного
контролируемого service Consumer-а. Transformer самостоятельно выпускает и
проверяет credentials, а PostgreSQL является долговечным authority их
состояния.

Эта модель рассчитана на следующие условия:

- provider и Consumer находятся под единым операционным контролем;
- service identities известны заранее и их мало;
- нет интерактивных пользователей и сторонних clients;
- полномочия не делегируются;
- не требуются scopes, audiences, federation, SSO или refresh tokens;
- operator-managed issue, rotation и revoke достаточны для lifecycle.

В системе нет users, passwords, OAuth clients, token exchange или embedded
claims. Появление разных permissions, delegated access, сторонних clients,
нескольких административных доменов или обязательного централизованного audit
credentials означает, что authentication model нужно проектировать заново с
поддерживаемым identity provider. Локальный token format не расширяется
постепенно до собственного OAuth-протокола.

## Credential и owner identity

Credential имеет форму `a.<base64url>` и содержит 64 случайных байта. Он не
является JWT и не несёт subject или authorization claims. Все действующие
credentials соответствуют стабильному owner subject `inventory`; несколько
одновременно выпущенных tokens нужны только для ротации и видят один
owner-scoped state jobs, models и aliases.

При выпуске создаётся отдельный UUID `token_id`. Это management handle для
`list` и `revoke`, а не credential: знание ID не позволяет пройти
аутентификацию. Исходный Bearer показывается только один раз в успешном выводе
`issue`.

Каждый token получает фиксированный `expires_at` через три календарных месяца
после выпуска. Использование token и ротация других credentials не продлевают
срок; sliding expiration отсутствует.

## Persistence и lifecycle

PostgreSQL хранит только:

- SHA-256 digest credential;
- `token_id`;
- owner subject;
- `created_at`;
- nullable `last_used_at`;
- `expires_at`.

SHA-256 служит lookup key для криптографически случайного credential высокой
энтропии, а не password hash. Исходный Bearer не сохраняется в PostgreSQL,
ORM, cache entries, logs или errors.

`revoke` физически удаляет строку. Revoked state, tombstone и встроенная
история отзывов отсутствуют, поэтому удалённый и никогда не существовавший
`token_id` неразличимы и одинаково возвращают `not found`. Истёкшая строка не
удаляется автоматически и остаётся доступной оператору до явного `revoke`.

## Проверка и process-local cache

Каждый новый RPC предъявляет Bearer credential. Flight process вычисляет его
SHA-256 digest и проверяет положительный process-local cache-aside. Cache:

- пуст после запуска процесса и ничего не загружает заранее;
- содержит не более 1024 положительных entries;
- хранит digest, subject и фиксированный `expires_at`, но не исходный Bearer;
- повторно проверяет credential в PostgreSQL не позднее чем через 60 секунд;
- никогда не использует entry после `expires_at`;
- объединяет одновременные misses одного digest в один запрос PostgreSQL;
- не сохраняет неизвестные, удалённые и истёкшие tokens;
- не использует PostgreSQL `LISTEN/NOTIFY`, trigger или отдельное listener
  connection.

На cache miss одна атомарная операция PostgreSQL проверяет digest и
`expires_at`, обновляет `last_used_at` и возвращает owner identity. Cache hits
не записываются в БД. Поэтому `last_used_at` показывает последнюю persisted
revalidation, может отставать от фактического последнего RPC максимум на
60 секунд и не является audit timestamp каждого обращения. LRU eviction может
вызвать revalidation раньше.

Новый token доступен на первом cache miss. После `revoke` ранее закэшированное
положительное решение может приниматься ещё максимум 60 секунд. Это
осознанная bounded revoke latency; немедленный distributed revoke не входит в
security contract. Перезапуск процесса очищает cache.

PostgreSQL обязателен на cache miss или после истечения entry. Ошибка БД
возвращает availability error, а не маскируется как неверный credential.
Действующая положительная entry может использоваться до собственной границы
60 секунд или `expires_at`.

Аутентификация выполняется на входе в RPC. Expiration или revoke не прерывает
уже авторизованный streaming RPC; следующий RPC проходит новую проверку.

## Transport и работа с secret

Bearer является replayable secret. Получивший точное значение может действовать
от имени `inventory` до expiration, revoke или окончания уже закэшированного
решения. Digest-only storage снижает последствия чтения database dump, но не
защищает credential в памяти Consumer-а, command output или network transport.

Transport определяется только certificate options команды `flight serve`:

- полная пара `--tls-cert-file`/`--tls-key-file` включает TLS;
- отсутствие обоих options выбирает plaintext;
- отдельного plaintext switch нет;
- mTLS дополняет channel security, но не заменяет Bearer identity.

При передаче через недоверенную сеть deployment обязан использовать TLS и
проверять identity server-а. Plaintext допустим только внутри явно доверенной
transport boundary. Credential передаётся Consumer-у через принятый secret
channel и не помещается в repository, shell history, command arguments,
structured logs или exception text.
