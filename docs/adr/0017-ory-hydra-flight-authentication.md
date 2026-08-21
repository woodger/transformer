# ADR 0017: аутентификация Flight через Ory Hydra

## Статус

Принято.

## Контекст

Transformer самостоятельно выдавал API credentials, хранил их в PostgreSQL и
обновлял in-process digest cache через `LISTEN/NOTIFY`. Это создавало отдельный
контур identity рядом с общей Ory Hydra и связывало долговечный owner с
короткоживущим credential.

Flight v5 уже передаёт bearer credential в metadata каждого RPC. Его wire
surface, Arrow schemas и job lifecycle менять для смены transport
authentication не требуется.

## Решение

Transformer принимает только opaque OAuth access token, выданный Ory Hydra по
`client_credentials`. Каждый новый Flight RPC проходит одну introspection на
service boundary:

```text
Flight RPC
  -> authorization: Bearer <opaque access token>
  -> POST /admin/oauth2/introspect
  -> active/token_type/audience/scope validation
  -> owner_subject = client_id
  -> существующий owner-scoped use case
```

Проверка применяется к `ListActions`, `DoAction`, `DoPut`, `GetFlightInfo` и
`DoGet`. Worker не получает token и не знает о Hydra.

Единственная runtime-настройка Transformer:

```dotenv
HYDRA_ENDPOINT=http://hp260g9.home:4445
```

Это базовый адрес Hydra Admin API. Путь
`/admin/oauth2/introspect` фиксирован внутри outbound adapter и не является
runtime-настройкой.

Следующие значения являются частью реализации Transformer и не настраиваются
через environment:

```text
audience = transformer
scope = transformer:invoke
timeout = 3000 ms
```

Introspection выполняется `POST` с `Content-Type:
application/x-www-form-urlencoded`; body содержит только поле `token`. Параметр
`scope` в запрос не передаётся. Для положительного решения Transformer требует:

- `active` — boolean `true`;
- `token_type` — регистронезависимое OAuth-значение `Bearer`;
- `client_id` — непустую case-sensitive строку длиной не более 256 символов
  без управляющих символов;
- `aud` — строку либо массив строк с точным элементом `transformer`;
- `scope` — space-separated строку с точным элементом `transformer:invoke`;
- `exp`, если поле присутствует, — целое значение Unix time в будущем.

Owner identity равна точному `client_id`. Access token, его digest и `sub` не
являются identity и не сохраняются. Поэтому новый token того же OAuth client
видит те же jobs, models, aliases и outputs, а другой `client_id` — нет.

Один RPC авторизуется только при входе. Истечение или отзыв token во время уже
начатого streaming RPC не обрывает его; следующий RPC снова выполняет
introspection. Cache в первой реализации отсутствует.

Стабильное отображение ошибок:

| Условие | Flight status |
| --- | --- |
| отсутствующий/malformed Bearer, inactive, expired, revoked или не access token | `UNAUTHENTICATED` |
| active token без требуемого audience или scope | `PERMISSION_DENIED` |
| timeout, network error, non-2xx либо malformed/incomplete Hydra response | `UNAVAILABLE` |

Ошибки и логи не содержат access token, authorization header, client secret,
form body или тело ответа introspection.

Локальная выдача и отзыв API tokens удаляются без fallback. Migration `0014`
необратимо удаляет `api_access_tokens`, notification trigger и функцию. Поля
`owner_subject` уже являются строковыми и не ссылаются на таблицу tokens,
поэтому существующий owner-scoped state не требует изменения schema. Cutover
выполняется только при отсутствии активных jobs и совместно с переходом
Inventory на OAuth client `inventory`.

## Последствия

- Flight v5, Arrow/ML/worker contracts и persisted job lifecycle не меняются.
- Доступность Hydra становится обязательной для начала нового Flight RPC, но её
  сбой не изменяет уже зафиксированное прикладное состояние.
- Transformer не хранит client secret и не управляет OAuth clients.
- Inventory запрашивает token с явными `audience=transformer` и
  `scope=transformer:invoke`; регистрация client задаёт разрешённые значения,
  но не заменяет параметры token request.
