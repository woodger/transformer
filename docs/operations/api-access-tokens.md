# Управление API access tokens

> Тип: операционное руководство. Выдача, просмотр, передача, ротация и отзыв
> Bearer credentials для Consumer-а Inventory.

Это руководство задаёт текущую операторскую процедуру. Устройство credential,
правила persistence, cache consistency и security boundary описаны в
[`справочнике аутентификации`](../authentication.md), а обязательный wire
format — в [`контракте Arrow Flight v5`](../../app/contracts/flight/v5/README.md).

Transformer поддерживает одну service identity с фиксированным owner subject
`inventory`. Одновременно выпущенные tokens принадлежат тому же owner-у и
нужны для ротации, а не для разделения Consumer-ов или permissions.

## Предварительные условия

- Команды выполняются из корня working copy через project `.venv`.
- CLI должен обращаться к той же PostgreSQL database, что и Flight service;
  schema должна быть обновлена согласно
  [`руководству по migrations`](database-migrations.md).
- До выпуска token подготовьте принятые в deployment secret storage и канал
  передачи credential Consumer-у.

## Выпустить credential

```bash
./.venv/bin/python ./app/main.py auth tokens issue
```

Команда выпускает token ровно на 180 суток (`180 × 24` часа) и выводит:

```text
Token ID: <UUID>
Expires: <ISO-8601 timestamp>
Token: a.<base64url>
```

`Token` — Bearer credential. Transformer показывает его только в этом выводе
и сохраняет в PostgreSQL только SHA-256 digest. Если credential потерян,
восстановить его из database или повторно вывести нельзя: выпустите новый
token и отзовите потерянный по `Token ID`.

`Token ID` — management handle для `list` и `revoke`, а не credential. Новый
token доступен Consumer-у на первом cache miss; перезапуск Flight service не
требуется.

## Передать credential Consumer-у

Передайте значение `Token`, но не `Token ID`, через принятый secret channel и
сохраните его в secret storage Consumer-а. Не помещайте credential в
repository, shell history, command arguments, логи или server `.env`.

Consumer предъявляет credential в каждом RPC:

```text
authorization: Bearer a.<base64url>
```

При передаче через недоверенную сеть используйте TLS с проверкой identity
server-а. Bearer является replayable secret; знание его точного значения даёт
доступ от имени `inventory` до expiration или revoke с учётом bounded cache
latency.

## Просмотреть tokens

```bash
./.venv/bin/python ./app/main.py auth tokens list
```

Команда не раскрывает Bearer credentials и показывает колонки `ID`, `Status`,
`Last used` и `Expires`. Возможны только состояния `Active` и `Expired`; до
первого использования `Last used` имеет значение `Never`.

`Last used` отражает последнюю persisted revalidation, а не каждый RPC, и
может отставать от фактического использования максимум на 60 секунд. Это
операционный индикатор, а не точный audit log.

## Выполнить штатную ротацию

1. Выпустите новый credential командой `auth tokens issue`.
2. Сохраните и передайте новый `Token` Consumer-у.
3. Переключите Consumer и подтвердите успешным новым RPC, что credential
   принят. `auth tokens list` может дополнительно показать обновлённый
   `Last used`.
4. Отзовите прежний token по сохранённому `Token ID`.

Оба credentials во время ротации представляют owner `inventory`, поэтому
переключение не меняет owner-scoped jobs, models или aliases. Ротацию нужно
завершить до `Expires` прежнего token.

## Отозвать token

При необходимости сначала найдите management ID без раскрытия credential:

```bash
./.venv/bin/python ./app/main.py auth tokens list
```

Затем передайте именно UUID из колонки `ID`:

```bash
./.venv/bin/python ./app/main.py auth tokens revoke <TOKEN_ID>
```

Успешный `revoke` физически удаляет строку из PostgreSQL. Она сразу исчезает
из `auth tokens list`. Повторный revoke и неизвестный ID возвращают
`not found`, поскольку удалённый и никогда не существовавший ID неразличимы.

Положительное решение, уже находящееся в process-local cache, может принимать
отозванный credential ещё максимум 60 секунд. Уже авторизованный streaming RPC
не прерывается. Перезапуск service для штатного revoke не требуется.

## Обработать expiration

После `Expires` новые RPC с token отклоняются. Истёкшая строка не удаляется
автоматически: она остаётся со статусом `Expired`, пока оператор явно не
выполнит `auth tokens revoke <TOKEN_ID>`.

Чтобы избежать перерыва, выпустите и подключите новый credential до
expiration, проверьте новый RPC и затем отзовите прежний token.

## Проверить полный lifecycle

Для сквозной проверки цепочки управления:

1. Выпустите token и сохраните отдельно `Token ID` и `Token`.
2. Выполните новый Flight RPC с выданным Bearer credential.
3. Убедитесь через `auth tokens list`, что строка присутствует и `Last used`
   перестал быть `Never`.
4. Выполните `auth tokens revoke <TOKEN_ID>`.
5. Убедитесь, что строка исчезла из `auth tokens list`.
6. После окончания 60-секундного cache window убедитесь, что новый RPC с
   отозванным credential завершается как `UNAUTHENTICATED`.

Повторный revoke в этом сценарии должен завершиться ошибкой `not found`.
