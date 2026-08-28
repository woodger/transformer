# Интеграция Consumer-ов с Transformer Arrow Flight v6

> Тип: интеграционное руководство. Durable workflow Consumer-а поверх
> текущего протокола.

Нормативный wire-контракт находится в
[`app/contracts/flight/v6`](../app/contracts/flight/v6/README.md). JSON Schema и
эталонные фикстуры из этого каталога имеют приоритет над данным руководством.
Эксплуатация сервиса и восстановление описаны в
[`руководстве по эксплуатации Flight`](operations/flight-service.md). Выдача,
передача, ротация и отзыв credentials описаны в
[`операционном руководстве`](operations/api-access-tokens.md), а credential
model, token verification и bounded revoke latency — в
[`справочнике аутентификации`](authentication.md).
Rationale единой cross-language indicator identity сохранён в
[ADR 0015](adr/0015-unified-indicator-identity-flight-v5.md); текущие значения
задаёт нормативный Flight contract.

Transformer Flight v6 — единственный текущий удалённый API. Consumer должен
требовать `protocolVersions`, равный `[6]`, и использовать нормативные actions,
descriptors и семантику состояний v6. V5 actions, descriptors, semantic aliases
и fallback отсутствуют.

## Транспорт и аутентификация

Используйте Arrow Flight `DoAction`, `DoPut`, `GetFlightInfo` и `DoGet`. Каждый
RPC содержит:

```text
authorization: Bearer a.<base64url>
```

Credential представляет одного owner subject, и доступ к job/model resources
ограничен этим owner-ом. Никогда не записывайте bearer credential или
непрозрачный output ticket в логи. Expiration, rotation и revoke procedure
задаёт [справочник аутентификации](authentication.md).

Точный action envelope, transport methods и error representation задаёт
[Flight contract](../app/contracts/flight/v6/README.md#конверт-и-аутентификация).
Для каждой мутации заранее сохраните стабильный `idempotencyKey` и канонический
request. Повторяйте потерянный запрос с тем же содержимым; не используйте ключ
для другой операции.

## Проверка capabilities

Полный список actions и их точные schemas задаёт
[Flight contract](../app/contracts/flight/v6/README.md#actions).

При запуске вызовите `capabilities` и завершитесь с ошибкой, если
`protocolVersions` не равен `[6]`. Одновременно проверьте объявленный
`mlContract`; несовпадение semantic IDs является ошибкой совместимости до
создания job. Эффективные лимиты из ответа являются
нормативными для текущего runtime; не копируйте значения по умолчанию из
репозитория в код Consumer-а.

## Выбор устройства

В create передавайте `device: "cpu"`, `device: "gpu"` или `device: "auto"`.
Явный `gpu` требует доступный GPU и не подменяется CPU; `auto` выбирает GPU при
наличии и CPU в противном случае. Значение `cuda` не поддерживается.

Доступность и capacity проверяйте через `devices.gpu` и `queue.gpuCapacity`.
Consumer не должен зависеть от конкретного GPU runtime: его выбор и mapping во
внутренний worker contract принадлежат Transformer.

## Сохранение идентичности до create

До первого сетевого запроса Consumer создаёт и надёжно фиксирует в своём
durable state следующие значения:

- `jobId` — стабильная удалённая идентичность логического job;
- `clientExecutionId` — идентичность текущего claim Consumer-а;
- `idempotencyKey` для create и неизменяемый документ create.

Fit create соответствует фикстуре
[`create-fit.request.json`](../app/contracts/flight/v6/fixtures/json/create-fit.request.json).
Predict create соответствует
[`create-predict.request.json`](../app/contracts/flight/v6/fixtures/json/create-predict.request.json).
Predict передаёт ровно один `modelRef` или ограниченный owner-ом `modelAlias`.
Transformer атомарно разрешает alias и возвращает неизменяемый
`resolvedModelRef`.

Начальный результат create:

```text
input.state     = OPEN
execution.state = WAITING_INPUT
fencingToken    = "1"
```

Сохраните результат целиком, особенно `resolvedModelRef`, сведения о владении
и лимиты загрузки. Если ответ потерян, повторите тот же логический create.
`jobId` создаётся клиентом, а Transformer сохраняет его компактную идентичность
после удаления тяжёлых данных job, поэтому отдельный resolve-вызов не нужен.

## Контракты данных и objective

Точные формы `dataContract` и `mlContract`, текущие semantic IDs и примеры
create принадлежат
[Flight contract](../app/contracts/flight/v6/README.md#ml-контракт-данных-и-модели)
и его fixtures. Не переносите их копию в Consumer как независимую
спецификацию.

Consumer владеет каноническим data-contract document: упорядоченными
идентичностями features, target semantics, нормализацией, missing-value policy
и `profile`. Transformer рассматривает `profile` как ограниченную,
регистрозависимую строку, сохраняет её без преобразования и включает в
identity. Predict create передаёт весь точный `dataContract`, для которого
сертифицирована выбранная модель.

Для fit `objectiveConfigSha256` вычисляется по фактической `trainingConfig`.
Точная форма документа задана
[`objective-config.schema.json`](../app/contracts/flight/v6/schemas/objective-config.schema.json),
а нормативная cross-language пара документ/digest — фикстурами
[`objective-config.fit.json`](../app/contracts/flight/v6/fixtures/json/objective-config.fit.json)
и create-fit fixture. Документ канонизируется строго по
[RFC 8785/JCS](https://www.rfc-editor.org/rfc/rfc8785.html), после чего SHA-256
вычисляется над полученными UTF-8 bytes. Cross-language проверка находится в
[`objective_config_sha256.mjs`](../app/contracts/flight/v6/fixtures/objective_config_sha256.mjs).
Transformer независимо строит тот же документ и отклоняет несовпадение до
создания job.

Predict должен передать точный `mlContract`, возвращённый `model.describe` для
выбранной модели. Нельзя подставлять только IDs из capabilities: конкретный
`objectiveConfigSha256` является свойством обученной модели.

## Межсистемное fencing и перехват владения

Каждая загрузка, close и cancel содержит текущую пару:

```json
{
  "clientExecutionId": "UUID",
  "fencingToken": "7"
}
```

Token представляет собой каноническую положительную десятичную строку, а не
JSON integer. При перехвате lease Consumer вызывает
`transformer.v6.job.acquire` с предыдущим execution ID, ожидаемым token и новым
execution ID. Transformer атомарно сравнивает старую пару и возвращает
следующий token. Сохраните этот ответ до выполнения мутаций от нового claim.

Устаревший owner получает `STALE_FENCE`. Fence проверяется до начала приёма
данных DoPut и повторно непосредственно перед надёжной фиксацией receipt,
поэтому запрос, пересёкшийся по времени с перехватом владения, не может
перезаписать вход нового owner-а.

## Загрузка

Один DoPut представляет один семантический payload. Границы RecordBatch нужны
только для транспортного разбиения. Descriptor, application metadata и точные
Arrow schemas берите из разделов
[«Загрузка»](../app/contracts/flight/v6/README.md#загрузка) и
[«Arrow-схемы»](../app/contracts/flight/v6/README.md#arrow-схемы) нормативного
контракта.

Завершение DoPut не по порядку разрешено. Сервер возвращает один PutResult
только после надёжной фиксации неизменяемого artifact и PostgreSQL receipt.
Сохраните PutResult целиком. `nextInputOrdinal` — первый отсутствующий ordinal;
`inputRevision` — монотонная commit revision конкретного job; `queued=true`
означает, что эта фиксация автоматически поставила execution в очередь.

Первый непустой непрерывный префикс ставит job в очередь. Поэтому fit worker
может находиться в состоянии `RUNNING`, пока input остаётся `OPEN`. Порядок и
время поступления не меняют логический порядок worker-а: сначала ordinal, затем
строка внутри payload.

## Сверка входных данных по revision

Если PutResult потерян, не считайте транспортную ошибку доказательством
неуспешной фиксации. Выполните сверку через
`transformer.v6.job.inputs.list`. Первая страница передаёт `afterRevision` и
фиксирует возвращённый `snapshotRevision`. Продолжайте с тем же snapshot и
возвращённым cursor, пока выполняется:

```text
cursor < commitRevision <= snapshotRevision
```

После обхода сохраните его `snapshotRevision` как следующий `afterRevision`.
Payload с малым ordinal, зафиксированный позднее, получает более высокую
revision и поэтому появляется при следующем обходе. Размер страницы не
превышает 100 записей.

Если receipt отсутствует, payload можно повторить с теми же `payloadId`,
ordinal, schema, rows и данными. Точный дубликат зафиксированного payload
возвращает существующий receipt; другое содержимое для занятой identity или
ordinal является конфликтом.

## Закрытие входа

Close обозначает EOF, а не команду запуска. Рассчитайте канонический digest по
всем server receipts, отсортированным по ordinal. Точный набор покрытых полей и
close request задаёт
[Flight contract](../app/contracts/flight/v6/README.md#закрытие-входа-и-digest-манифеста).
Исключите `commitRevision`, timestamps и порядок поступления и передайте только
каноническую сводку.

До фиксации `input.state=CLOSED` Transformer проверяет непрерывность ordinal
`0..payloadCount-1`, итоговые значения, digest, единую физическую Arrow-схему и
единый digest контракта данных. После этого новые загрузки отклоняются.

Закрытие пустого fit завершается с `EMPTY_INPUT` и оставляет input в состоянии
`OPEN`. Пустой predict допустим: ноль payload-ов даёт ноль outputs.
Типизированный пустой predict payload даёт один типизированный пустой output с
запрошенной колонкой prediction.

## Состояния и polling

Status предоставляет независимые input и execution state axes; точные enums и
terminal rules задаёт
[Flight contract](../app/contracts/flight/v6/README.md#состояние).

Сочетание `OPEN + RUNNING` является нормальным. Для terminal success требуется
закрытый input и атомарная публикация. Выполняйте polling не чаще, чем указано
в `pollAfterMs`. Размер status ограничен; он содержит счётчики, а не все input
или output receipts.

Для выполняющегося fit status после durable global epoch возвращает компактный
checkpoint-aligned progress. AMP, gradient, target metrics и timings остаются
только в telemetry. Если core progress ещё недоступен, Consumer может
использовать `recovery.latestCheckpoint.completedEpochs` и `globalStep` как
fallback.

До EOF после сбоя fit attempt незавершённая нулевая epoch повторяется с начала;
надёжно зафиксированные inputs сохраняются. После EOF recovery checkpoints
находятся на границах полных global epochs. Consumer должен корректно
обрабатывать `RETRYING`, не отправляя уже зафиксированные payloads повторно.

## Результаты prediction

Результаты доступны только после `execution.state=SUCCEEDED`:

1. Получите все страницы `transformer.v6.job.outputs.list`.
2. Для каждого ordinal вызовите `GetFlightInfo` с нормативным output
   descriptor.
3. До expiration используйте возвращённый непрозрачный ticket в `DoGet`.

Descriptor, ticket semantics и точную output schema задаёт
[Flight contract](../app/contracts/flight/v6/README.md#доступ-к-результатам).
Проверьте полученную Arrow schema до обработки строк. Значения уже находятся в
target-space, поэтому Consumer вычисляет per-target метрики напрямую, но не
использует общую MAE/MSE по разнородным координатам как quality score.

Transformer отклоняет неканоническую входную схему с `INVALID_ARGUMENT` до
резервирования или фиксации payload. Job остаётся нетерминальным с открытым
входом, поэтому Consumer может исправить схему и повторить тот же ordinal в
новой транспортной попытке.

Transformer может готовить результаты локально для attempt при открытом входе,
но частичный output не становится видимым. Все output receipts и `SUCCEEDED`
фиксируются одной terminal transaction.

## Модели

Разрешите owner-scoped alias через `model.describe` и сохраните возвращённый
immutable `modelRef`. Для predict используйте этот точный reference и полный
`dataContract`/`mlContract` из ответа. `predictionColumn` относится к prediction
job, а не к модели. Exact request, response и lifecycle error codes принадлежат
[Flight contract](../app/contracts/flight/v6/README.md#ml-контракт-данных-и-модели).

## Решения о повторных запросах

| Потерянный или неуспешный шаг | Поведение Consumer-а |
| --- | --- |
| Ответ create | Повторить тот же логический create; `jobId` уже известен |
| Ответ acquire | Повторить acquire с тем же idempotency key |
| Ответ DoPut | Сверить input receipts и повторить точный payload, только если он отсутствует |
| Ответ close | Повторить close с тем же idempotency key и сводкой |
| Ответ cancel | Повторить cancel с тем же idempotency key и текущим fence |
| Status/list/model describe | Повторить как read-only запрос |
| Истечение GetFlightInfo/ticket | Получить новый ticket после terminal success |
| Прерывание DoGet | Получить новый ticket и начать загрузку данного output заново |
| `STALE_FENCE` | Прекратить мутации старого claim; выполнять acquire только при перехвате lease |

`requestId` может меняться между транспортными попытками. Стабильные
идентичности job, payload, ownership и idempotency меняться не должны.

## Проверка интеграции

1. Проверьте аутентифицированные `capabilities` и `health`; требуйте
   `protocolVersions`, равный `[6]`, и точные semantic IDs `mlContract`.
2. Проверьте идемпотентный повтор create и перехват владения.
3. Проверьте DoPut с fencing, пагинацию по revision и сверку после потери
   `PutResult`.
4. Проверьте, что неверный `objectiveConfigSha256` отклоняется до создания fit.
5. Проверьте streaming fit, закрытие EOF и terminal-публикацию модели.
6. Проверьте получение target-aligned outputs и прямой расчёт всех шести
   per-target metrics.
7. Проверьте, что модель прежнего objective не принимается для predict.
