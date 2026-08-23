# Интеграция Inventory с Transformer Arrow Flight v5

> Тип: справочник. Руководство по интеграции Consumer-а с текущим протоколом.

Нормативный wire-контракт находится в
[`app/contracts/flight/v5`](../app/contracts/flight/v5/README.md). JSON Schema и
эталонные фикстуры из этого каталога имеют приоритет над данным руководством.
Эксплуатация сервиса и восстановление описаны в
[`руководстве по эксплуатации Flight`](flight-operations.md), а архитектурное
решения — в [`ADR 0005`](adr/0005-durable-streaming-flight-v3.md),
[`ADR 0007`](adr/0007-target-aligned-flight-v4.md) и
[`ADR 0015`](adr/0015-unified-indicator-identity-flight-v5.md). Transport
authentication закреплена в
[`ADR 0018`](adr/0018-postgresql-api-access-tokens.md).

Transformer Flight v5 — единственный текущий удалённый API. Inventory должен
требовать `protocolVersions`, равный `[5]`, и использовать нормативные actions,
descriptors и семантику состояний v5, описанные ниже. V4 actions, descriptors,
semantic aliases и fallback отсутствуют.

## Транспорт и аутентификация

Используйте Arrow Flight `DoAction`, `DoPut`, `GetFlightInfo` и `DoGet`. V5 не
использует `DoExchange` и `PollFlightInfo`. Каждый RPC содержит:

```text
authorization: Bearer a.<base64url>
```

Credential представляет одного owner subject. Jobs, model aliases, model
references, status, receipts, tickets и outputs ограничены owner-ом. Никогда не
записывайте bearer credential или непрозрачный output ticket в логи.

Каждый документ action начинается с:

```json
{
  "contract": "transformer-flight",
  "version": 5,
  "requestId": "UUID"
}
```

Для мутаций дополнительно требуется стабильный `idempotencyKey`, специфичный
для action. Повтор одного канонического запроса безопасен; повторное
использование ключа для другого запроса отклоняется. Неуспешная операция
представляет собой ошибку Flight RPC. Стабильные application codes входят в
безопасный текст ошибки и terminal status.

## Actions

Сервер объявляет строго следующий список:

| Action | Назначение | Мутация с fencing |
| --- | --- | --- |
| `transformer.v5.capabilities` | Версии, схемы, ML-контракт, лимиты и доступная ёмкость | Нет |
| `transformer.v5.health` | Аутентифицированная проверка liveness/readiness | Нет |
| `transformer.v5.job.create` | Создать fit- или predict-job с заданной клиентом идентичностью | Начальное владение |
| `transformer.v5.job.acquire` | Передать владение Inventory и увеличить fence | Compare-and-swap |
| `transformer.v5.job.status` | Получить ограниченное состояние job и сводку результата | Нет |
| `transformer.v5.job.inputs.list` | Сверить зафиксированные inputs по revision | Нет |
| `transformer.v5.job.input.close` | Зафиксировать EOF и неизменяемую сводку входа | Да |
| `transformer.v5.job.outputs.list` | Получить terminal output receipts | Нет |
| `transformer.v5.job.cancel` | Отменить нетерминальный job | Да |
| `transformer.v5.model.describe` | Разрешить ссылку и описать неизменяемую модель | Нет |

При запуске вызовите `capabilities` и завершитесь с ошибкой, если
`protocolVersions` не равен `[5]`. Одновременно проверьте объявленный
`mlContract`; несовпадение semantic IDs является ошибкой совместимости до
создания job. Эффективные лимиты из ответа являются
нормативными для текущего runtime; не копируйте значения по умолчанию из
репозитория в Inventory.

## Сохранение идентичности до create

До первого сетевого запроса Inventory создаёт и фиксирует в своей транзакции
PostgreSQL следующие значения:

- `jobId` — стабильная удалённая идентичность логического job;
- `clientExecutionId` — идентичность текущего claim Inventory;
- `idempotencyKey` для create и неизменяемый документ create.

Fit create соответствует фикстуре
[`create-fit.request.json`](../app/contracts/flight/v5/fixtures/json/create-fit.request.json).
Predict create соответствует
[`create-predict.request.json`](../app/contracts/flight/v5/fixtures/json/create-predict.request.json).
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

Каждый create содержит принадлежащую Inventory семантическую идентичность:

```json
{
  "dataContract": {
    "id": "inventory.learning-dataset",
    "version": 2,
    "profile": "research-dividend-events-v2",
    "dataContractSha256": "64 lowercase hex characters",
    "seqLen": 10,
    "featureDim": 891,
    "targetSchemaId": "inventory.target.v2"
  }
}
```

Inventory владеет каноническим документом, digest которого указан здесь. В
документ входят упорядоченные идентичности features, семантика target,
нормализация, политика missing values и `profile`. Поле `profile` —
непрозрачная для Transformer, ограниченная и регистрозависимая строка.
Transformer сохраняет и возвращает её без преобразования; поле входит в
identity контракта вместе с остальными полями и покрывается digest. Predict
create должен передать весь точный `dataContract`, для которого сертифицирована
разрешённая модель. Несовпадение любого поля отклоняется до загрузки с
`MODEL_SCHEMA_MISMATCH`.

Кроме `dataContract`, каждый create содержит target-aligned `mlContract`:

```json
{
  "mlContract": {
    "targetSchemaId": "inventory.target.v2",
    "predictionSchemaId": "transformer.prediction.target-aligned.v2",
    "objectiveId": "transformer.objective.target-aligned.v2",
    "objectiveConfigSha256": "64 lowercase hex characters",
    "checkpointFormat": "transformer-checkpoint-v4",
    "targetWidth": 6,
    "predictionSpace": "target"
  }
}
```

Для fit `objectiveConfigSha256` вычисляется по фактической `trainingConfig`.
Точная форма документа задана
[`objective-config.schema.json`](../app/contracts/flight/v5/schemas/objective-config.schema.json),
а нормативная cross-language пара документ/digest — фикстурами
[`objective-config.fit.json`](../app/contracts/flight/v5/fixtures/json/objective-config.fit.json)
и create-fit. Документ канонизируется строго по
[RFC 8785/JCS](https://www.rfc-editor.org/rfc/rfc8785.html), после чего SHA-256
вычисляется над полученными UTF-8 bytes. В частности, JCS использует
ECMAScript-сериализацию чисел, поэтому `1.0` и `1` дают одинаковое
представление. `NaN`, `Infinity` и другие значения вне I-JSON запрещены.
Нормативный digest fit fixture:
`ae695d62d643a636d5daaef31275c83eb8baa75e354e70368061c1997b39c2fb`.
Node.js-проверка находится в
[`objective_config_sha256.mjs`](../app/contracts/flight/v5/fixtures/objective_config_sha256.mjs).
Transformer независимо строит тот же документ и отклоняет несовпадение до
создания job.

На максимальном loss stage каждая из шести координат имеет прямой supervised
loss. `directLossWeights` содержит шесть положительных весов. Поле `selection`
имеет два режима:

- object `{minDelta, patience}` — best-checkpoint и early stopping работают
  только по полным stage-4 epochs и только по глобально агрегированным
  `L0…L5`;
- `null` или отсутствие поля — выполняется заданное число epochs и публикуется
  последний checkpoint максимального stage.

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
JSON integer. При перехвате lease Inventory вызовите
`transformer.v5.job.acquire` с предыдущим execution ID, ожидаемым token и новым
execution ID. Transformer атомарно сравнивает старую пару и возвращает
следующий token. Сохраните этот ответ до выполнения мутаций от нового claim.

Устаревший owner получает `STALE_FENCE`. Fence проверяется до начала приёма
данных DoPut и повторно непосредственно перед надёжной фиксацией receipt,
поэтому запрос, пересёкшийся по времени с перехватом владения, не может
перезаписать вход нового owner-а.

## Загрузка

Один DoPut представляет один семантический payload. Границы RecordBatch нужны
только для транспортного разбиения. Используйте:

```text
pathDescriptor("transformer", "v5", "jobs", jobId, "inputs", ordinal)
```

До RecordBatch запишите одно сообщение application metadata:

```json
{
  "contract": "transformer-flight",
  "version": 5,
  "jobId": "UUID",
  "clientExecutionId": "UUID",
  "fencingToken": "7",
  "payloadId": "UUID",
  "ordinal": 0,
  "schemaId": "inventory.sequence.fit.v2",
  "dataContractSha256": "64 lowercase hex characters",
  "rows": 1820
}
```

Физическая схема задана точно:

```text
inventory.sequence.fit.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
  tgt: non-null FixedSizeList<Float32>[6]

inventory.sequence.predict.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
```

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
`transformer.v5.job.inputs.list`. Первая страница передаёт `afterRevision` и
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
всем server receipts, отсортированным по ordinal. Используются поля:

```text
payloadId, ordinal, schemaId, dataContractSha256, rows, batches, bytes,
sha256, schemaFingerprint
```

Исключите `commitRevision`, временные метки и порядок поступления. Передайте
только сводку:

```json
{
  "jobId": "UUID",
  "clientExecutionId": "UUID",
  "fencingToken": "7",
  "payloadCount": 5,
  "totalRows": 9100,
  "totalBytes": 324625520,
  "manifestSha256": "64 lowercase hex characters"
}
```

До фиксации `input.state=CLOSED` Transformer проверяет непрерывность ordinal
`0..payloadCount-1`, итоговые значения, digest, единую физическую Arrow-схему и
единый digest контракта данных. После этого новые загрузки отклоняются.

Закрытие пустого fit завершается с `EMPTY_INPUT` и оставляет input в состоянии
`OPEN`. Пустой predict допустим: ноль payload-ов даёт ноль outputs.
Типизированный пустой predict payload даёт один типизированный пустой output с
запрошенной колонкой prediction.

## Состояния и polling

Status предоставляет две независимые оси состояния:

```text
input.state:
  OPEN | CLOSED | ABORTED

execution.state:
  WAITING_INPUT | QUEUED | RUNNING | RETRYING | CANCELLING |
  SUCCEEDED | FAILED | CANCELLED
```

Сочетание `OPEN + RUNNING` является нормальным. Для terminal success требуется
закрытый input и атомарная публикация. Выполняйте polling не чаще, чем указано
в `pollAfterMs`. Размер status ограничен; он содержит счётчики, а не все input
или output receipts.

Для выполняющегося fit после каждой durable global epoch status возвращает
компактный live progress:

```json
{
  "epoch": 4,
  "step": 2940,
  "loss_stage": 4,
  "loss": -3.149016
}
```

Он фиксируется атомарно с recovery checkpoint. AMP, gradient, target metrics и
timings остаются только в telemetry. Если core progress ещё недоступен,
Inventory может использовать `recovery.latestCheckpoint.completedEpochs` и
`globalStep` как fallback.

До EOF после сбоя fit attempt незавершённая нулевая epoch повторяется с начала;
надёжно зафиксированные inputs сохраняются. После EOF recovery checkpoints
находятся на границах полных global epochs. Inventory должен корректно
обрабатывать `RETRYING`, не отправляя уже зафиксированные payloads повторно.

## Результаты prediction

Результаты доступны только после `execution.state=SUCCEEDED`:

1. Получите все страницы `transformer.v5.job.outputs.list`.
2. Для каждого ordinal вызовите `GetFlightInfo` с:

   ```text
   pathDescriptor("transformer", "v5", "jobs", jobId, "outputs", ordinal)
   ```

3. До истечения срока действия используйте возвращённый непрозрачный ticket в
   `DoGet`.

Схема output:

```text
transformer.prediction.target-aligned.v2
  <predictionColumn>: non-null FixedSizeList<Float32>[6]
```

Значения уже находятся в target-space и сопоставляются target по индексу:

| Индекс | Семантика | Диапазон |
| --- | --- | --- |
| `0` | `MeanReturn` | `[-1, 1]` |
| `1` | `SigmaReturn` | `[0, 1]` |
| `2` | `ProbTP` | `[0, 1]` |
| `3` | `ProbSL` | `[0, 1]` |
| `4` | `VolatilityNext` | `[0, 1]` |
| `5` | `HittingProbTP` | `[0, 1]` |

Все значения конечны. Координаты `ProbTP` и `ProbSL` независимы и не обязаны
давать сумму `1`. Raw logits и private uncertainty scale в output отсутствуют.
Поэтому Inventory вычисляет per-target метрики напрямую, но не использует
общую MAE/MSE по шести разнородным координатам как quality score.

Все поля верхнего уровня используют `nullable=false`. Вложенное дочернее поле
`FixedSizeList` называется `item`, имеет тип `Float32` и использует
`nullable=true`. Это часть точной физической схемы; фактические строки и
дочерние значения с null по-прежнему отклоняются runtime-валидацией значений.
Metadata схемы не входит в физическую identity или `schemaFingerprint`.

Transformer отклоняет неканоническую входную схему с `INVALID_ARGUMENT` до
резервирования или фиксации payload. Job остаётся нетерминальным с открытым
входом, поэтому Inventory может исправить схему и повторить тот же ordinal в
новой транспортной попытке.

Transformer может готовить результаты локально для attempt при открытом входе,
но частичный output не становится видимым. Все output receipts и `SUCCEEDED`
фиксируются одной terminal transaction.

## Модели

`transformer.v5.model.describe` принимает один `modelRef` или ограниченный
owner-ом `modelAlias`. Пока модель существует, её `modelRef` и generation
неизменяемы и не имеют автоматического TTL. После штатного hard delete metadata
модели не сохраняется; минимальный audit record удерживает identity, timestamps
и монотонность generation, но не разрешается через Flight. Consumer должен
идентифицировать существующую модель точным `modelRef`.
`predictionColumn` относится к prediction job, а не к модели. Ответ возвращает
полный `dataContract`, включая `profile`, а
`mlContract.targetSchemaId` и `targetWidth`. Отдельного списка `targets` нет:
порядок и имена шести координат однозначно определяет `inventory.target.v2`.

Стабильные ошибки lifecycle:

| Код | Значение |
| --- | --- |
| `NOT_FOUND` | Нет видимой owner-у идентичности модели |
| `MODEL_UNAVAILABLE` | Metadata существует, но checkpoint отсутствует |
| `MODEL_CORRUPT` | Неверны checkpoint или заявленная current semantic metadata |
| `MODEL_SCHEMA_MISMATCH` | Контракт данных/ML не совпадает либо модель относится к прежнему контракту |

## Решения о повторных запросах

| Потерянный или неуспешный шаг | Поведение Inventory |
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
   `protocolVersions`, равный `[5]`, и точные semantic IDs `mlContract`.
2. Проверьте идемпотентный повтор create и перехват владения.
3. Проверьте DoPut с fencing, пагинацию по revision и сверку после потери
   `PutResult`.
4. Проверьте, что неверный `objectiveConfigSha256` отклоняется до создания fit.
5. Проверьте streaming fit, закрытие EOF и terminal-публикацию модели.
6. Проверьте получение target-aligned outputs и прямой расчёт всех шести
   per-target metrics.
7. Проверьте, что модель прежнего objective не принимается для predict.
