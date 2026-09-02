# Контракт Transformer Arrow Flight v9

Этот каталог содержит нормативный контракт обмена данными между Consumer-ами и
Transformer, не зависящий от языка реализации. JSON Schema, Arrow-схемы и
эталонные фикстуры версионируются вместе. Интеграционный flow описан в
[`consumer-flight-integration.md`](../../../../docs/consumer-flight-integration.md),
эксплуатационный lifecycle — в
[`flight-service.md`](../../../../docs/operations/flight-service.md), а credential
model и channel security — в
[`authentication.md`](../../../../docs/authentication.md).

## Конверт и аутентификация

Каждый запрос и результат action представляет собой JSON-объект в кодировке
UTF-8, содержащий:

```json
{"contract":"transformer-flight","version":9,"requestId":"UUID"}
```

Для каждого RPC требуется заголовок `authorization: Bearer TOKEN`. Credential
формата `a.<base64url>` представляет точный owner subject и действует ровно
180 суток (`180 × 24` часа) с момента выпуска. Неизвестный, отозванный или
истёкший credential отклоняется как `UNAUTHENTICATED`. Мутации также содержат
`idempotencyKey`. Канонический хеш запроса — SHA-256 от компактного JSON с
отсортированными ключами после удаления `requestId` и `idempotencyKey`. Точный
повтор возвращает зафиксированный результат; повторное использование ключа для
другого запроса отклоняется.

## Actions

Сервер объявляет строго следующий список:

- `transformer.v9.capabilities`
- `transformer.v9.health`
- `transformer.v9.job.create`
- `transformer.v9.job.acquire`
- `transformer.v9.job.status`
- `transformer.v9.job.inputs.list`
- `transformer.v9.job.input.close`
- `transformer.v9.job.outputs.list`
- `transformer.v9.job.cancel`
- `transformer.v9.model.describe`

Этот список полностью определяет доступную поверхность actions.
`job.input.close` обозначает EOF. `DoExchange` и `PollFlightInfo` не входят в
контракт.

Схемы запросов и результатов закрыты: неизвестные поля отклоняются.
`action-result.schema.json` содержит закрытое объединение всех результатов
actions. Эталонные документы находятся в `fixtures/json/`.

На runtime-границе каждый action request и DoPut metadata сначала проверяются
соответствующей JSON Schema Draft 2020-12 через `jsonschema`. Эти versioned
schemas являются единственным источником структурных правил: required и
unknown fields, JSON types, enum, patterns, ranges и взаимоисключающие формы
запросов не дублируются Python-проверками. После schema validation inbound
adapter выполняет только семантические проверки между полями и mapping в
типизированные внутренние DTO. Semantic identities регистрозависимы и не
нормализуются.

## Устройство выполнения

Поле `device` принимает только `cpu`, `gpu` или `auto`. `gpu` явно требует
доступный GPU, а `auto` выбирает GPU при наличии и CPU в противном случае.
Явный `gpu` никогда не подменяется CPU. Значение `cuda` не входит в контракт и
отклоняется schema validation.

Capabilities публикует `devices.gpu`, `queue.gpuCapacity` и
`features.deviceAwareGpu`; health публикует `gpu`. Термин `gpu` обозначает
логический класс ресурса и не закрепляет конкретный vendor/runtime за
Consumer-ом. Backend runtime, внутренний worker protocol и его version не
являются частью этой public device identity. Нехватка памяти GPU возвращается
как `GPU_OUT_OF_MEMORY`.

## Состояние

V9 предоставляет две независимые оси состояния:

```text
input.state     OPEN | CLOSED | ABORTED
execution.state WAITING_INPUT | QUEUED | RUNNING | RETRYING |
                CANCELLING | SUCCEEDED | FAILED | CANCELLED
```

Первый зафиксированный непустой непрерывный префикс входных данных
автоматически ставит job в очередь. Worker может работать, пока вход остаётся
открытым. `job.input.close` фиксирует EOF и неизменяемый манифест receipt-ов,
но не является командой запуска. Для успешного завершения и публичного
доступа к результатам вход должен быть закрыт.

Попытка закрыть пустой fit возвращает `EMPTY_INPUT`, не закрывая вход. Пустой
predict допустим: отсутствие payload-ов даёт отсутствие outputs, а один
типизированный пустой payload — один типизированный пустой output.

## Стабильная идентичность и fencing

Consumer создаёт и сохраняет `jobId` до вызова create. После удаления тяжёлых
данных job Transformer сохраняет компактную, ограниченную owner-ом запись об
идентичности. Благодаря этому потерянный ответ create восстанавливается точным
повтором запроса, а UUID нельзя позднее использовать для другого запроса.

Каждая публичная мутация защищена парой `clientExecutionId` и выданным сервером
монотонным `fencingToken`, закодированным десятичной строкой. `job.acquire`
атомарно сравнивает предыдущее владение и увеличивает token. DoPut проверяет
fence перед приёмом данных и повторно — в транзакции фиксации входа. Устаревшая
загрузка может оставить только свой уникальный candidate без ссылок на него;
она не может заменить победивший input artifact.

## Загрузка

Descriptor входных данных:

```text
pathDescriptor("transformer", "v9", "jobs", jobId, "inputs", ordinal)
```

Metadata соответствует `upload-metadata.schema.json`. Один DoPut представляет
один семантический payload; границы RecordBatch используются только для
транспортного разбиения. Сервер надёжно публикует неизменяемый candidate,
фиксирует его receipt и только после этого отправляет один PutResult. PutResult
содержит `inputRevision`, первый отсутствующий `nextInputOrdinal` и признак
того, привела ли эта фиксация к автоматической постановке в очередь.

Фиксация не по порядку разрешена. Worker получает только непрерывный префикс
ordinal, поэтому время поступления не влияет на логический порядок строк.

## Закрытие входа и digest манифеста

При закрытии передаются счётчики постоянного размера и `manifestSha256`, а не
полный массив. Digest — SHA-256 от компактного JSON с отсортированными ключами
для следующих полей server receipt, упорядоченных по ordinal:

```text
payloadId, ordinal, schemaId, dataContractSha256, rows, batches, bytes,
sha256, schemaFingerprint
```

`commitRevision`, временные метки, состояние очереди и порядок поступления
исключены. Перед фиксацией `CLOSED` сервер проверяет непрерывность ordinal,
итоговые значения, единую физическую Arrow-схему и единый
`dataContractSha256`.

## Пагинация

Страницы входных данных упорядочены по `commitRevision` конкретного job. Первый
запрос передаёт `afterRevision` и фиксирует возвращённый `snapshotRevision`.
Следующие страницы повторно используют этот snapshot и удовлетворяют условию:

```text
cursor < commitRevision <= snapshotRevision
```

После завершения обхода следующий начинается с предыдущего `snapshotRevision`
в качестве `afterRevision`. Поэтому payload с малым ordinal, зафиксированный
позже, получает новую revision и не может потеряться. Размер страницы не
превышает 100 записей.

Status содержит только ограниченные по размеру сводные данные. Input receipts
и output descriptors возвращаются соответствующими list actions.

## Arrow-схемы

Физические схемы заданы точно:

```text
inventory.sequence.fit.v3
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]
  tgt: non-null FixedSizeList<Float32>[targetWidth]

inventory.sequence.predict.v2
  src: non-null FixedSizeList<Float32>[seqLen * featureDim]

transformer.prediction.target-aligned.v3
  <predictionColumn>: non-null FixedSizeList<Float32>[targetWidth]
```

Каждое поле верхнего уровня имеет `nullable=false`. У каждого `FixedSizeList`
есть дочернее поле `item` типа `Float32` с `nullable=true`, что соответствует
каноническому представлению PyArrow. Это свойство физической схемы не разрешает
null-значения в ML-данных: ingress-валидация и валидация output отклоняют любую
строку или дочернее значение с null. Metadata схемы не имеет семантического
значения и исключается из точного сравнения и `schemaFingerprint`.

Неканоническая входная схема отклоняется в DoPut с `INVALID_ARGUMENT` до
создания input reservation, durable artifact или запуска worker. Вход job
остаётся открытым, поэтому тот же ordinal можно загрузить повторно с
канонической схемой.

Строки и элементы с null, а также бесконечности отклоняются. `src` может
содержать NaN; значения target и prediction должны быть конечными. Выбранная
координата `MeanReturn` находится в `[-1, 1]`, любая другая — в `[0, 1]`.
`targetWidth` равен числу выбранных targets от `1` до `6`. Фиксированная ширина
списков однозначно задаёт типизированный пустой payload без batch-ей.

## ML-контракт данных и модели

Consumer владеет семантическим документом набора данных. Create передаёт его
`id`, `version`, регистрозависимый `profile`, `dataContractSha256`, `seqLen`,
`featureDim` и `targetSchemaId`. `profile` является непрозрачной ограниченной
строкой, покрывается digest и не перечисляется в capabilities. Transformer
хранит и возвращает весь документ без преобразования, не воспроизводя
семантику features Consumer-а. Predict create отклоняется с
`MODEL_SCHEMA_MISMATCH` до загрузки, если выбранная неизменяемая модель не
сертифицирована для всего точного `dataContract`.

Fit create передаёт `targets`, декларативный `objective` и optional
`diagnostics`. `targets` является непустым подмножеством следующего
канонического порядка:

```text
MeanReturn, SigmaReturn, ProbTP, ProbSL, VolatilityNext, HittingProbTP
```

Порядок выбранных значений сохраняет этот порядок. Transformer валидирует
`{targets, objective}`, создаёт только выбранные public model heads, принимает
`tgt` той же ширины и публикует prediction с теми же координатами. Private
`returnScale` head добавляется только при зависимости objective и не входит в
prediction.

Objective schema v1 закрыта и поддерживает `WeightedSum` с
`GlobalRowMean`, static balancing, строго положительные веса и следующие
operators:

- direct: `SmoothL1`, `BinaryCrossEntropyWithLogits`, `LogMSE`;
- auxiliary: `GaussianNLL`, `ExpectedValue`,
  `RiskAdjustedExpectedValue`.

Direct loss обязан точно соответствовать каждому выбранному target и идти в
том же порядке. `GaussianNLL` требует `MeanReturn`; `ExpectedValue` требует
`ProbTP` и `ProbSL`; `RiskAdjustedExpectedValue` дополнительно требует
`GaussianNLL`. Expected-value operators взаимоисключающие. Все объявленные
losses активны с первого optimizer step; stage schedule отсутствует.

Transformer вычисляет следующий полный `mlContract`:

```text
targetSchemaId         inventory.target.v2
predictionSchemaId     transformer.prediction.target-aligned.v3
objectiveId            transformer.objective.declarative.v1
objectiveConfigSha256  SHA-256 canonical {targets, objective}
checkpointFormat       transformer-checkpoint-v5
targets                выбранные public semantic identities
targetWidth            число выбранных targets
predictionSpace        target
objective              проверенный declarative objective
```

Predict create передаёт этот `mlContract` целиком. Он должен точно совпасть с
опубликованной моделью; другой target subset или objective возвращает
`MODEL_SCHEMA_MISMATCH` до загрузки входа.

Точная форма objective задана `schemas/objective-config.schema.json`.
Фикстура `fixtures/json/objective-config.fit.json` соответствует полю
`objective` из `create-fit.request.json`. Для digest оно оборачивается вместе с
`targets` в объект `{targets, objective}`. Документ канонизируется строго по
[RFC 8785/JCS](https://www.rfc-editor.org/rfc/rfc8785.html), а SHA-256
вычисляется над полученными UTF-8 bytes. JCS использует ECMAScript serialization
для JSON numbers: `1.0` и `1`, а также `0.0` и `0`, дают одинаковые bytes.
Нормативный digest fixture равен
`ffd75380fa9a303f44a38205bd32ae8f63350782f4d306e32796190f955de115`.
Скрипт `fixtures/objective_config_sha256.mjs` независимо вычисляет его в
Node.js; Python и Node.js результаты проверяются одним contract test. Это
нормативная cross-language пара для реализации хеша Consumer-ом.

Публичный prediction совпадает с выбранным target по индексу. Полный
канонический набор имеет вид:

```text
0 MeanReturn       [-1, 1]
1 SigmaReturn      [0, 1]
2 ProbTP           [0, 1]
3 ProbSL           [0, 1]
4 VolatilityNext  [0, 1]
5 HittingProbTP    [0, 1]
```

Индексы в конкретном target vector назначаются после выбора подмножества.
`ProbTP` и `ProbSL` независимы. Raw logits и private Gaussian scale не
пересекают Flight boundary. У каждого выбранного target есть direct
supervision.

`trainingConfig`, `objective` и `diagnostics` являются разными контрактами.
Device и AMP исполняет Transformer. Diagnostics schema v1 может включить
`gradientInteractions.sampleEverySteps`: sampled steps вычисляют нормы
target-task gradients и попарные cosine относительно общей representation
model head. Эти наблюдения не входят в `objectiveConfigSha256`, не балансируют
gradients и не меняют checkpoint compatibility.

## Инициализация fit и lineage модели

Каждый fit create обязан передать ровно один закрытый документ
`initialization`:

```json
{"kind":"random"}
```

или:

```json
{"kind":"publishedModel","modelRef":"mdl_..."}
```

Capabilities возвращает поддерживаемый порядок
`fitInitializations: ["random", "publishedModel"]`. `publishedModel` принимает
только точный неизменяемый `modelRef` того же owner-а; alias, label и путь к
checkpoint не являются допустимыми selectors.
Отсутствующая или принадлежащая другому owner-у generation возвращает
`NOT_FOUND`.

При `publishedModel` Transformer до создания job требует точного совпадения
`modelConfig`, всего `mlContract`, включая `targets` и `objective`, и
структурных полей `dataContract`: `id`, `version`, `profile`, `seqLen`,
`featureDim` и `targetSchemaId`. Отличаться разрешено только
`dataContractSha256`, то есть принадлежащее Consumer-у семантическое наполнение
совместимых target/context slots. Transformer не интерпретирует предметные
идентификаторы и не переставляет feature blocks. Несовместимость возвращает
`MODEL_SCHEMA_MISMATCH` до создания job.

После проверки сервис фиксирует разрешённый lineage:

```json
{
  "kind": "publishedModel",
  "parentModelRef": "mdl_...",
  "parentCheckpointSha256": "<sha256>",
  "parentDataContractSha256": "<sha256>",
  "dataContractSha256": "<sha256>"
}
```

`publishedModel` является weights-only initialization, а не продолжением
прежнего training run. Worker загружает весь `state_dict` без частичной
выборки и semantic remapping. Optimizer, AMP
scaler, RNG, global step, checkpoint selection и recovery state создаются
заново из нового fit request; все параметры модели остаются обучаемыми. Успех
всегда публикует новую immutable generation с новым `modelRef`, точным новым
`dataContract` и не изменяет parent. Predict дочерней модели снова требует
точного совпадения всего её `dataContract`.

Разрешённый `initialization` возвращается в create и status, сохраняется в
PostgreSQL, metadata модели, checkpoint metadata и fit-run telemetry и входит
в неизменяемую конфигурацию job и её recovery fence. В fit responses поле
`resolvedModelRef` равно `null`: точная parent identity находится в
`initialization.parentModelRef`. `model.describe` возвращает lineage
опубликованной generation. Metadata существующего checkpoint-v5 без поля
lineage трактуется как `{"kind":"random"}`.

Recovery незавершённого warm-start fit продолжает именно новый job с его
optimizer и training state; оно не превращается в повторную загрузку training
state parent-а. Явное удаление parent generation блокируется, пока на неё
ссылается незавершённый job.

`modelAlias` ограничен owner-ом и во время create атомарно разрешается в
`resolvedModelRef`. `model.describe` возвращает неизменяемую для существующей
модели generation, digest checkpoint, конфигурацию модели и полный
`dataContract`, включая `profile`, без пути на сервере. После штатного hard
delete рабочая строка и model metadata не сохраняются; отдельный минимальный
audit record удерживает только identity, timestamps и generation history.
Точные `targets` и objective находятся внутри возвращаемого `mlContract`; их
отдельная top-level копия в model response отсутствует. У опубликованных
моделей нет автоматического TTL. Стабильные ошибки:
`NOT_FOUND`, `MODEL_UNAVAILABLE`, `MODEL_CORRUPT` и
`MODEL_SCHEMA_MISMATCH`. Модель прежнего ML-контракта получает
`MODEL_SCHEMA_MISMATCH`; checkpoint текущего format с противоречивой semantic
metadata — `MODEL_CORRUPT`.

## Доступ к результатам

Клиенты получают output descriptors через `job.outputs.list`. GetFlightInfo
использует:

```text
pathDescriptor("transformer", "v9", "jobs", jobId, "outputs", ordinal)
```

Операция разрешена только после `execution.state = SUCCEEDED`. Tickets остаются
случайными, непрозрачными, ограниченными owner-ом и сроком действия. Prediction
artifacts могут подготавливаться при открытом входе, но все outputs публикуются
одновременно с terminal success; частичный успех никогда не виден.

## Потоковая обработка и восстановление

Нулевая epoch читает открытый непрерывный поток. Optimizer batches и
ограниченные shuffle windows пересекают границы RecordBatch и payload. На
текущей границе доступных данных worker сообщает об ожидании; EOF сбрасывает
последнее окно и завершает epoch. Последующие epochs повторно читают закрытый
durable dataset.

Сбой до EOF перезапускает незавершённую epoch с начала. Recovery checkpoints
остаются снимками на границах полных global epochs. `inputIdleTimeout`
отсчитывается только после подтверждения worker-ом ожидания следующего
непрерывного ordinal; фиксация вне порядка не продлевает timeout.

После каждой зафиксированной fit epoch `job.status.progress` содержит только
`epoch`, global `step` и `loss`. Эти три поля фиксируются атомарно с recovery
checkpoint. Расширенная telemetry обучения в status не
публикуется; `recovery.latestCheckpoint.completedEpochs` и `globalStep` можно
использовать как fallback.

При одинаковых упорядоченных данных, seed, deterministic-конфигурации и
hardware/runtime отложенная потоковая подача и полностью закрытый вход должны
давать одинаковый порядок строк и shuffle, optimizer steps, ML-state после
каждой epoch, семантически одинаковый checkpoint и итоговую модель. Wall-clock
telemetry и digest сериализованного файла не являются критериями
эквивалентности.
