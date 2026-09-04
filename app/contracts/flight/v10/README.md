# Контракт Transformer Arrow Flight v10

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
{"contract":"transformer-flight","version":10,"requestId":"UUID"}
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

- `transformer.v10.capabilities`
- `transformer.v10.health`
- `transformer.v10.job.create`
- `transformer.v10.job.acquire`
- `transformer.v10.job.status`
- `transformer.v10.job.inputs.list`
- `transformer.v10.job.input.close`
- `transformer.v10.job.outputs.list`
- `transformer.v10.job.cancel`
- `transformer.v10.model.describe`

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

Flight v10 предоставляет две независимые оси состояния:

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
pathDescriptor("transformer", "v10", "jobs", jobId, "inputs", ordinal)
```

Metadata соответствует `upload-metadata.schema.json`. Один DoPut представляет
один физический payload, а каждая строка Arrow — один self-contained chunk
диапазона. Границы RecordBatch и payload не являются границами логической
обучающей выборки. Metadata заранее объявляет `chunks`, `logicalRows` и массив
`nativeRows` с одним физическим счётчиком на feature block; сервер сверяет их
с принятым Arrow stream до фиксации.

Сервер надёжно публикует неизменяемый candidate, фиксирует его receipt и только
после этого отправляет один PutResult. Receipt дополнительно содержит
фактические `batches`, bytes, digests и границы диапазонов. PutResult содержит
`inputRevision`, первый отсутствующий `nextInputOrdinal` и признак того,
привела ли эта фиксация к автоматической постановке в очередь.

Фиксация не по порядку разрешена. Worker получает только непрерывный префикс
payload `ordinal`, поэтому время поступления не влияет на логический порядок
строк. Пустой payload имеет нулевые `chunks`, `logicalRows` и все элементы
`nativeRows`; его четыре поля границ равны `null`.

`rangeOrdinal` нумерует только успешно переданные диапазоны плотно от нуля.
Диапазоны, пропущенные Consumer-ом при подготовке данных, в эту нумерацию не
входят. `exampleOffset` задаёт индекс первого логического example внутри
диапазона. Следующий непустой chunk либо продолжает тот же range точно с
предыдущего `nextExampleOffset`, либо начинает `rangeOrdinal + 1` с offset
нуля. Один range разрешено делить между RecordBatch и payload; каждый его chunk
при этом остаётся self-contained и может повторить необходимый physical halo.
Sequence никогда не пересекает границу range.

## Закрытие входа и digest манифеста

При закрытии передаются счётчики постоянного размера и `manifestSha256`, а не
полный массив. Digest — SHA-256 от компактного JSON с отсортированными ключами
для следующих полей server receipt, упорядоченных по payload `ordinal`:

```text
payloadId, ordinal, schemaId, dataContractSha256, chunks, logicalRows,
nativeRows, firstRangeOrdinal, firstExampleOffset, lastRangeOrdinal,
nextExampleOffset, batches, bytes, sha256, schemaFingerprint
```

`commitRevision`, временные метки, состояние очереди и порядок поступления
исключены. Перед фиксацией `CLOSED` сервер проверяет непрерывность payload
ordinal, range/example boundaries, `rangeCount`, все физические и логические
итоги, единую Arrow-схему и единый `dataContractSha256`. `totalLogicalRows`
обозначает число восстановленных training/prediction examples;
`totalNativeRows` — сумму физически переданных native feature rows по блокам;
`totalBytes` — размер зафиксированных IPC artifacts.

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

Create содержит закрытый документ физического представления:

```json
{
  "sourceEncoding": {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
      {"position": 0, "windowRows": 100, "nativeRowWidth": 648},
      {"position": 64800, "windowRows": 8, "nativeRowWidth": 6}
    ]
  }
}
```

`featureBlocks` непуст и следует block-major порядку Consumer-owned feature
layout, идентичность которого фиксирует `dataContractSha256`. Первый `position`
равен нулю, каждый следующий равен
`position + windowRows × nativeRowWidth` предыдущего блока, а конец последнего
равен `dataContract.featureDim`. Таким образом блоки без разрывов задают точную
block-major раскладку одного observation. Transformer не интерпретирует
instrument, interval, indicator или causal projection: Consumer вычисляет
native feature rows и выбирает offsets до отправки.

Входные схемы строятся из `sourceEncoding` и заданы точно:

```text
transformer.indexed-feature-blocks.fit.v1
  rangeOrdinal: non-null UInt32
  exampleOffset: non-null UInt64
  features: non-null Struct<
    b0: non-null Struct<
      nativeRows: non-null List<FixedSizeList<Float32>[nativeRowWidth[0]]>,
      observationOffsets: non-null List<FixedSizeList<UInt32>[seqLen]>
    >,
    ...
  >
  tgt: non-null List<FixedSizeList<Float32>[targetWidth]>

transformer.indexed-feature-blocks.predict.v1
  rangeOrdinal, exampleOffset, features: как в fit

transformer.prediction.target-aligned.v3
  <predictionColumn>: non-null FixedSizeList<Float32>[targetWidth]
```

Каждая top-level Arrow row содержит локальный `nativeRows` и список
`observationOffsets` для каждого блока. Длина `observationOffsets` одинакова у
всех блоков и равна числу logical examples этого chunk; в fit ей также равна
длина `tgt`. Для example `e`, sequence position `s` и блока `b` worker берёт:

```text
start = observationOffsets[b][e][s]
featureBlock[b][e][s] =
  flatten(nativeRows[b][start : start + windowRows[b]])
feature[e][s] = concat(
  featureBlock[0][e][s], featureBlock[1][e][s], ...
)
```

Каждый offset локален для своего self-contained chunk и обязан вместе с окном
лежать внутри его `nativeRows`. Offsets разных блоков независимы: например,
часовой last-completed block может перейти на следующее окно отдельно от
минутного anchor block. Непоследовательные offsets допустимы и позволяют точно
представить отбор observations при условии сохранения логического порядка.

Worker читает compact artifacts через memory map и материализует dense
`[rows, seqLen, featureDim]` ограниченными срезами, а не весь dataset. После
реконструкции optimizer batching, bounded shuffle и loss получают тот же
упорядоченный поток примеров независимо от Arrow, chunk и payload boundaries.

Каждое top-level и именованное struct/list поле имеет `nullable=false`.
Дочерние value fields контейнеров имеют каноническую для PyArrow nullability,
но это не разрешает null в данных: ingress-валидация отклоняет null на любой
глубине. Metadata схемы не имеет семантического значения и исключается из
точного сравнения и `schemaFingerprint`.

Неканоническая входная схема отклоняется в DoPut с `INVALID_ARGUMENT` до
создания input reservation, durable artifact или запуска worker. Вход job
остаётся открытым, поэтому тот же ordinal можно загрузить повторно с
канонической схемой.

Строки и элементы с null, а также бесконечности в native features отклоняются.
Native features могут содержать NaN; значения target и prediction должны быть
конечными. Выбранная координата `MeanReturn` находится в `[-1, 1]`, любая
другая — в `[0, 1]`. `targetWidth` равен числу выбранных targets от `1` до `6`.
Типизированный пустой payload содержит каноническую схему и ноль chunks;
RecordBatch с нулём строк не меняет эту counters.

Нормативные fixtures включают обычное разбиение range, одноблочную геометрию
`production-core-v2`, multi-timeframe геометрию `production-core-v6` с
переходом часового окна, пропущенную observation и ошибку отсутствующего
native-prefix. Эти имена служат cross-project fixture identities; profile
остаётся непрозрачной Consumer-owned строкой.

Flight v10 принимает только `indexedFeatureBlocks`; dense schema Flight v9 и
fallback отсутствуют. `sourceEncoding` входит в create request hash, хранится
для durable replay и recovery fencing и возвращается в create/status. Он не
входит в `dataContractSha256`, `objectiveConfigSha256`, checkpoint или model
identity: при численно точном восстановлении logical tensor ML-семантика и
идентичность модели не меняются. `maxRowsPerPayload` применяется к
`logicalRows`, а byte quotas — к физически принятым и сохранённым IPC bytes.

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
`dataContractSha256`. Поэтому parent и текущий fit должны использовать одну
Consumer-owned семантическую привязку target/context slots. Границы временного
периода `from`/`to` не входят в Flight `dataContract` или его digest и могут
отличаться. Несовпадение digest возвращает `MODEL_SCHEMA_MISMATCH` до создания
job.

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

Для принятого warm start значения `parentDataContractSha256` и
`dataContractSha256` совпадают. `publishedModel` является weights-only
initialization, а не продолжением прежнего training run. Worker загружает весь
`state_dict` без частичной выборки и semantic remapping. Optimizer, AMP
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
pathDescriptor("transformer", "v10", "jobs", jobId, "outputs", ordinal)
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
