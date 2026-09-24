# Transformer Arrow Flight v17

> ДОКУМЕНТ КОНТРАКТА. Этот каталог определяет публичную границу Arrow Flight,
> независимую от внешней предметной семантики. Это единственная
> поддерживаемая revision workflow Flight.

Flight v17 — чистый переход action surface: service не объявляет actions,
aliases или reader Flight v16. Вызывающая система передаёт намерение модели и
геометрию данных; Transformer выпускает identity задания, владеет исполнением
и деталями artifacts и раскрывает лишь сведения, нужные для загрузки,
ожидания и получения результатов.

## Actions

```text
transformer.v17.capabilities
transformer.v17.health
transformer.v17.fit.create
transformer.v17.predict.create
transformer.v17.job.acquire
transformer.v17.job.status
transformer.v17.job.inputs.list
transformer.v17.job.input.close
transformer.v17.job.outputs.list
transformer.v17.job.cancel
transformer.model-catalog.v4.list
transformer.model-catalog.v4.detail
transformer.model-topology.v2.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
```

Все запросы Flight jobs содержат UUID `requestId`; create и mutation requests
также содержат предоставленный вызывающей системой `idempotencyKey`. У fit и
predict отдельные закрытые create schemas. Успешный create возвращает
выпущенные Transformer `jobId` и один непрозрачный `mutationLease`, требуемые
последующим mutation actions и metadata DoPut. Вызывающая система не передаёт
client execution ID, fence counter, schema ID или runtime identity Worker-а.

`fit.create` получает label модели, запрошенное устройство, `dataBinding`,
semantic v4 `modelContract`, training configuration, diagnostics и запрошенную
initialization (`source: random` либо `source: publishedModel` с `modelRef`).
`predict.create` получает только точный `modelRef`, запрошенное устройство и
`dataBinding`; он не передаёт model contract повторно. Его результат содержит
принадлежащее checkpoint-у prediction definition: sequence length, target
output width и упорядоченные непрозрачные target identities с их публичными
transformations. Для
`PositiveClassWeightedBinaryCrossEntropyWithLogits` описание также содержит
`positiveClassWeight`: public output в этом случае определён Semantic v4 как
`sigmoid(rawLogit - log(positiveClassWeight))`.

`dataBinding` содержит непрозрачные `dataContractSha256`, `tensorGeometry` и
`inputLayout`. В нём нет читаемого profile, revision данных, feature catalogue
или внешних предметных metadata.

## Жизненный цикл входных данных

`inputLayout.featureBlocks` объявляет каждый block через `windowRows` и
`nativeRowWidth`. Positions выводятся Transformer, а ширины всех blocks должны
давать в сумме `featureDim`. `indexedFeatureBlocks` — единственный input layout
v17; discriminator source encoding отсутствует.

Каждый DoPut — один физический payload. Metadata загрузки содержит только
выпущенное job, непрозрачный lease, identity payload, ordinal, число logical
rows, chunks и native row counts. Transformer выводит и надёжно сохраняет
physical schema identity, schema fingerprint, receipts и фактические totals.
`job.input.close` получает выведенный из receipts `manifestSha256` и может
содержать `expectedLogicalRows` только для раннего выявления EOF; он не
повторяет рассчитанные client-ом totals.

Логическое восстановление не изменилось. Для каждого block Transformer
разрешает локальные offsets observations по native rows, разворачивает
настроенное окно и конкатенирует blocks в `[rows, seqLen, featureDim]`. Fit
дополнительно принимает конечные target vectors, ширина которых выводится из
упорядоченных slots. Output predict — выровненный с target-ами конечный vector
Float32 той же ширины. Этот контракт сохраняет логическое восстановление
tensor-а, но не обещает неизменность сериализованных байтов Arrow IPC.

## Исполнение и artifacts

`job.status` сообщает lifecycle state, input progress, выбранное устройство,
проекцию terminal error/result и подсказку polling. Он не раскрывает версии
протокола Worker, byte counts или hashes checkpoint-а, filesystem paths,
recovery descriptors, worker logs или внутренние fences. `job.outputs.list`
предоставляет ограниченный обход outputs predict.

При любом restart service незавершённые jobs не возобновляются. До запуска
WorkerPool Transformer завершает `WAITING_INPUT`, `QUEUED`, `RUNNING` и
`RETRYING` как `FAILED / EXECUTION_INTERRUPTED`; `CANCELLING` — как
`CANCELLED`. Для новой попытки вызывающая система создаёт новый job. Это не
меняет schema actions, model compatibility или формат recovery checkpoint-а;
recovery остаётся внутренним механизмом retryable сбоя Worker внутри одного
работающего service.

Transformer валидирует semantic v4 document и вычисляет его identities D1 во
время create. Совместимость fit и warm start точна на уровнях data и model
definition. Stored corruption, недопустимые request values и несовместимые
definitions используют разные structured error reasons. Physical validation
checkpoint-а остаётся ответственностью provider-а.

`capabilities` объявляет доступные устройства, limits загрузки, доступность
поверхностей catalog, topology и telemetry query. Он не публикует literals
архитектуры, версии Worker/checkpoint или topology хранения. Блок `semantic`
объявляет закрытую revision и доступные direct primitives.

## Связанные query contracts

Model discovery/detail определён в
[`model_catalog/v4`](../../model_catalog/v4/README.md). Training telemetry
определена в [`training_telemetry/v4`](../../training_telemetry/v4/README.md).
Статическая публичная topology определена в
[`model_topology/v2`](../../model_topology/v2/README.md). Все три поверхности
активируются actions выше, но сохраняют собственные revision и семантику.

## Чистый переход

Migration `0028` применяется только при terminal jobs и удаляет durable state,
несовместимый с Semantic v4 / Flight v17: jobs, published generations,
checkpoint/recovery metadata, idempotency records и database telemetry
metadata. Legacy reader, alias, conversion weights и warm start старых models
отсутствуют; models обучаются заново. Deployment отдельно пересоздаёт metrics
v8 OpenSearch indices по обязательным templates.

Физическая Arrow плоскость не меняется: `indexedFeatureBlocks`, logical
reconstruction и schema IDs fit/predict/output сохраняются. Меняется только
семантика target/objective, checkpoint-owned prediction definition и связанные
identity generation.

## Fixtures

`fixtures/` покрывает активные формы layout, initialization и разрешённого
job-config. Его manifest хеширует только bundle fixtures для офлайн
межпроектной проверки. Он не участвует в runtime dispatch, request validation
или compatibility модели.
