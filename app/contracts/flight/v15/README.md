# Transformer Arrow Flight v15

> ДОКУМЕНТ КОНТРАКТА. Этот каталог определяет публичную границу Arrow Flight,
> независимую от внешней предметной семантики. Это единственная
> поддерживаемая revision workflow Flight.

Flight v15 — разрушительный чистый переход. В нём нет actions, aliases и
reader v14. Публичная граница намеренно сокращена: вызывающая система передаёт
намерение модели и геометрию данных; Transformer выпускает identity задания,
владеет исполнением и деталями artifacts и раскрывает лишь сведения, нужные
для загрузки, ожидания и получения результатов.

## Actions

```text
transformer.v15.capabilities
transformer.v15.health
transformer.v15.fit.create
transformer.v15.predict.create
transformer.v15.job.acquire
transformer.v15.job.status
transformer.v15.job.inputs.list
transformer.v15.job.input.close
transformer.v15.job.outputs.list
transformer.v15.job.cancel
transformer.model-catalog.v3.list
transformer.model-catalog.v3.detail
transformer.training-telemetry.v3.report
transformer.training-telemetry.v3.gradient-interactions
```

Все запросы Flight jobs содержат UUID `requestId`; create и mutation requests
также содержат предоставленный вызывающей системой `idempotencyKey`. У fit и
predict отдельные закрытые create schemas. Успешный create возвращает
выпущенные Transformer `jobId` и один непрозрачный `mutationLease`, требуемые
последующим mutation actions и metadata DoPut. Вызывающая система не передаёт
client execution ID, fence counter, schema ID или runtime identity Worker-а.

`fit.create` получает label модели, запрошенное устройство, `dataBinding`,
semantic v3 `modelContract`, training configuration, diagnostics и запрошенную
initialization (`source: random` либо `source: publishedModel` с `modelRef`).
`predict.create` получает только точный `modelRef`, запрошенное устройство и
`dataBinding`; он не передаёт model contract повторно. Его результат содержит
принадлежащее checkpoint-у prediction definition: sequence length, target
output width и упорядоченные непрозрачные target identities с их публичными
transformations.

`dataBinding` содержит непрозрачные `dataContractSha256`, `tensorGeometry` и
`inputLayout`. В нём нет читаемого profile, revision данных, feature catalogue
или внешних предметных metadata.

## Жизненный цикл входных данных

`inputLayout.featureBlocks` объявляет каждый block через `windowRows` и
`nativeRowWidth`. Positions выводятся Transformer, а ширины всех blocks должны
давать в сумме `featureDim`. `indexedFeatureBlocks` — единственный input layout
v15; discriminator source encoding отсутствует.

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

Transformer валидирует semantic v3 document и вычисляет его identities D1 во
время create. Совместимость fit и warm start точна на уровнях data и model
definition. Stored corruption, недопустимые request values и несовместимые
definitions используют разные structured error reasons. Physical validation
checkpoint-а остаётся ответственностью provider-а.

`capabilities` объявляет доступные устройства, limits загрузки и доступность
поверхностей catalog и telemetry query. Он не публикует literals архитектуры,
версии Worker/checkpoint, catalogues primitives или topology хранения.

## Связанные query contracts

Model discovery/detail определён в
[`model_catalog/v3`](../../model_catalog/v3/README.md). Training telemetry
определена в
[`training_telemetry/v3`](../../training_telemetry/v3/README.md).
Оба активируются actions выше, но сохраняют собственные revision и семантику
cursor.

## Чистый переход

Migration `0027_public_contract_simplification` отказывается запускаться при
наличии non-terminal job, затем удаляет jobs, models, recovery records и
привязанную к базе telemetry, которые v15 не может прочитать. Startup
reconciliation удаляет их не имеющие ссылок managed filesystem artifacts.
Индексы метрик OpenSearch v6 должны быть удалены release procedure и созданы
заново из templates v7. Существующие generations недоступны для predict, warm
start, recovery или catalog queries; после deployment обучите новые
generations.

## Fixtures

`fixtures/` покрывает активные формы layout, initialization и разрешённого
job-config. Его manifest хеширует только bundle fixtures для офлайн
межпроектной проверки. Он не участвует в runtime dispatch, request validation
или compatibility модели.
