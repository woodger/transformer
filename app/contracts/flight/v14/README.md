# Контракт Arrow Flight v14

> CONTRACT DOCUMENT. Этот каталог задаёт нормативный публичный
> consumer-neutral wire contract Transformer.

Flight v14 является clean-cut action boundary: v13 actions и compatibility
aliases не принимаются. Durable job lifecycle, `indexedFeatureBlocks` и ML
граница с полным `ModelContract` из
[`semantic/v2`](../../semantic/v2/README.md) сохраняются. Transformer не интерпретирует
Consumer target identities, profile или data identity.

## Actions и документы

```text
transformer.v14.capabilities
transformer.v14.health
transformer.v14.job.create
transformer.v14.job.acquire
transformer.v14.job.status
transformer.v14.job.inputs.list
transformer.v14.job.input.close
transformer.v14.job.outputs.list
transformer.v14.job.cancel
transformer.model-catalog.v2.list
transformer.model-catalog.v2.detail
transformer.training-telemetry.v2.report
transformer.training-telemetry.v2.gradient-interactions
```

Все JSON documents используют UTF-8 и закрытые Draft 2020-12 schemas. Job,
health и capabilities actions передают `contract=transformer-flight`,
`version=14` и canonical UUID `requestId`. Model Catalog actions используют
независимые `contract=transformer-model-catalog`, `revision=2` и документы из
[`model_catalog/v2`](../../model_catalog/v2/README.md). Request/result schemas
этого пакета и независимого
[`training_telemetry/v2`](../../training_telemetry/v2/README.md) нормативны;
человекочитаемый текст не расширяет их.

`job.create` всегда содержит opaque `dataContract`, полный `modelContract` и
`sourceEncoding`. Fit дополнительно содержит `modelLabel`, `initialization` и
может переопределить training/diagnostics policy. Predict содержит ровно один
`modelRef` или `modelAlias`. Service materialize-ит все operational defaults,
валидирует semantic model, вычисляет D1 digests и `jobConfigSha256` до durable
input upload.

Точный resolved preimage `jobConfigSha256` задан
`job-config.schema.json`: fit и predict имеют разные закрытые формы и не
включают request/idempotency/job/fencing identities, selected physical device,
progress или receipts. Golden JCS vectors находятся в
`fixtures/json/job-config.*.json`.

`dataContract` содержит только opaque Consumer identity, revision, profile,
Consumer-owned digest и `seqLen/featureDim`. Значения не имеют специальных
литералов Transformer. `seqLen/featureDim` обязаны совпасть с `modelContract`,
а `sourceEncoding.featureBlocks` непрерывно покрывает `featureDim`.

## Capabilities

`capabilities-result.schema.json` объявляет Flight v14, Worker v13,
checkpoint/recovery v7, сохранённые physical Arrow schema IDs и hybrid objective
language capabilities. Primitive arrays и architectures имеют canonical ASCII
order. Schema проверяет форму; semantic registry определяет, какие primitive
identities известны и доступны deployment-у.

Поле `modelCatalog` публикует полный capability document query revision 2.
Поле `trainingTelemetry` публикует полный capability document Training
Telemetry Query revision 2. Версии обоих query languages после первичной
активации не обязаны меняться вместе с Flight workflow.

`maxTargetSlots`, `maxObjectiveComponents` и `maxPrivateResources` являются
runtime capacity limits. Фиксированного каталога Consumer targets и лимита
шесть в контракте нет.

## Errors

При action или DoPut failure application `code` остаётся префиксом Flight
message для человека. `FlightError.extra_info` содержит exact UTF-8 JSON из
`error-detail.schema.json`; Consumer branching использует `code`, `reason` и
структурные поля, а не message.

Validation происходит до upload в порядке, заданном Semantic v2. Target value
errors указывают opaque identity, вычисленный physical index и logical row.
Compatibility mismatch содержит layer и expected/actual digest. Stored
corruption и correct-but-incompatible request не смешиваются.

## Arrow и `indexedFeatureBlocks`

Физические schemas сохраняют IDs:

```text
transformer.indexed-feature-blocks.fit.v1
transformer.indexed-feature-blocks.predict.v1
transformer.prediction.target-aligned.v3
```

Входные Arrow schemas строятся из validated `sourceEncoding`:

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
  tgt: non-null List<FixedSizeList<Float32>[slots.length]>

transformer.indexed-feature-blocks.predict.v1
  rangeOrdinal, exampleOffset, features: как в fit

transformer.prediction.target-aligned.v3
  <predictionColumn>: non-null FixedSizeList<Float32>[slots.length]
```

Первый feature block начинается с `position=0`; каждый следующий начинается
после `windowRows * nativeRowWidth` предыдущего, а конец последнего равен
`featureDim`. Для logical example `e`, sequence position `s` и блока `b`:

```text
start = observationOffsets[b][e][s]
block = flatten(nativeRows[b][start : start + windowRows[b]])
x[e][s] = concat(block[0], block[1], ...)
```

Все offsets локальны self-contained chunk и не выходят за `nativeRows`.
Длины observation arrays одинаковы между blocks и, для fit, равны длине
`tgt`. Native features допускают NaN, но не infinity; `tgt` и prediction
обязаны быть finite Float32. Null запрещён на любой глубине.

Их математическая reconstruction semantics не изменилась: Consumer передаёт
ordered native Float32 blocks и локальные offsets, Transformer восстанавливает
logical `[rows, seqLen, featureDim]` ограниченными slices. Fit target width и
prediction width теперь выводятся из `modelContract.targetContract.slots`.
Per-slot `tgt` validation использует declared constraints, а не target name.

`sourceEncoding` входит в `jobConfigSha256`, input manifests и recovery
fencing, но не в target/objective/model digests. Provider-neutral golden Arrow
fixtures находятся в `fixtures/indexed-feature-blocks/`. Они фиксируют один и
heterogeneous blocks, независимые offsets, repeated halo при split range,
одинаковый logical order при разных payload boundaries и rejection
отсутствующего native prefix. Specific Consumer profiles в этот package не
входят. `fixtures/manifest.json` и `manifest.sha256` относятся только к
golden fixture bundle для offline conformance; они не являются runtime input,
capability или compatibility fence.

## Compatibility и artifacts

Predict и `publishedModel` требуют exact data, target, objective и model digest
layers. Published model загружает полный state dictionary и создаёт новое
training state. Recovery дополнительно проверяет job config и input manifest.

Flight v14, Worker v13, checkpoint/recovery v7, queries v2 и metrics v6
активируются одним clean cut. Generations, checkpoints, recovery state и
telemetry прежних revisions под новые identities не мигрируют и не получают
compatibility reader; migration 0026 удаляет registry и lifecycle records,
после чего модели обучаются заново. Filesystem artifacts удаляет startup
reconciliation; lifecycle OpenSearch indices относится к deployment procedure.

## Предметный словарь revision

JSON envelope использует `encoding` для physical source, `source` для
requested initialization и Semantic v2 vocabulary. `RequestedInitialization`
содержит только `source` и optional exact `modelRef`; resolved parent lineage приходит
из checkpoint v7 и не принимается как create intent. Model Catalog и Training
Telemetry публикуются как query revision 2. Свойство `kind` и универсальная
замена `type` в accepted public documents отсутствуют.

Arrow schema identities, physical fields и `indexedFeatureBlocks`
reconstruction остаются неизменными. Fixture pairs v13/v14 имеют одинаковые
chunks и expected logical tensors; различается только JSON envelope.
