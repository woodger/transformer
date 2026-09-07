# Контракт Arrow Flight v13

> CONTRACT DOCUMENT. Этот каталог задаёт текущий публичный consumer-neutral
> wire contract Transformer.

Flight v13 является clean-cut action boundary: v12 actions и compatibility
aliases не принимаются. Durable job lifecycle, `indexedFeatureBlocks` и ML
граница с полным `ModelContract` из
[`semantic/v1`](../../semantic/v1/README.md) сохраняются. Transformer не интерпретирует
Consumer target identities, profile или data identity.

## Actions и документы

```text
transformer.v13.capabilities
transformer.v13.health
transformer.v13.job.create
transformer.v13.job.acquire
transformer.v13.job.status
transformer.v13.job.inputs.list
transformer.v13.job.input.close
transformer.v13.job.outputs.list
transformer.v13.job.cancel
transformer.model-catalog.v1.list
transformer.model-catalog.v1.detail
transformer.training-telemetry.v1.report
transformer.training-telemetry.v1.gradient-interactions
```

Все JSON documents используют UTF-8 и закрытые Draft 2020-12 schemas. Job,
health и capabilities actions передают `contract=transformer-flight`,
`version=13` и canonical UUID `requestId`. Model Catalog actions используют
независимые `contract=transformer-model-catalog`, `revision=1` и документы из
[`model_catalog/v1`](../../model_catalog/v1/README.md). Request/result schemas
этого пакета и независимого
[`training_telemetry/v1`](../../training_telemetry/v1/README.md) нормативны;
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

`capabilities-result.schema.json` объявляет Flight v13, Worker v12,
checkpoint/recovery v6, сохранённые physical Arrow schema IDs и hybrid objective
language capabilities. Primitive arrays и architectures имеют canonical ASCII
order. Schema проверяет форму; semantic registry определяет, какие primitive
identities известны и доступны deployment-у.

Поле `modelCatalog` публикует полный capability document query revision 1.
Поле `trainingTelemetry` публикует полный capability document Training
Telemetry Query revision 1. Версии обоих query languages после первичной
активации не обязаны меняться вместе с Flight workflow.

`maxTargetSlots`, `maxObjectiveComponents` и `maxPrivateResources` являются
runtime capacity limits. Фиксированного каталога Consumer targets и лимита
шесть в контракте нет.

## Errors

При action или DoPut failure application `code` остаётся префиксом Flight
message для человека. `FlightError.extra_info` содержит exact UTF-8 JSON из
`error-detail.schema.json`; Consumer branching использует `code`, `reason` и
структурные поля, а не message.

Validation происходит до upload в порядке, заданном semantic v1. Target value
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
входят. `fixtures/manifest.json` и `manifest.sha256` фиксируют byte-identical
Flight bundle для offline copy Consumer-а.

## Compatibility и artifacts

Predict и `publishedModel` требуют exact data, target, objective и model digest
layers. Published model загружает полный state dictionary и создаёт новое
training state. Recovery дополнительно проверяет job config и input manifest.

Модели, checkpoint/recovery v6 и metrics v5, созданные Flight v11/v12 runtime,
сохраняют исполняемость: их semantic и artifact contracts не изменились.
Migration `0025` добавляет только монотонную границу catalog traversal и
индексы; Flight v13 runtime ограничивает размер новых published checkpoints.
Новая PostgreSQL migration для Training Telemetry Query не требуется: query
читает owner-gated registry metadata и provider-owned metrics projection.
OpenSearch layout не меняется.
