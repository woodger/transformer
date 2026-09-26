# Топология модели v3

> ДОКУМЕНТ КОНТРАКТА. Пакет определяет owner-scoped read-only проекцию
> публичной вычислительной topology опубликованной generation для активного
> Flight v22 action `transformer.model-topology.v3.detail`.

## Граница

Запрос принимает только `requestId` и точный `modelRef`. Owner выводится из
аутентифицированного subject. Неизвестная, чужая и удалённая generation
возвращают одинаковый `MODEL_NOT_FOUND`.

Проекция строится Transformer из проверенной metadata published model:
Semantic v5 `ModelContract`, resolved `ModelConfig` и
`modelDefinitionSha256`. Она не читает checkpoint artifact, не повторно
вычисляет его SHA-256 и не доказывает его целостность. Это остаётся задачей
Model Catalog detail.

Ответ содержит точный `modelRef`, provider-issued `modelDefinitionSha256`,
`topologyRevision: 3`, nodes и edges. `modelDefinitionSha256` не включает
topology projection в свой preimage; `topologyRevision` задаёт отдельный язык
её представления.

## Публичный уровень графа

Topology показывает только объявленный высокий уровень исполнения:

- feature input, context preparation, input projection, positional encoding,
  ordered encoder layers и выбор последнего valid state;
- raw target coordinates, обе transformations каждого slot и observed values;
- для weighted binary target — выводимую поправку
  `subtractLogPositiveClassWeight` между raw logit и public transformation;
- abstract private resources, declared direct/auxiliary components и total
  objective;
- `GlobalRowMean`, component weight и `WeightedSum` как необходимые связи
  между epoch telemetry и total loss.

`attentionHeadCount` и `encoderNormalizationOrder` находятся на node encoder
layer. Последнее значение повторяет exact `ModelConfig` generation и не
выводится из layer count. Отдельные attention head nodes не публикуются: они
не являются самостоятельными public ports текущей модели.

Не публикуются PyTorch module/class paths, parameter names/values, device,
dtype, strides, physical tensor layout, checkpoint paths/bytes, private
resource values, source code либо storage topology. Topology не является raw
autograd trace.

У каждого target slot имеются отдельные nodes для
`lossInputTransformation` и `publicPredictionTransformation`, включая
`Identity`. Component node публикует unweighted `GlobalRowMean` в output port
`mean` и weighted contribution в port `weighted`. Поэтому telemetry direct и
auxiliary losses связываются с `mean`, а total loss — с `weighted`.

Для `PositiveClassWeightedBinaryCrossEntropyWithLogits` topology добавляет
node с `positiveClassWeight` и
`derivedCorrection: subtractLogPositiveClassWeight`. Он получает raw logit,
выдаёт `rawLogit - log(positiveClassWeight)` и является единственным source
публичной `Sigmoid` transformation. Это раскрывает правило интерпретации
output без публикации implementation details.

`gradientFlow` edge явно принимает одно из значений:

- `propagates` — backward path идёт от destination к source;
- `stopped` — forward value используется, но operator останавливает gradient;
- `notApplicable` — source не является trainable model value, например
  observed target или feature input.

Так `uncertaintyScale` для `RiskAdjustedExpectedValue` имеет `stopped`, а
`scale` для `GaussianNLL` — `propagates`.

## Идентичности, порядок и формы

`id` node и `id` port — provider-issued opaque keys, стабильные в пределах
`topologyRevision`. Client не разбирает их по разделителям. `label` — только
plain text для отображения; он не является ключом или источником семантики.
Target/component/resource identities передаются отдельными полями. Transformer
не интерпретирует их значения.

Nodes идут в deterministic execution order: inputs, encoder path, ordered
target slots, resources, direct components, auxiliary components и aggregate.
Edges идут в deterministic порядке их materialization. Direct components
следуют target slots; resources и auxiliary components сохраняют canonical
Semantic v5 order.

`logicalShape` edge описывает logical value. `batch` имеет `size: null`; все
остальные axes имеют положительный размер. Пустой массив означает scalar. Axis
names ограничены `sequence`, `feature`, `contextFeature` и `hidden`; physical
tensor representation из них не выводится.

## Limits и ошибки

Один result содержит не более 1024 nodes, 4096 edges и 8 MiB encoded JSON.
Превышение limit возвращает
`RESOURCE_EXHAUSTED / MODEL_TOPOLOGY_RESPONSE_BUDGET_EXCEEDED`; ответ не
усекается. Schema запрещает неизвестные fields и control characters в labels.
Если capability `queries.modelTopology` равна `false`, вызов возвращает
`FAILED_PRECONDITION / MODEL_TOPOLOGY_QUERY_UNAVAILABLE`.

Порядок outcome: owner-scoped registry lookup; validation stored metadata/D1;
materialization topology; проверка schema и response budget. Недоступность
registry возвращает `MODEL_REGISTRY_UNAVAILABLE`; повреждённая metadata —
`STORED_MODEL_METADATA_INVALID`; внутренне противоречивая topology —
`MODEL_TOPOLOGY_INVALID`.

## Fixtures

`fixtures/` содержит request, full graph с несколькими opaque targets, shared
resource и auxiliary components, а также structured error examples. Manifest
хеширует только этот небольшой offline bundle и не является runtime fence.
