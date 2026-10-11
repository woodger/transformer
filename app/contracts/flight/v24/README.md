# Контракт Arrow Flight v24

> ДОКУМЕНТ КОНТРАКТА. Closed package Semantic v7 с
> декларативными Bernoulli confidence и entropy penalties.

Flight v24 образует clean cut с v23 и принимает только Semantic v7
`ModelContract`. Он разрешает optional auxiliary `BernoulliConfidencePenalty`
и `BernoulliEntropyPenalty`
над публичной вероятностью существующего target. Без декларации компонента
формулы objective остаются прежними. Обязательный
`modelTuning.encoderNormalizationOrder: "postNorm" | "preNorm"` сохраняется.
Неизвестная или неверная declaration отклоняется до создания job через
Semantic v7 structured error detail.

## Actions

```text
transformer.v24.capabilities
transformer.v24.health
transformer.v24.fit.create
transformer.v24.predict.create
transformer.v24.job.acquire
transformer.v24.job.status
transformer.v24.job.inputs.list
transformer.v24.job.input.close
transformer.v24.job.outputs.list
transformer.v24.job.cancel
transformer.model-catalog.v9.list
transformer.model-catalog.v9.detail
transformer.model-topology.v5.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v5.report
```

`transformer.v24.capabilities.semantic` возвращает closed Semantic v7
capabilities, включая ordered `encoderNormalizationOrders`:

```json
["postNorm", "preNorm"]
```

`queries.targetHeadDiagnostics` возвращает Target Head Diagnostics v5
capabilities с тем же набором orders. Вызов не выводит order из глубины,
названия цели или deployment setting.

## Семантика experiment

`postNorm` materializes PyTorch `norm_first=false` и исполняет блок как
`attention → residual → norm1 → feed-forward → residual → norm2`.
`preNorm` materializes `norm_first=true` и исполняет
`norm1 → attention → residual → norm2 → feed-forward → residual`.

Значение принадлежит immutable resolved model definition. Оно входит в
`modelDefinitionSha256`, checkpoint/recovery v14 metadata, Model Catalog v9
и Model Topology v5. `predict.create` принимает только `modelRef` и data
binding; он не получает order повторно. Warm start между разными orders
отклоняется обычной проверкой точного model definition.

Target Head Diagnostics v5 публикует `normalizationOrder` и неизменный набор
named boundaries (`input`, `norm1`, `attentionResidual`, `norm2`,
`feedForwardResidual`). Семантика Training Telemetry v4 не меняется.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction, существующие
формулы operators, optimizer, AMP, gradient clipping, selection, public `predict` и
Training Telemetry v4 сохраняют прежнее значение. Bernoulli penalties
применяются только по declaration objective. Revision 0031
выполняет destructive clean cut для новых D1 и checkpoint/recovery v14;
compatibility reader Flight v23 отсутствует. Metrics v13 связывает observations
с checkpoint v14; deployment заменяет индексы metrics v12 на v13.
Capabilities Semantic v7 публикует `auxiliaryOperators`.

`fixtures/` содержит офлайн conformance bundle. Его manifest хеширует только
fixtures; он не является runtime fence, входом predict/fit или частью D1.
