# Контракт Arrow Flight v23

> ДОКУМЕНТ КОНТРАКТА. Closed package Semantic v6 с
> декларативным Bernoulli confidence penalty.

Flight v23 образует clean cut с v22 и принимает только Semantic v6
`ModelContract`. Он разрешает optional auxiliary `BernoulliConfidencePenalty`
над публичной вероятностью существующего target. Без декларации компонента
формулы objective остаются прежними. Обязательный
`modelTuning.encoderNormalizationOrder: "postNorm" | "preNorm"` сохраняется.
Неизвестная или неверная declaration отклоняется до создания job через
Semantic v6 structured error detail.

## Actions

```text
transformer.v23.capabilities
transformer.v23.health
transformer.v23.fit.create
transformer.v23.predict.create
transformer.v23.job.acquire
transformer.v23.job.status
transformer.v23.job.inputs.list
transformer.v23.job.input.close
transformer.v23.job.outputs.list
transformer.v23.job.cancel
transformer.model-catalog.v8.list
transformer.model-catalog.v8.detail
transformer.model-topology.v4.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v5.report
```

`transformer.v23.capabilities.semantic` возвращает closed Semantic v6
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
`modelDefinitionSha256`, checkpoint/recovery v13 metadata, Model Catalog v8
и Model Topology v4. `predict.create` принимает только `modelRef` и data
binding; он не получает order повторно. Warm start между разными orders
отклоняется обычной проверкой точного model definition.

Target Head Diagnostics v5 публикует `normalizationOrder` и неизменный набор
named boundaries (`input`, `norm1`, `attentionResidual`, `norm2`,
`feedForwardResidual`). Семантика Training Telemetry v4 не меняется.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction, существующие
формулы operators, optimizer, AMP, gradient clipping, selection, public `predict` и
Training Telemetry v4 сохраняют прежнее значение. Confidence penalty
применяется только по declaration objective. Revision 0030
выполняет destructive clean cut для новых D1 и checkpoint/recovery v13;
compatibility reader Flight v22 отсутствует. Metrics v12 связывает observations
с checkpoint v13; deployment заменяет индексы metrics v11 на v12.
Capabilities Semantic v6 публикует `auxiliaryOperators`.

`fixtures/` содержит офлайн conformance bundle. Его manifest хеширует только
fixtures; он не является runtime fence, входом predict/fit или частью D1.
