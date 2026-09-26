# Контракт Arrow Flight v22

> ДОКУМЕНТ КОНТРАКТА. Подготовленный closed package для controlled experiment
> порядка нормализации encoder. До отдельного review он не меняет активный
> Flight v21 или runtime.

Flight v22 образует clean cut с v21. Он принимает только Semantic v5
`ModelContract`; поле `modelTuning.encoderNormalizationOrder` обязательно и
имеет одно из двух значений: `postNorm` или `preNorm`. Отсутствующее либо
неизвестное значение отклоняется до создания job как
`INVALID_ARGUMENT / INVALID_MODEL_CONTRACT` через Semantic v5 error detail.

## Actions

```text
transformer.v22.capabilities
transformer.v22.health
transformer.v22.fit.create
transformer.v22.predict.create
transformer.v22.job.acquire
transformer.v22.job.status
transformer.v22.job.inputs.list
transformer.v22.job.input.close
transformer.v22.job.outputs.list
transformer.v22.job.cancel
transformer.model-catalog.v7.list
transformer.model-catalog.v7.detail
transformer.model-topology.v3.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v5.report
```

`transformer.v22.capabilities.semantic` возвращает closed Semantic v5
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
`modelDefinitionSha256`, checkpoint/recovery v12 metadata, Model Catalog v7
и Model Topology v3. `predict.create` принимает только `modelRef` и data
binding; он не получает order повторно. Warm start между разными orders
отклоняется обычной проверкой точного model definition.

Target Head Diagnostics v5 публикует `normalizationOrder` и неизменный набор
named boundaries (`input`, `norm1`, `attentionResidual`, `norm2`,
`feedForwardResidual`). Семантика Training Telemetry v4 не меняется.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction, target/objective
формулы, optimizer, AMP, gradient clipping, selection, public `predict` и
Training Telemetry v4 сохраняют прежнее значение. Flight v22 не добавляет
runtime override, миграцию, OpenSearch index или compatibility reader v21.

`fixtures/` содержит офлайн conformance bundle. Его manifest хеширует только
fixtures; он не является runtime fence, входом predict/fit или частью D1.
