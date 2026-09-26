# Контракт Arrow Flight v21

> ДОКУМЕНТ КОНТРАКТА. Текущая публичная граница Arrow Flight Transformer.

Flight v21 сохраняет физическую плоскость данных и порядок работы предыдущей
границы, но использует Target Head Diagnostics v4 и Model Catalog v6. Action
surface закрыта и не объявляет псевдонимы предыдущих версий Flight.

## Actions

```text
transformer.v21.capabilities
transformer.v21.health
transformer.v21.fit.create
transformer.v21.predict.create
transformer.v21.job.acquire
transformer.v21.job.status
transformer.v21.job.inputs.list
transformer.v21.job.input.close
transformer.v21.job.outputs.list
transformer.v21.job.cancel
transformer.model-catalog.v6.list
transformer.model-catalog.v6.detail
transformer.model-topology.v2.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v4.report
```

`fit.create` и resolved `jobConfigSha256` получают закрытую runtime-
конфигурацию `diagnostics`:

```json
{
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact",
  "encoderLayerDiagnostics": "directComponentPerBatch"
}
```

`encoderLayerDiagnostics` допустим только с `targetHead:
"fullCommittedArtifact"`. Он собирает pre-clip gradients direct component и
post-step parameter updates по группам каждого encoder layer. Настройка входит
в `jobConfigSha256` и recovery fence, но не меняет `ModelContract`, Semantic
v4, D1 layers, `modelDefinitionSha256`, public `predict`, objective или
совместимость published-model initialization.

Flight request намеренно не содержит `schemaVersion`: сервис сохраняет эту
transport-форму как checkpoint/recovery v11 document с `schemaVersion: 3`.

Неверная diagnostics configuration отклоняется до создания job с
`INVALID_ARGUMENT / INVALID_DIAGNOSTICS_CONFIGURATION`; `path` указывает на
некорректное поле. Ошибки чтения уже опубликованной diagnostics projection
остаются в `target_head_diagnostics/v4/error-detail.schema.json`.

Допустимый размер полного artifact объявляется в
`queries.targetHeadDiagnostics.maxCommittedArtifactRows`. Если закрытый input
превышает этот limit, fit не меняет свой terminal outcome, но complete
diagnostics artifact не создаётся; lazy query возвращает
`unavailable / NO_COMPLETE_REPORT`.

## Возможности и связанные packages

`transformer.v21.capabilities` содержит обычные limits Flight и в
`queries.targetHeadDiagnostics` полный документ capabilities из
[`target_head_diagnostics/v4`](../../target_head_diagnostics/v4/README.md).
Когда поверхность выключена deployment-ом, поле равно `false`. Вызывающая
система проверяет этот block перед отображением или вызовом
диагностики, а не выводит доступность из версии модели или названия цели.

`action-result.schema.json` принимает результат diagnostics v4, а
`error-detail.schema.json` принимает его structured errors. `queries` также
содержит Model Catalog v6, Model Topology v2 и Training Telemetry v4;
семантика последних двух surfaces не меняется.

Общий `action-result.schema.json` использует `anyOf`, а не `oneOf`: состояния
`pending` и `unavailable` нескольких read-only actions намеренно имеют одну
форму. Конкретный action определяет авторитетную request/result schema; общий
документ проверяет, что ответ принадлежит хотя бы одной объявленной форме.

Новая диагностика связана с Worker v19, checkpoint/recovery v11 и Model
Catalog v6. Semantic v4, Topology v2 и Metrics v10 остаются прежними
пакетами; замена OpenSearch indices не требуется.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction,
`PositiveClassWeightedBinaryCrossEntropyWithLogits`, derived public correction
и output `predict` не изменяются. Target-head diagnostics не читается из
публичного checkpoint-а во время запроса и не раскрывает artifact paths,
параметры или строки входных данных.

`fixtures/` содержит прежние физические fixtures и обновлённые JSON
fixtures Flight v21, включая capabilities и exact `jobConfigSha256` для
включённой диагностики. Его manifest предназначен только для офлайн-проверки
между проектами и не является runtime fence.
