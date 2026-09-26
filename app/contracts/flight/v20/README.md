# Контракт Arrow Flight v20

> ДОКУМЕНТ КОНТРАКТА. Текущая публичная граница Arrow Flight Transformer.

Flight v20 сохраняет физическую плоскость данных и порядок работы предыдущей
границы, но использует Target Head Diagnostics v3 и Model Catalog v6. Action
surface закрыта и не объявляет псевдонимы предыдущих версий Flight.

## Actions

```text
transformer.v20.capabilities
transformer.v20.health
transformer.v20.fit.create
transformer.v20.predict.create
transformer.v20.job.acquire
transformer.v20.job.status
transformer.v20.job.inputs.list
transformer.v20.job.input.close
transformer.v20.job.outputs.list
transformer.v20.job.cancel
transformer.model-catalog.v6.list
transformer.model-catalog.v6.detail
transformer.model-topology.v2.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v3.report
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
остаются в `target_head_diagnostics/v3/error-detail.schema.json`.

Допустимый размер полного artifact объявляется в
`queries.targetHeadDiagnostics.maxCommittedArtifactRows`. Если закрытый input
превышает этот limit, fit не меняет свой terminal outcome, но complete
diagnostics artifact не создаётся; lazy query возвращает
`unavailable / NO_COMPLETE_REPORT`.

## Возможности и связанные packages

`transformer.v20.capabilities` содержит обычные limits Flight и в
`queries.targetHeadDiagnostics` полный документ capabilities из
[`target_head_diagnostics/v3`](../../target_head_diagnostics/v3/README.md).
Когда поверхность выключена deployment-ом, поле равно `false`. Вызывающая
система проверяет этот block перед отображением или вызовом
диагностики, а не выводит доступность из версии модели или названия цели.

`action-result.schema.json` принимает результат diagnostics v3, а
`error-detail.schema.json` принимает его structured errors. `queries` также
содержит Model Catalog v6, Model Topology v2 и Training Telemetry v4;
семантика последних двух surfaces не меняется.

Общий `action-result.schema.json` использует `anyOf`, а не `oneOf`: состояния
`pending` и `unavailable` нескольких read-only actions намеренно имеют одну
форму. Конкретный action определяет авторитетную request/result schema; общий
документ проверяет, что ответ принадлежит хотя бы одной объявленной форме.

Новая диагностика связана с Worker v18, checkpoint/recovery v11 и Model
Catalog v6. Semantic v4 и Topology v2 остаются прежними пакетами; Metrics v10
переводит telemetry storage на checkpoint v11.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction,
`PositiveClassWeightedBinaryCrossEntropyWithLogits`, derived public correction
и output `predict` не изменяются. Target-head diagnostics не читается из
публичного checkpoint-а во время запроса и не раскрывает artifact paths,
параметры или строки входных данных.

`fixtures/` содержит прежние физические fixtures и обновлённые JSON
fixtures Flight v20, включая capabilities и exact `jobConfigSha256` для
включённой диагностики. Его manifest предназначен только для офлайн-проверки
между проектами и не является runtime fence.
