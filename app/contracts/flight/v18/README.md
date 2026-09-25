# Контракт Arrow Flight v18

> ДОКУМЕНТ КОНТРАКТА. Этот каталог задаёт действующую публичную границу
> Arrow Flight Transformer. Flight v17 не обслуживается runtime.

Flight v18 сохраняет физическую плоскость данных и workflow Flight v17, но
добавляет отдельную read-only поверхность диагностики выходных головок целей.
Новый action не расширяет v17: v18 является единственным объявляемым action
surface без aliases v17.

Отсутствие wire aliases не удаляет уже опубликованные Semantic v4 поколения.
Для retained checkpoint metadata v9 provider выполняет только явно заданную
проекцию diagnostics v1 в diagnostics v2 с `targetHead: null`; это позволяет
Model Catalog v5 и Target Head Diagnostics v1 вернуть `notConfigured` без
изменения старого checkpoint-а. Такая проекция не является reader-ом Flight
v17 и не создаёт исторические observations.

## Actions

```text
transformer.v18.capabilities
transformer.v18.health
transformer.v18.fit.create
transformer.v18.predict.create
transformer.v18.job.acquire
transformer.v18.job.status
transformer.v18.job.inputs.list
transformer.v18.job.input.close
transformer.v18.job.outputs.list
transformer.v18.job.cancel
transformer.model-catalog.v5.list
transformer.model-catalog.v5.detail
transformer.model-topology.v2.detail
transformer.training-telemetry.v4.report
transformer.training-telemetry.v4.gradient-interactions
transformer.target-head-diagnostics.v1.report
```

`fit.create` и resolved `jobConfigSha256` получают расширенную закрытую
runtime-конфигурацию `diagnostics`:

```json
{
  "gradientInteractions": null,
  "targetHead": "fullCommittedArtifact"
}
```

`targetHead: null` не собирает новую диагностику. Значение
`fullCommittedArtifact` включает наблюдательный post-update проход по полному
закрытому входному артефакту. Эта настройка не меняет `ModelContract`, Semantic
v4, D1 layers, `modelDefinitionSha256`, public `predict`, objective или
совместимость published-model initialization.

Допустимый размер полного artifact объявляется в
`queries.targetHeadDiagnostics.maxCommittedArtifactRows`. Если закрытый input
превышает этот limit, fit не меняет свой terminal outcome, но complete
diagnostics artifact не создаётся; lazy query возвращает
`unavailable / NO_COMPLETE_REPORT`.

## Возможности и связанные packages

`transformer.v18.capabilities` содержит обычные limits Flight и в
`queries.targetHeadDiagnostics` полный документ capabilities из
[`target_head_diagnostics/v1`](../../target_head_diagnostics/v1/README.md).
Когда поверхность выключена deployment-ом, поле равно `false`. Вызывающая
система проверяет этот block перед отображением или вызовом
диагностики, а не выводит доступность из версии модели или названия цели.

`action-result.schema.json` принимает результат нового action, а
`error-detail.schema.json` принимает его structured errors. `queries` также
сохраняет Model Catalog v5, Model Topology v2 и Training Telemetry v4;
семантика последних двух surfaces не меняется.

Общий `action-result.schema.json` использует `anyOf`, а не `oneOf`: состояния
`pending` и `unavailable` нескольких read-only actions намеренно имеют одну
форму. Конкретный action определяет авторитетную request/result schema; общий
документ проверяет, что ответ принадлежит хотя бы одной объявленной форме.

Новая диагностика связана с Worker v16, checkpoint/recovery v10 и Model
Catalog v5. Semantic v4, Metrics v9 и Topology v2 остаются exact прежними
пакетами.

## Сохранённые свойства

`indexedFeatureBlocks`, Arrow schemas, logical reconstruction,
`PositiveClassWeightedBinaryCrossEntropyWithLogits`, derived public correction
и output `predict` не изменяются. Target-head diagnostics не читается из
публичного checkpoint-а во время запроса и не раскрывает artifact paths,
параметры или строки входных данных.

`fixtures/` содержит прежние физические fixtures v17 и обновлённые JSON
fixtures Flight v18, включая capabilities и exact `jobConfigSha256` для
включённой диагностики. Его manifest предназначен только для офлайн-проверки
между проектами и не является runtime fence.
