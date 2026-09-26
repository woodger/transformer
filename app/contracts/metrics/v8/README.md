# Метрики обучения Transformer v8

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v4.

`transformer.training-metrics.v8` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v8 использует `transformer.metrics-point.v8` в
`metrics-points-v8`.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v8.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Прежние индексы
metrics несовместимы с чистым переходом v17 и удаляются release procedure,
вместо чтения их runtime v8.
