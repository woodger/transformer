# Метрики обучения Transformer v7

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v3.

`transformer.training-metrics.v7` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v7 использует `transformer.metrics-point.v7` в
`metrics-points-v7`.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v3; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v7.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Прежние индексы
metrics несовместимы с чистым переходом v15 и удаляются release procedure,
вместо чтения их runtime v7.
