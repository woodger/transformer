# Метрики обучения Transformer v9

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v4.

`transformer.training-metrics.v9` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v9 использует `transformer.metrics-point.v9` в
`metrics-points-v9`.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v9.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Прежние индексы
metrics несовместимы с чистым переходом v18 и удаляются release procedure,
вместо чтения их runtime v9.
