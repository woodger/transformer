# Метрики обучения Transformer v10

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v4.

`transformer.training-metrics.v10` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v10 использует `transformer.metrics-point.v10` в
`metrics-points-v10`.

v10 сохраняет состав и семантику observations v9, но связывает record с
`transformer-checkpoint-v11`; поэтому format документа и OpenSearch index
version меняются вместе.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v10.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Индексы Metrics
v9 несовместимы с v10 и удаляются release procedure вместо их чтения runtime.
