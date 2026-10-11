# Метрики обучения Transformer v13

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v7.

`transformer.training-metrics.v13` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v13 использует `transformer.metrics-point.v13` в
`metrics-points-v13`.

v13 сохраняет состав и семантику observations v12, но связывает record с
`transformer-checkpoint-v14`; поэтому format документа и OpenSearch index
version меняются вместе.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v13.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Индексы Metrics
v12 несовместимы с v13 и удаляются release procedure вместо их чтения runtime.
