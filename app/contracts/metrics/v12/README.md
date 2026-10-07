# Метрики обучения Transformer v12

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v6.

`transformer.training-metrics.v12` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v12 использует `transformer.metrics-point.v12` в
`metrics-points-v12`.

v12 сохраняет состав и семантику observations v11, но связывает record с
`transformer-checkpoint-v13`; поэтому format документа и OpenSearch index
version меняются вместе.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v12.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Индексы Metrics
v11 несовместимы с v12 и удаляются release procedure вместо их чтения runtime.
