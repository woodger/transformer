# Метрики обучения Transformer v11

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутренний для provider-а artifact
> epoch metrics и его point projection OpenSearch для запусков Semantic v5.

`transformer.training-metrics.v11` записывает immutable epoch observations с
упорядоченными непрозрачными references target/component, значениями loss,
MAE/RMSE target-ов, gradient diagnostics и health counters optimizer-а. Point
projection v11 использует `transformer.metrics-point.v11` в
`metrics-points-v11`.

v11 сохраняет состав и семантику observations v10, но связывает record с
`transformer-checkpoint-v12`; поэтому format документа и OpenSearch index
version меняются вместе.

Эти документы — storage telemetry provider-а, а не публичный query contract.
Вызывающая система получает нормализованные reports только через Training
Telemetry Query v4; имена индексов, IDs документов и mappings OpenSearch ей не
передаются.

`opensearch/metrics-points-v11.template.json` — обязательный строгий index
template. Deployment должен установить его до создания индекса. Индексы Metrics
v10 несовместимы с v11 и удаляются release procedure вместо их чтения runtime.
