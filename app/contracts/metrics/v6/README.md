# Контракт training metrics v6

> CONTRACT DOCUMENT. Этот каталог задаёт staged metrics carrier contract для
> Semantic v2. Действующий metrics v5 и его runtime projection не меняются.

`transformer.training-metrics.v6` и `transformer.metrics-point.v6` сохраняют
прежние numerical observations, ordered target/component references и health
counters. Версия меняется только для exact Semantic v2 D1 и
`transformer-checkpoint-v7` format fence.

`opensearch/metrics-points-v6.template.json` является будущим strict template.
Его установка и создание индекса не входят в staged package.

Fixture manifest проверяет только целостность conformance fixtures и не является
runtime compatibility identity.
