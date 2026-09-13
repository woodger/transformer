# Контракт training metrics v6

> CONTRACT DOCUMENT. Этот каталог задаёт metrics carrier contract для Semantic
> v2.

`transformer.training-metrics.v6` и `transformer.metrics-point.v6` сохраняют
прежние numerical observations, ordered target/component references и health
counters. Версия меняется только для exact Semantic v2 D1 и
`transformer-checkpoint-v7` format fence.

`opensearch/metrics-points-v6.template.json` задаёт strict template v6.
Его применение и lifecycle индекса относятся к deployment procedure.

Fixture manifest проверяет только целостность conformance fixtures и не является
runtime compatibility identity.
