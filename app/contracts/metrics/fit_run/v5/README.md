# Контракт terminal fit-run metrics v5

> CONTRACT DOCUMENT. Этот каталог задаёт будущую terminal fit summary и
> OpenSearch run document consumer-neutral model. Runtime v4 не изменяется.

Summary связывает published model с четырьмя D1 semantic layers, resolved
initialization и физическими/логическими compact-input counters. Ordered
targets сохраняются как `{index, identity}`; полного TargetContract или
Objective в telemetry artifact нет.

`opensearch/metrics-runs-v5.template.json` является future strict template с
нулём replicas. Установка template и создание/удаление indices остаются
отдельной operational change.

`targets` содержит полный ordered layout с непрерывными indices от нуля и
exact identities checkpoint-а. `counts.nativeRows` имеет по одному элементу
на `sourceEncoding.featureBlocks`; остальные physical/logical counters
проверяются против durable input manifest. Все timestamps и durations должны
образовывать фактически наблюдённый successful fit lifecycle, а числовые
значения обязаны быть finite.
