# Контракт terminal fit-run metrics v6

> CONTRACT DOCUMENT. Этот каталог задаёт staged terminal fit summary для
> Semantic v2. Действующий fit-run v5 и OpenSearch projection не меняются.

Summary связывает published model с четырьмя D1 layers, checkpoint-owned
`ResolvedInitialization` v7 и прежними compact-input counters. Initialization
использует предметное поле `source`; requested и catalog-summary формы здесь
недопустимы. Ordered targets и numerical lifecycle semantics не меняются.

`opensearch/metrics-runs-v6.template.json` является будущим strict template.
Его установка и создание/удаление indices не входят в staged package.
