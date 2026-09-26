# Terminal fit-run метрики Transformer v8

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v4.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v8 использует `transformer.metrics-fit-run.v8` в
`metrics-runs-v8`. `opensearch/metrics-runs-v8.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

Документы прежних fit-run после чистого перехода Flight v17 не читаются.
