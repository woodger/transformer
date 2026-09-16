# Terminal fit-run метрики Transformer v7

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v3.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v7 использует `transformer.metrics-fit-run.v7` в
`metrics-runs-v7`. `opensearch/metrics-runs-v7.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v3, но никогда прямым доступом к
OpenSearch.

Документы прежних fit-run после чистого перехода Flight v15 не читаются.
