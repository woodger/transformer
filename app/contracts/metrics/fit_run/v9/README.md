# Terminal fit-run метрики Transformer v9

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v4.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v9 использует `transformer.metrics-fit-run.v9` в
`metrics-runs-v9`. `opensearch/metrics-runs-v9.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

Документы прежних fit-run после чистого перехода Flight v18 не читаются.
