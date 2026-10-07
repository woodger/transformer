# Terminal fit-run метрики Transformer v12

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v6.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v12 использует `transformer.metrics-fit-run.v12` в
`metrics-runs-v12`. `opensearch/metrics-runs-v12.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

v12 сохраняет смысл terminal summary v11, но фиксирует
`transformer-checkpoint-v13` и versioned initialization schema v13.

Документы fit-run v11 после перехода на v12 не читаются.
