# Terminal fit-run метрики Transformer v13

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v7.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v13 использует `transformer.metrics-fit-run.v13` в
`metrics-runs-v13`. `opensearch/metrics-runs-v13.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

v13 сохраняет смысл terminal summary v12, но фиксирует
`transformer-checkpoint-v14` и versioned initialization schema v14.

Документы fit-run v12 после перехода на v13 не читаются.
