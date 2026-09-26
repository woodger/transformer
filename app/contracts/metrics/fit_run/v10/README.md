# Terminal fit-run метрики Transformer v10

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v4.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v10 использует `transformer.metrics-fit-run.v10` в
`metrics-runs-v10`. `opensearch/metrics-runs-v10.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

v10 сохраняет смысл terminal summary v9, но фиксирует
`transformer-checkpoint-v11` и versioned initialization schema v11.

Документы fit-run v9 после перехода на v10 не читаются.
