# Terminal fit-run метрики Transformer v11

> ДОКУМЕНТ КОНТРАКТА. Этот пакет определяет внутреннее для provider-а terminal
> fit summary и его проекцию OpenSearch для запусков Semantic v5.

Immutable summary связывает producing job и опубликованную model с её
identities D1, разрешённой initialization, milestones обучения,
продолжительностями, input counts и physical artifact fences. Это completion
marker, используемый materializer telemetry; он не является публичной записью
registry моделей.

Projection v11 использует `transformer.metrics-fit-run.v11` в
`metrics-runs-v11`. `opensearch/metrics-runs-v11.template.json` должен быть
установлен до создания индекса. Вызывающая система получает эти observations
только через Training Telemetry Query v4, но никогда прямым доступом к
OpenSearch.

v11 сохраняет смысл terminal summary v10, но фиксирует
`transformer-checkpoint-v12` и versioned initialization schema v12.

Документы fit-run v10 после перехода на v11 не читаются.
