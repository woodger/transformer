# Контракт итоговой метрики fit run v1

`transformer.fit-run-summary.v1` — неизменяемый model-owned JSON artifact
успешного fit. Он формируется только после завершения worker-а и до terminal
транзакции публикации модели. Artifact содержит агрегированные контрольные
точки lifecycle, но не row- и batch-события.

PostgreSQL остаётся источником истины для job lifecycle, recovery и
checkpoint-aligned epoch metrics. OpenSearch получает только идемпотентную
post-commit проекцию `inventory.metrics.fit-run.v1` в обычный индекс
`metrics-runs-v1`.

Duration-поля могут перекрываться: потоковое обучение может идти до EOF.
Поэтому их нельзя суммировать для вычисления полного времени; критический путь
задаётся `remoteFitMs`.
