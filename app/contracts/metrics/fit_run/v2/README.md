# Контракт итоговой метрики fit run v2

`transformer.fit-run-summary.v2` — неизменяемый model-owned JSON artifact
успешного fit. Он формируется только после завершения worker-а и до terminal
транзакции публикации модели. Artifact содержит агрегированные контрольные
точки lifecycle, но не row- и batch-события.

PostgreSQL остаётся источником истины для job lifecycle, recovery и
checkpoint-aligned epoch metrics. OpenSearch получает только идемпотентную
post-commit проекцию `inventory.metrics.fit-run.v2` в обычный индекс
`metrics-runs-v2`.

Duration-поля могут перекрываться: потоковое обучение может идти до EOF.
Поэтому их нельзя суммировать для вычисления полного времени; критический путь
задаётся `remoteFitMs`.

`targetStatistics` содержит ровно шесть записей в порядке ML-контракта:
`meanReturn`, `sigmaReturn`, `probTP`, `probSL`, `volatilityNext`,
`hittingProbTP`. Для каждой координаты сохраняются count, min, max, mean,
population std, zeroCount и oneCount по Float32-значениям принятого immutable
manifest. `count` обязан совпадать с `counts.inputRows`.

OpenSearch document хранит те же значения как strict nested records. Поле
`modelRef` является top-level keyword, поэтому terminal summary одной model
generation однозначно находится точным запросом по `modelRef`. Недоступность
OpenSearch не влияет на model publication: источником истины остаётся
model-owned `run-summary.json` и связанная с ним PostgreSQL outbox.
