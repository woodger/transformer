# Контракт итоговой метрики fit run v2

`transformer.fit-run-summary.v2` — необязательный неизменяемый run-owned JSON
artifact успешного fit. Он формируется best effort после terminal transaction
публикации модели. Artifact содержит агрегированные контрольные точки lifecycle,
но не row- и batch-события.

PostgreSQL остаётся источником истины для job lifecycle и recovery. Epoch
metrics в PostgreSQL, run-owned summary и OpenSearch documents являются
best-effort observability-данными. OpenSearch получает идемпотентную post-commit
проекцию `inventory.metrics.fit-run.v2` в обычный индекс `metrics-runs-v2`.

Duration-поля могут перекрываться: потоковое обучение может идти до EOF.
Поэтому их нельзя суммировать для вычисления полного времени; критический путь
задаётся `remoteFitMs`.

Поле `modelRef` является корреляционным top-level keyword, поэтому terminal
summary опубликованного run находится точным запросом по модели. Статистики
исходных training targets не входят в Transformer telemetry: этими данными и
соответствующими baseline владеет Consumer. Недоступность
OpenSearch и отсутствие самого summary не влияют на model publication.
`run-summary.json` и связанная с ним PostgreSQL outbox являются только
best-effort observability projection.
