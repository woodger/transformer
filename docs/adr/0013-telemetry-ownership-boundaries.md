# ADR 0013: структурные границы training telemetry

- Статус: принято
- Дата: 2026-08-18
- Уточняет: ADR 0008, ADR 0009, ADR 0011 и ADR 0012

## Контекст

После внедрения централизованных метрик telemetry формально была best effort,
но её структуры оставались частью нескольких прикладных владельцев:

- один `TrainMetrics` одновременно представлял результат обучения, состояние
  checkpoint selection и runtime-наблюдения;
- telemetry records и методы находились в service domain и общем PostgreSQL
  ledger;
- core artifact publisher одновременно публиковал модель, строил metrics
  artifacts и регистрировал OpenSearch outbox;
- административная модель lifecycle содержала статус доставки метрик.

Такое размещение позволяло случайно вернуть зависимость результата fit,
recovery или model lifecycle от вспомогательной наблюдаемости, хотя политика
метрик это запрещает.

## Решение

Вводятся три независимых вида состояния.

### Состояние обучения

`app/worker/training/epoch.py` содержит только `TrainingEpochResult`: число
строк и batch-ей, training loss, шесть direct loss components, auxiliary loss,
selection score, stage, learning rate и global step. Эти значения участвуют в
оптимизации, checkpoint selection и детерминированном recovery.

Core training state не импортирует telemetry. Текущий durable recovery format
`transformer-training-recovery-v3` сохраняется. Поле `best_metrics` остаётся в
нём только как инертная часть уже опубликованного формата и не восстанавливает
observability state.

### Execution progress

Recovery transaction обновляет в job только компактный прикладной progress:

```json
{
  "completedEpochs": 4,
  "globalStep": 2940
}
```

Полный epoch metrics document не является job state и не попадает в Flight
status. Он может отсутствовать без потери recovery generation.

### Telemetry

`app/worker/telemetry/` владеет необязательными runtime-наблюдениями:

- AMP, optimizer и gradient counters;
- per-target MAE/RMSE;
- missing-data ratios и phase timings;
- JSONL, console formatting и локальными plots.

`ObservedTrainingEpoch` расширяет core result наблюдениями только на runtime
границе trainer-а. Ошибка расчёта telemetry отключает наблюдения текущей epoch,
но не меняет loss, optimizer, selection или recovery.

`training/losses.py` вычисляет только objective и core loss components.
Per-target MAE/MSE вычисляются в telemetry slice. Trainer объединяет core
scalars, optional target observations и gradient norm в одну CUDA→CPU
передачу; при ошибке optional observations повторно материализуется только
core часть.

Service-side telemetry имеет собственные application records, ports и
publisher в `app/service/application/telemetry/` и
`app/service/application/ports/telemetry.py`. Реализации размещены у своих
outbound-владельцев:

```text
service/adapters/outbound/artifacts/telemetry/
service/adapters/outbound/postgres/telemetry/
service/adapters/outbound/opensearch/
```

Общий PostgreSQL ledger не предоставляет metrics API. Service domain не
содержит telemetry records. Core `WorkerArtifactPublisher` публикует checkpoint
и model generation, после чего возвращает минимальный immutable результат
необязательному `FitRunTelemetryPublisher`. Любая его ошибка наблюдаема, но не
изменяет уже зафиксированный `SUCCEEDED`.

Статус доставки не является полем `ModelLifecycleRecord` и не присоединяется к
административному представлению модели. Удаление модели не зависит от наличия
или состояния telemetry.

## Что не меняется

- публичный Flight v4;
- worker process остаётся v6; checkpoint и core result fields не меняются;
- публичные training metric names и fit-run lifecycle fields;
- локальный CLI-формат `metrics.jsonl`.

## Контроль

Architecture tests проверяют:

- отсутствие прежних `worker/metrics` и общих service metrics paths;
- наличие явных worker/service telemetry packages;
- отсутствие импорта telemetry из core epoch result, service domain, общего
  PostgreSQL ledger и core model publication;
- отсутствие циклов в import graph.

## Последствия

Telemetry становится отдельным наблюдателем прикладных границ, а не частью
самих границ. Потеря наблюдений может уменьшить диагностическую полноту, но не
повреждает ML-state и не меняет результат job. Новые telemetry-поля должны
добавляться в этот slice; добавление их в domain records, job progress или
core artifact publication требует нового архитектурного решения.
