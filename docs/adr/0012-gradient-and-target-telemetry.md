# ADR 0012: AMP/gradient telemetry и статистика training targets

## Статус

Принято, 2026-08-16.

## Контекст

Одна non-finite gradient norm могла превратить среднее всей epoch в `null` и
скрыть корректные значения остальных batch-ей. Поле `step` при этом отражало
завершённые training batches, но не позволяло отличить применённый optimizer
update от AMP overflow и пропуска GradScaler.

Inventory также нужен training-mean baseline по тому же immutable dataset, на
котором обучалась модель. Восстанавливать target statistics из test dataset
или из повторного Flight upload нельзя: это создаёт другой источник истины.

## Решение

Внутренний worker process contract повышается до v6. Каждая завершённая epoch
содержит:

- `trainingBatchesCompleted`;
- `optimizerUpdatesApplied` и `optimizerUpdatesSkipped`;
- `ampOverflowBatches`;
- `finiteGradientBatches` и `nonFiniteGradientBatches`;
- `preClipGradientNormMean`, `preClipGradientNormMax` и
  `preClipGradientNormP95`.

Применение optimizer update определяется фактическим вызовом
`Optimizer.step`, а не изменением AMP scale. Это не добавляет CUDA→CPU
синхронизацию. `globalTrainingStep` и поле `step` сохраняют прежнюю семантику
числа завершённых training batches.

Обязательные инварианты:

```text
trainingBatchesCompleted
  = optimizerUpdatesApplied + optimizerUpdatesSkipped
  = finiteGradientBatches + nonFiniteGradientBatches
```

Mean, max и nearest-rank P95 считаются только по конечным неотрицательным
pre-clip norms. Если конечных значений нет, все три агрегата равны `null`.
Non-finite batch учитывается отдельным counter и не меняет агрегаты конечных
batch-ей.

Fit worker один раз на ordinal читает Float32 targets принятого immutable
manifest и считает для каждой из шести координат count, min, max, mean,
population standard deviation, zeroCount и oneCount. Новая attempt после
recovery пересчитывает значения из тех же durable artifacts; только terminal
result текущей attempt попадает в model publication, поэтому повторного
durable учёта нет.

Target statistics становятся обязательной частью model-owned
`run-summary.json`. Service повторно проверяет identity, диапазоны, finite
values и равенство `count == inputRows` до terminal transaction.

## Версирование и durable boundary

Строгие форматы повышаются атомарно:

- `transformer.training-metrics.v2`;
- `inventory.metrics.point.v2` и `inventory.metrics.artifact.v2`;
- `transformer.fit-run-summary.v2`;
- `inventory.metrics.fit-run.v2`;
- outbox projection `inventory.metrics.v3`.

Новые документы доставляются в обычные индексы `metrics-points-v2`,
`metrics-artifacts-v2` и `metrics-runs-v2`. Readers v1 остаются только для
доставки уже зафиксированных outbox entries. Новые fit runs всегда используют
v2 artifacts и projection v3.

Epoch metric и recovery checkpoint по-прежнему фиксируются одной транзакцией
PostgreSQL. Target statistics входят в terminal summary; summary metadata,
model generation, outbox и `SUCCEEDED` фиксируются существующей terminal
транзакцией `publish_model`. OpenSearch остаётся post-commit проекцией и его
недоступность не влияет на fit.

Публичный Flight v4 и PostgreSQL schema не меняются. Перед развёртыванием
worker v6 активные fit jobs следует завершить или отменить: checkpoint-aligned
строки v1 и v2 нельзя смешивать в одном immutable epoch artifact.

## Последствия

По `modelRef` в `metrics-runs-v2` однозначно находится один successful run
summary с training target statistics. Inventory может вычислить training-mean
baseline без повторного чтения dataset. Failed/cancelled attempt telemetry,
row/batch events и отправка из ML worker остаются вне решения.
