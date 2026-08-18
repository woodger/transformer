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

Target statistics входят в model-owned `run-summary.json`, если worker смог их
собрать. Service повторно проверяет identity, диапазоны, finite values и
равенство `count == inputRows`; невалидная telemetry отбрасывается после
прикладной terminal transaction и не меняет fit outcome.

## Версирование и durable boundary

Строгие форматы повышаются атомарно:

- `transformer.training-metrics.v2`;
- `inventory.metrics.point.v2` и `inventory.metrics.artifact.v2`;
- `transformer.fit-run-summary.v2`;
- `inventory.metrics.fit-run.v2`;
- outbox projection `inventory.metrics.v3`.

Новые документы доставляются в обычные индексы `metrics-points-v2`,
`metrics-artifacts-v2` и `metrics-runs-v2`. Экспериментальные артефакты v1,
записи outbox и индексы удаляются при атомарном переключении; слой
совместимости для них не сохраняется. Все запуски fit используют артефакты v2
и проекцию v3.

Recovery checkpoint фиксируется отдельной прикладной транзакцией PostgreSQL;
epoch metric сохраняется best effort после неё. Model generation и `SUCCEEDED`
фиксируются до попытки создать target statistics summary, artifact metadata и
outbox. OpenSearch остаётся post-commit проекцией и его недоступность не влияет
на fit.

Публичный Flight v4 и PostgreSQL schema не меняются. Перед развёртыванием
worker v6 активные fit jobs следует завершить или отменить, а созданные
экспериментальной версией модели и телеметрию удалить.

## Последствия

При успешной публикации telemetry по `modelRef` в `metrics-runs-v2` однозначно
находится один run summary с training target statistics. При её отсутствии
baseline недоступен; модель и fit остаются корректными. Failed/cancelled
attempt telemetry, row/batch events и отправка из ML worker остаются вне
решения.
