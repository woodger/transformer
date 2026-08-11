# Training runtime и checkpoint

> Type: Reference. Checkpoint format, обучение, missing-data semantics и
> training metrics local CLI.

Параметры команд находятся в [справочнике CLI](./cli/index.md), точные формулы
loss — в [документе о функции потерь](./losses.md), а input/output schema — в
[локальном Arrow и stream contract](./local-arrow-protocol.md).

## Checkpoint contract

Новый формат `transformer-checkpoint-v2` содержит `state_dict`, версию
приложения, model config, train config, `data_schema` и metadata выбора best
checkpoint. Model config включает `seq_len`, `hidden`, `layers`, `dropout`,
`nhead`, `context_mode`, `out_dim` и фактический `feature_dim` training input.
`data_schema` фиксирует имена и ширину `src`/`tgt`, допустимые element types,
`float32` tensor dtype, исходный и подготовленный model input dimension,
отсутствие normalization и missing policy. При prediction входной `feature_dim`
должен совпасть с сохранённым.

`fit` и `fit-stream` всегда создают новую модель; продолжение обучения из
checkpoint не реализовано. `--checkpoint-out` задаёт только конечную цель:
существующий checkpoint остаётся нетронутым во время обучения и атомарно
заменяется лишь при успешном save.

Loader принимает v2, предыдущий wrapped v1 и raw legacy `state_dict`;
неизвестный wrapped format отклоняется. Для v2 явно заданные model options
служат проверкой совпадения с checkpoint. Для legacy checkpoint без model
config `--seq-len` обязателен, а остальные не-default architecture options
задаются вручную.

## Monitor, early stopping и loss schedule

Критерий early stopping и выбора checkpoint задаётся `--monitor`; требуемая для
MAE monitors доля улучшения относительно zero-return baseline —
`--monitor-min-improvement`. Их defaults определены в `app/config.py`:

```python
TRAIN_MONITOR = "ret_mae_skill"          # loss | ret_mae | ret_mae_skill
TRAIN_MONITOR_MIN_IMPROVEMENT = 0.0      # 0.01 означает лучше baseline на 1%
SAVE_BEST_CHECKPOINT = True
```

Для `ret_mae_skill` формула такая:

```text
ret_mae_skill = ret_mae / ret_mae_baseline
```

Значение `< 1.0` означает, что модель лучше нулевого прогноза `meanR=0`.
Monitor вычисляется по тому же training pass, на котором обновлялись веса;
отдельного validation split этот CLI не создаёт. Early stopping начинает
останавливать только на максимальном настроенном loss stage, сбрасывает свой
счётчик при смене stage и отключён при `--patience=0`. Best checkpoint
обновляется только если baseline пройден, monitor конечен и улучшился; если
baseline ни разу не пройден, сохраняются текущие веса последней эпохи.

Loss stage соответствует следующим компонентам:

| Stage | Компоненты |
| --- | --- |
| `1` | Gaussian NLL для return |
| `2` | stage 1 + BCE для TP/SL logits |
| `3` | stage 2 + Bayesian EV/risk |
| `4` | stage 3 + log-volatility loss |

Stage 1 NLL может быть отрицательным — это допустимое значение Gaussian NLL,
а не признак сломанного обучения. Точные формулы находятся в
[`docs/losses.md`](./losses.md).

Loss schedule можно зафиксировать вручную:

```bash
./.venv/bin/python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-schedule=none \
  --loss-stage=1
```

Или включить автоматический curriculum по эпохам:

```bash
./.venv/bin/python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-stage=4 \
  --loss-schedule=epoch \
  --stage-size=4
```

Для schedule по optimizer steps:

```bash
./.venv/bin/python ./app/main.py fit-stream \
  --seq-len=20 \
  --loss-stage=4 \
  --loss-schedule=step \
  --stage-size=100
```

При step schedule счётчик проверяется перед каждым optimizer step, поэтому
активный stage может смениться посреди epoch.

## File, standalone stream и Flight fit

В standalone `fit-stream`, читающем stdin, каждый непустой Arrow frame
обучается отдельным циклом `epoch=1..--epochs` до срабатывания `--patience`.
Веса модели при этом не сбрасываются между frames; optimizer step также
остаётся глобальным, а per-frame early stopping начинается заново. Пустые Arrow
tables пропускаются; если непустых frames не было, команда завершается ошибкой
и модель не сохраняется. После terminator или clean EOF с хотя бы одним
непустым frame команда атомарно сохраняет checkpoint.

Flight fit использует durable input stream: epoch 0 начинает обработку
непрерывного префикса payloads до EOF, а последующие job-wide эпохи перечитывают
закрытый immutable dataset по ordinal, с едиными loss
schedule, optimizer, checkpoint selection и early stopping на весь job. Строки
проходят через ограниченное job-wide окно перемешивания; его границы и optimizer
batches могут пересекать payload и не зависят от транспортного разбиения.

## Контекстные пропуски

`src` может содержать `NaN` в отдельных фичах контекстного timestep. Режим
`--mode` задаёт, как такие timesteps попадают в Transformer:

- `strict` — timestep маскируется, если хотя бы одна фича `NaN`; значения
  `NaN` заменяются на `0.0`, per-feature flags не добавляются.
- `relaxed` — timestep маскируется, только если все фичи `NaN`; значения `NaN`
  заменяются на `0.0`, а к каждой фиче добавляется бинарный missing-флаг.
  Поэтому `0` остаётся численным placeholder, а информация о частичном
  пропуске не теряется.

Текущий default — `relaxed`. При таком contract producer должен передавать
`NaN` для отсутствующих context candles; Transformer строит mask и, в relaxed
mode, missing-флаги из исходных `NaN` до любых tensor ops, и только затем
заменяет `NaN -> 0`.

Trading head получает фактический последний незамаскированный timestep, а не
просто число валидных токенов: ведущие, внутренние и хвостовые пропуски поэтому
обрабатываются корректно. Для полностью пустой последовательности первый
timestep с заполненными нулями значениями временно остаётся unmasked как
безопасный placeholder; relaxed missing-флаги при этом сохраняются. Это не даёт
attention создать non-finite значения.

## Метрики обучения

`fit` и `fit-stream` один раз печатают конфигурацию запуска, а затем компактную
summary-строку для каждого epoch. Standalone `fit-stream` дополнительно пишет
номер входного frame; Flight fit пишет одну агрегированную строку на job-wide
эпоху без `frame`. Loss schedule продвигается по выбранному `--loss-schedule`:

- `none` — всегда используется `--loss-stage`;
- `epoch` — stage считается от epoch внутри текущего standalone frame или
  всего Flight job;
- `step` — stage считается от глобального optimizer step и не сбрасывается
  между frames.

```text
frame=1 epoch=2 monitor_value=3.82703 loss=-3.149016 mae=0.0225603 baseline=0.00589499 skill=3.82703x status=WORSE sigma=0.0231593 grad=476.013 rows=67249 batches=263 time=181.7s stage=1/4
```

Summary показывает основной результат эпохи, сравнение с baseline, среднюю
`sigmaR`, gradient norm до clipping, объём данных, время и активный loss stage.
`status=BETTER` означает `skill < 1.0`, `status=WORSE` — что baseline пока лучше
модели. Для non-finite skill выводятся `skill=n/a status=N/A`.

Полный набор метрик доступен в JSONL:

- `loss` — итоговый loss после всех весов компонентов;
- `batch_size`, `loss_schedule`, `stage_size`, `max_loss_stage`, `hidden`,
  `layers`, `seq_len`, `device` — параметры запуска, записываются в каждую
  строку JSONL;
- `loss_ret`, `loss_prob`, `loss_ev`, `loss_vol` — компоненты loss;
- `sigma_min`, `sigma_p05`, `sigma_mean` — статистика предсказанного `sigmaR`
  для диагностики Gaussian NLL;
- `ret_mae`, `ret_rmse` — ошибка прогноза `meanR` против target `meanR`;
- `ret_mae_baseline` — MAE нулевого прогноза `meanR=0`; полезно сравнивать с
  `ret_mae`, чтобы видеть, лучше ли модель простой нулевой гипотезы;
- `ret_mae_skill` — отношение `ret_mae / ret_mae_baseline`; меньше `1.0`
  означает лучше baseline;
- `ret_mae_improvement` — `1 - ret_mae_skill`;
- `monitor_value`, `best_monitor`, `baseline_passed`, `checkpoint_best` —
  состояние критерия early stopping и выбора checkpoint;
- `grad_norm` — gradient norm до clipping;
- `rows`, `batches` — объём данных в проходе;
- `nan_ratio` — доля NaN во входном `src`;
- `masked_token_ratio` — доля timesteps, скрытых от attention текущим
  `--mode`;
- `complete_token_ratio` — доля timesteps без `NaN`;
- `partial_token_ratio` — доля timesteps с частью заполненных фичей и `NaN`;
- `empty_token_ratio` — доля timesteps, где все фичи `NaN`;
- `step` — глобальный номер optimizer step к концу строки метрик;
- `lr` — текущий learning rate;
- `loss_stage` — активный этап функции потерь;
- `input_pipeline_ms` — host wall time получения batch из input pipeline,
  включая Arrow replay, validation, streaming wait и CPU shuffle;
- `missing_stats_ms` — host wall time расчёта NaN и token ratios;
- `host_to_device_ms` — host wall time вызовов CPU-to-device transfer;
- `train_step_ms` — host wall time forward, loss, backward, optimizer step и
  сбора batch metrics;
- `elapsed_ms` — время training pass.

Фазовые значения являются диагностикой host pipeline. Они не добавляют CUDA
synchronize и поэтому не являются точным GPU kernel time; их сумма может быть
меньше `elapsed_ms` на служебные операции между измеряемыми фазами. Как и
`elapsed_ms`, они исключаются из deterministic equivalence ML-state.

Чтобы сохранять метрики, добавьте `--metrics-out`:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --metrics-out=train.jsonl
```

Для `fit-stream` используется тот же аргумент. Каждая строка — один JSON object
с полным набором числовых полей, параметрами запуска и контекстом `epoch`;
standalone stream также добавляет `frame`. Non-finite значения сериализуются как
JSON `null`.

Построить SVG-графики по JSONL:

```bash
./.venv/bin/python ./app/main.py plot-metrics train.jsonl \
  --plots-dir=./metrics/plots
```

`plot-metrics` создаёт отдельные SVG-файлы для `loss`, компонентов loss,
`sigma_min`, `sigma_p05`, `sigma_mean`, `grad_norm`, `nan_ratio`, token ratios,
`rows`, `batches`, `step`, `lr`, `loss_stage`, фазовые durations и
`elapsed_ms`. Output directory
создаётся автоматически; существующие одноимённые SVG перезаписываются.
Невалидная JSON-строка в `METRICS_FILE` прерывает команду.
