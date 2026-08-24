# Training runtime и checkpoint

> Type: Reference. Формат checkpoint, обучение, missing-data semantics и
> training telemetry локального CLI и Flight worker.

Параметры команд находятся в [справочнике CLI](./cli/index.md). Состав model
heads, формулы loss components и stage composition принадлежат
[описанию функции потерь](./losses.md), а schedule transitions, checkpoint
selection, recovery и runtime telemetry — этому документу. Нормативный remote
ML-контракт находится в
[`app/contracts/flight/v5`](../app/contracts/flight/v5/README.md).
Rationale target-aligned public semantics сохранён в
[ADR 0007](./adr/0007-target-aligned-flight-v4.md); текущие форматы и значения
определяют contract и этот reference.

## Checkpoint contract

Текущий формат — `transformer-checkpoint-v4`. Он содержит только закрытый
набор полей:

- `state_dict` и версию приложения;
- полные `model_config` и `train_config`;
- physical/model input schema;
- `data_contract`;
- полный `ml_contract` с `objectiveConfigSha256`;
- каноническую `objective_config`;
- metadata выбора checkpoint.

Model config фиксирует `seq_len`, `feature_dim`, `hidden`, `layers`, `dropout`,
`nhead`, `context_mode` и public `out_dim=6`. Фактическая модель имеет ещё одну
private uncertainty head, которая не меняет public width.

Loader принимает только точный формат v4. Предыдущие wrapped и raw legacy
checkpoint не интерпретируются автоматически. Для Flight prediction другой
корректный format даёт `MODEL_SCHEMA_MISMATCH`; текущий format с неполной или
противоречивой semantic metadata даёт `MODEL_CORRUPT`.

`fit` и `fit-stream` всегда создают новую модель. `--checkpoint-out` задаёт
конечную цель: существующий файл атомарно заменяется только после успешного
обучения и validation.

## Target-aligned objective

Публичный prediction имеет шесть координат в том же порядке, что target:

```text
MeanReturn, SigmaReturn, ProbTP, ProbSL, VolatilityNext, HittingProbTP
```

Для каждой координаты JSONL содержит отдельные MAE и RMSE. Общая MAE/MSE по
шести разнородным величинам не вычисляется и не используется для оценки
модели. `trainingLoss` может содержать direct и auxiliary components, но
checkpoint selection использует только прямые `L0…L5`.

## Loss schedule

Максимальный `--loss-stage` зафиксирован в `4`. Способы перехода:

- `none` — stage 4 активен с первого optimizer step;
- `epoch` — stage повышается каждые `--stage-size` epochs;
- `step` — stage повышается каждые `--stage-size` завершённых training
  batches и может смениться внутри epoch.

Stage 4 непосредственно обучает все шесть public heads. Запуск, в котором не
завершилась ни одна полная epoch stage 4, считается ошибочным и не публикует
checkpoint.

## Selection и early stopping

По умолчанию selection выключен: выполняется фиксированное число epochs и
сохраняется последний checkpoint максимального stage.

Включение:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --select-best-checkpoint \
  --selection-min-delta=0.001 \
  --selection-patience=5
```

Selection начинает работать только после полной epoch stage 4. При входе в
этот режим прежние best/patience/baseline сбрасываются. Score равен сумме шести
direct losses с положительными `--direct-loss-weights`, агрегированных по всем
строкам. Auxiliary NLL и EV не участвуют.

Candidate принимается только при `score < best - minDelta`; tie сохраняет
ранний checkpoint. Нефинитный или неполный score завершает обучение ошибкой.
`--selection-patience=0` отключает остановку, но сохраняет выбор лучшего
candidate.

Вся selection policy входит в `objectiveConfigSha256` и recovery state.

## File, standalone stream и Flight fit

File fit обучается на одном Arrow IPC dataset. Standalone `fit-stream`
обрабатывает каждый непустой frame отдельным циклом epochs, сохраняя модель и
optimizer между frames. Пустые frames пропускаются; полностью пустой fit
завершается ошибкой. Скрытый spooled режим, используемый внутренними
сценариями recovery, выполняет global epochs над полным набором payloads.

Flight fit использует durable input stream. Epoch 0 начинает обучение после
первого committed непустого payload и может ждать следующий contiguous
ordinal при открытом input. `input.close` задаёт EOF. Последующие epochs
перечитывают закрытый immutable dataset.

RecordBatch и payload boundaries не являются optimizer batch, shuffle window
или epoch boundaries. CPU pipeline закрытых epochs готовит не более одного
следующего batch параллельно текущему training step. Pinned memory и
asynchronous H2D намеренно не используются.

Полная physical/value validation выполняется до durable commit DoPut. Worker
один раз за attempt проверяет receipt, размер и SHA-256, затем использует fast
replay с проверкой schema и row count.

## Recovery

Текущий формат — `transformer-training-recovery-v4`. Checkpoint создаётся
только на границе завершённой global epoch после EOF и содержит:

- model, optimizer и AMP scaler state;
- global epoch и optimizer step;
- Python, NumPy, PyTorch и CUDA RNG state;
- shuffle generator state;
- selection state, best candidate и patience;
- `objectiveConfigSha256`, model/train/data contracts.

Сбой в открытой epoch 0 повторяет её с начала; committed inputs не теряются.
Recovery другого objective, data contract или immutable input manifest
отклоняется.

## Контекстные пропуски

`src` может содержать `NaN` в отдельных features:

- `strict` — timestep маскируется, если отсутствует хотя бы одна feature;
- `relaxed` — timestep маскируется, только если отсутствуют все features;
  дополнительно к значениям передаются per-feature missing flags.

В обоих режимах mask строится до `NaN → 0`. Trading head получает последний
незамаскированный timestep. Полностью пустая последовательность использует
безопасный zero placeholder, чтобы attention не породил non-finite значения.

## Метрики

Результат global epoch, влияющий на selection и recovery, отделён от
необязательной telemetry. Core значения находятся в `worker/training/epoch.py`,
а AMP/gradient counters, per-target errors, phase timings, JSONL и plots — в
`worker/telemetry/`. Ошибка observability отключает запись текущей epoch, но не
меняет optimizer, checkpoint selection или результат fit.

Core loss scalars и optional target/gradient observations по-прежнему
объединяются в одну CUDA→CPU передачу. Если optional часть не может быть
материализована, trainer повторяет только core transfer и продолжает обучение
без telemetry текущей epoch.

Компактная строка epoch выглядит так:

```text
epoch=4 selection=0.1842 loss=0.233100 mean_mae=0.012 sigma_mae=0.021 tp_mae=0.11 sl_mae=0.10 vol_mae=0.03 hit_mae=0.09 grad_mean=1.000 rows=67249 batches=263 time=181.7s stage=4/4
```

`selection=n/a` означает, что текущая epoch не является полной epoch
максимального stage либо selection выключен.

JSONL содержит:

- `loss`, `loss_l0…loss_l5`, `loss_nll`, `loss_ev`;
- по каждой public semantic поля `<name>_mae` и `<name>_rmse`;
- `selection_score`, `checkpoint_best`, `best_selection_score`;
- `trainingBatchesCompleted`, `optimizerUpdatesApplied`,
  `optimizerUpdatesSkipped`, `ampOverflowBatches`;
- `finiteGradientBatches`, `nonFiniteGradientBatches`,
  `preClipGradientNormMean`, `preClipGradientNormMax`,
  `preClipGradientNormP95`;
- `rows`, `batches`, `step`, `lr`;
- `loss_stage`, `minimum_loss_stage`, `maximum_loss_stage`;
- `nan_ratio`, `masked_token_ratio`, `complete_token_ratio`,
  `partial_token_ratio`, `empty_token_ratio`;
- `input_pipeline_ms`, `missing_stats_ms`, `host_to_device_ms`,
  `train_step_ms`, `elapsed_ms`.

Фазовые таймеры являются host wall-clock telemetry и не добавляют отдельную
CUDA synchronization. Loss/gradient scalars объединяются в один CUDA tensor и
переносятся на CPU одной операцией за batch. Wall-clock поля исключены из
критерия deterministic equivalence.

`step` и `trainingBatchesCompleted` означают число завершённых training
batches. Фактически применённые optimizer updates считаются отдельно: при AMP
overflow GradScaler пропускает update. Инварианты epoch:

```text
trainingBatchesCompleted
  = optimizerUpdatesApplied + optimizerUpdatesSkipped
  = finiteGradientBatches + nonFiniteGradientBatches
```

Mean, max и P95 считаются только по finite pre-clip gradient norms. P95
использует nearest-rank policy; если все norms non-finite, эти три поля равны
`null`.

Сохранение и визуализация:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --metrics-out=train.jsonl

./.venv/bin/python ./app/main.py plot-metrics train.jsonl \
  --plots-dir=./metrics/plots
```

`plot-metrics` создаёт отдельный SVG для каждого доступного числового поля.

Flight fit best effort сохраняет завершённые global epochs в immutable
`telemetry/{jobId}/metrics.jsonl`. Durable boundary, OpenSearch
projection и различие между `step` и фактическими AMP optimizer updates
описаны в [политике metrics](./policy/metrics-policy.md) и текущих versioned
[`metrics contracts`](../app/contracts/metrics/). Успешный fit также может
получить run-owned `run-summary.json` с lifecycle durations и counters.
Отсутствие или повреждение telemetry не меняет результат fit и model
publication.
