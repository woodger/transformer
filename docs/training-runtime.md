# Training runtime и checkpoint

> Type: Reference. Формат checkpoint, обучение, missing-data semantics и
> training telemetry локального CLI и Flight worker.

Параметры команд находятся в [справочнике CLI](./cli/index.md). Состав model
heads, declarative objective и формулы loss operators принадлежат
[описанию функции потерь](./losses.md), а checkpoint selection, recovery и
runtime telemetry — этому документу. Нормативный remote ML-контракт находится в
[`app/contracts/flight/v9`](../app/contracts/flight/v9/README.md).
Rationale target-aligned public semantics сохранён в
[ADR 0007](./adr/0007-target-aligned-flight-v4.md); текущие форматы и значения
определяют contract и этот reference.

## Checkpoint contract

Текущий формат — `transformer-checkpoint-v5`. Он содержит только закрытый
набор полей:

- `state_dict` и версию приложения;
- полные `model_config` и `train_config`;
- physical/model input schema;
- `data_contract`;
- полный `ml_contract` с `objectiveConfigSha256`;
- канонический документ `objective` с выбранными targets;
- metadata выбора checkpoint.

Checkpoint, опубликованный Flight fit, дополнительно сохраняет разрешённый
`initialization` с lineage parent checkpoint. У локального fit parent
отсутствует и отдельная Flight lineage metadata не создаётся.

Model config фиксирует `seq_len`, `feature_dim`, `hidden`, `layers`, `dropout`,
`nhead`, `context_mode` и public `out_dim`. Значение `out_dim` равно числу
выбранных targets от `1` до `6`. Private `returnScale` head существует только
для objective с `GaussianNLL` или `RiskAdjustedExpectedValue` и не меняет
public width.

Loader принимает только точный формат v5. Предыдущие wrapped и raw legacy
checkpoint не интерпретируются автоматически. Для Flight prediction другой
корректный format даёт `MODEL_SCHEMA_MISMATCH`; текущий format с неполной или
противоречивой semantic metadata даёт `MODEL_CORRUPT`.

Локальные `fit` и `fit-stream` всегда начинают со случайной инициализации.
`--checkpoint-out` задаёт конечную цель: существующий файл атомарно заменяется
только после успешного обучения и validation.

Flight fit также может использовать опубликованную generation как weights-only
initialization. Strict warm start требует те же model/data/ML contracts;
cross-instrument transfer отдельно разрешает новый data-contract digest при
неизменных structural fields, model config, targets и objective. В обоих
случаях optimizer, AMP scaler, RNG, progress и checkpoint selection не
наследуются. Результатом остаётся новая immutable generation с data contract
текущего fit, а не изменение parent и не продолжение его training run. Точную
wire-форму `initialization` задаёт
[Flight contract](../app/contracts/flight/v9/README.md#инициализация-fit-и-lineage-модели).
Recovery относится к состоянию уже созданного нового job.

## Target selection и objective

Flight fit выбирает каноническое непустое подмножество:

```text
MeanReturn, SigmaReturn, ProbTP, ProbSL, VolatilityNext, HittingProbTP
```

`tgt`, model heads, prediction и per-target telemetry содержат только выбранные
координаты в этом порядке. Общая MAE/MSE по разнородным величинам не
вычисляется и не используется для оценки модели. Полный `{targets, objective}`
входит в `objectiveConfigSha256`; все объявленные losses активны с первого
optimizer step.

Training policy задаёт optimizer, число epochs, weight decay, reproducibility,
AMP и optional checkpoint selection. Она сохраняется отдельно от objective и
не меняет его digest. Execution diagnostics также имеют отдельную schema и не
влияют на model identity.

## Selection и early stopping

По умолчанию selection выключен: выполняется фиксированное число epochs и
сохраняется checkpoint последней epoch.

Включение:

```bash
./.venv/bin/python ./app/main.py fit ./data/train.arrow \
  --seq-len=20 \
  --select-best-checkpoint \
  --selection-min-delta=0.001 \
  --selection-patience=5
```

Selection работает с первой завершённой epoch. Score равен сумме direct losses
выбранных targets с положительными objective weights, агрегированных по всем
строкам. Auxiliary losses не участвуют.

Candidate принимается только при `score < best - minDelta`; tie сохраняет
ранний checkpoint. Нефинитный или неполный score завершает обучение ошибкой.
`--selection-patience=0` отключает остановку, но сохраняет выбор лучшего
candidate.

Selection policy входит в `train_config` и recovery state, но не в
`objectiveConfigSha256`.

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

Текущий формат — `transformer-training-recovery-v5`. Checkpoint создаётся
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

Компактная строка epoch для выбранных targets выглядит так:

```text
epoch=4 selection=0.1842 loss=0.233100 MeanReturn_mae=0.012 SigmaReturn_mae=0.021 grad_mean=1.000 rows=67249 batches=263 time=181.7s
```

`selection=n/a` означает, что checkpoint selection выключен.

JSONL содержит:

- `loss`, структурированные `directLosses` и `auxiliaryLosses`;
- `targets` и структурированные `targetMetrics` с MAE/RMSE;
- optional `gradientInteractions` с числом samples, mean component norms и
  mean pairwise cosine;
- `selection_score`, `checkpoint_best`, `best_selection_score`;
- `trainingBatchesCompleted`, `optimizerUpdatesApplied`,
  `optimizerUpdatesSkipped`, `ampOverflowBatches`;
- `finiteGradientBatches`, `nonFiniteGradientBatches`,
  `preClipGradientNormMean`, `preClipGradientNormMax`,
  `preClipGradientNormP95`;
- `rows`, `batches`, `step`, `lr`;
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

Mean, max и P95 считаются только по finite pre-clip full-model gradient norms. P95
использует nearest-rank policy; если все norms non-finite, эти три поля равны
`null`.

Если `diagnostics.gradientInteractions` задан, sample выполняется после каждого
указанного числа optimizer steps. Для выбранного batch сохраняются gradients
objective components относительно общей representation model head. Нулевой
norm не превращается в искусственный cosine: такая пара не входит в итоговое
среднее. Diagnostics увеличивает стоимость только sampled steps и остаётся
best-effort telemetry.

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
