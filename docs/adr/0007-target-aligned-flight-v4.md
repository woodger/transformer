# ADR 0007: target-aligned ML-контракт Flight v4

- Статус: принято
- Дата: 2026-08-12
- Заменяет публичный ML-контракт Flight v3 из ADR 0005
- Сохраняет durable streaming lifecycle и fencing из ADR 0005

## Контекст

В Flight v3 target Inventory и публичный prediction Transformer имели ширину
шесть, но не совпадали по смыслу. Inventory передавал:

```text
0 meanReturn
1 sigmaReturn
2 probTP
3 probSL
4 volatilityNext
5 hittingProbTP
```

Модель возвращала `meanR`, внутренний Gaussian scale, три raw logit и
volatility. Часть target-координат не имела прямого supervision, а Consumer
сравнивал значения с разной семантикой по одинаковому индексу. В частности,
`sigmaReturn = tanh(EWMA sigma × 100)` не является Gaussian uncertainty scale
для `meanReturn`.

Изменение наблюдаемой семантики prediction является breaking change. Оно не
может быть скрыто под прежним `transformer.prediction.v2` или Flight v3.

## Решение

Текущий публичный протокол — только Transformer Flight v4. V3 dispatcher,
action aliases, descriptor aliases и fallback отсутствуют. Durable streaming,
две оси состояния, client-generated `jobId`, fencing, revision pagination и
global-epoch recovery переносятся без изменения семантики.

Публичные идентификаторы:

```text
targetSchemaId     inventory.target.v1
predictionSchemaId transformer.prediction.target-aligned.v1
objectiveId        transformer.objective.target-aligned.v1
checkpointFormat   transformer-checkpoint-v3
worker contract    transformer-worker v3
recovery format    transformer-training-recovery-v3
```

Физические input schema IDs остаются
`inventory.sequence.fit.v2` и `inventory.sequence.predict.v2`: их Arrow layout
не изменился.

## Публичный target-space

Prediction и target имеют одинаковую фиксированную семантику:

| Индекс | Семантика | Диапазон |
| --- | --- | --- |
| `0` | `meanReturn` | `[-1, 1]` |
| `1` | `sigmaReturn` | `[0, 1]` |
| `2` | `probTP` | `[0, 1]` |
| `3` | `probSL` | `[0, 1]` |
| `4` | `volatilityNext` | `[0, 1]` |
| `5` | `hittingProbTP` | `[0, 1]` |

Все значения конечны. `probTP` и `probSL` независимы; инвариант
`probTP + probSL = 1` не вводится. Raw logits не пересекают Flight boundary.
Worker публикует:

```text
[
  predictedMeanReturn,
  predictedSigmaReturn,
  sigmoid(probTpLogit),
  sigmoid(probSlLogit),
  predictedVolatilityNext,
  sigmoid(hittingProbTpLogit)
]
```

Gaussian uncertainty при необходимости использует отдельную седьмую private
head `returnScale`. Она не меняет ширину и семантику публичного prediction.

## Training objective

Каждая публичная координата имеет собственный прямой supervised path:

```text
L0 = SmoothL1(meanReturn, target[0])
L1 = SmoothL1(sigmaReturn, target[1])
L2 = BCEWithLogits(probTpLogit, target[2])
L3 = BCEWithLogits(probSlLogit, target[3])
L4 = mean((log(volatilityNext + 1e-6)
           - log(target[4] + 1e-6))²)
L5 = BCEWithLogits(hittingProbTpLogit, target[5])
```

На максимальном stage активны все `L0…L5`. Их веса строго положительны и
являются частью неизменяемой training configuration. Private Gaussian NLL и
EV/risk могут участвовать только как auxiliary components. Они не заменяют
direct supervision и не меняют публичную семантику координат.

`trainingLoss` управляет оптимизацией и включает активные direct и auxiliary
components. Публичные quality metrics — только MAE и RMSE отдельно для каждой
из шести target-координат. Общая MAE/MSE по разнородным координатам не является
метрикой качества или критерием выбора модели.

## Этапы objective

`lossStage` зафиксирован в `4`. Schedule может быть `none`, `epoch` или `step`:

| Stage | Активные компоненты |
| --- | --- |
| `1` | `L0`, `L1`, private Gaussian NLL |
| `2` | stage 1 + `L2`, `L3`, `L5` |
| `3` | stage 2 + auxiliary EV/risk |
| `4` | stage 3 + `L4` |

Это гарантирует direct supervision всех публичных heads на максимальном
stage. Конфигурация, которая не успевает завершить полную epoch stage 4,
завершается ошибкой и не публикует модель.

## Выбор checkpoint

Есть два однозначных режима.

1. Selection включён — best-checkpoint и early stopping работают только после
   полной epoch максимального stage.
2. Selection выключен — выполняется фиксированное число epochs и публикуется
   последний checkpoint максимального stage.

При первом полном проходе stage 4 сбрасываются прежние best, patience и
baseline. Candidate score включает только `L0…L5`:

```text
candidateScore = Σ directLossWeight[i] × globalRowMean(Li)
```

Компоненты агрегируются по всем строкам, а не как среднее batch-метрик. Поэтому
payload и batch boundaries не меняют score. Все шесть весов положительны.
Auxiliary NLL и EV не участвуют в выборе.

Candidate считается лучшим только при
`candidateScore < bestScore - minDelta`. Равенство сохраняет более ранний
checkpoint. Нефинитный или неполный score является ошибкой обучения и не может
стать кандидатом.

## Каноническая objective configuration

`objectiveConfigSha256` — SHA-256 от UTF-8 JSON полной objective
configuration. JSON сериализуется с сортировкой ключей, без пробелов и без
NaN/Infinity. Документ включает:

- тип, semantic index, `global_row_mean` normalization и вес каждого `L0…L5`;
- private auxiliary losses и их коэффициенты;
- schedule, stage size, maximum stage и состав каждого stage;
- selection enabled, weights, `minDelta`, `patience`;
- `maximum_only_reset` stage policy;
- `earliest` tie policy и `none` baseline policy;
- `fail_training` для invalid score;
- политику best либо last-maximum-stage publication.

`objectiveConfigSha256` входит в передаваемый при `job.create` `mlContract`,
checkpoint, recovery metadata, published model metadata и результат
`model.describe`. Fit отклоняется до создания job, если хеш не соответствует
фактической `trainingConfig` и текущему objective Transformer.

## Model lifecycle

Checkpoint и published metadata содержат:

```text
targetSchemaId
predictionSchemaId
objectiveId
objectiveConfigSha256
checkpointFormat
targetWidth
dataContractSha256
modelConfig
```

Predict разрешён только при совпадении data contract, полного `mlContract` и
model configuration. Стабильные ошибки:

- другой корректный protocol/checkpoint contract — `MODEL_SCHEMA_MISMATCH`;
- текущий format с отсутствующей или противоречивой semantic metadata —
  `MODEL_CORRUPT`;
- отсутствующий artifact — `MODEL_UNAVAILABLE`;
- неизвестный model identity — `NOT_FOUND`.

Alias остаётся owner-scoped и атомарно разрешается в конкретный неизменяемый
`resolvedModelRef` при `job.create`.

## Recovery и эквивалентность

Worker v3 и recovery format v3 меняются атомарно с Flight v4. Recovery хранит
model, optimizer, scaler, training/RNG/shuffle/selection state и
`objectiveConfigSha256`. State от прежнего objective не восстанавливается.

Критерий детерминированной эквивалентности:

```text
одинаковый ordered dataset
+ seed
+ trainingConfig с deterministic=true
+ одинаковые hardware/runtime
→ одинаковый row/shuffle order
→ одинаковые optimizer steps
→ одинаковые ML-метрики после каждой epoch
→ одинаковое ML-state
→ семантически одинаковый checkpoint
→ одинаковая итоговая модель
```

Wall-clock telemetry исключается. Checkpoint сравнивается после
десериализации: model, optimizer, scaler, training/RNG/shuffle/selection state
и выбор кандидата. SHA-256 сериализованного файла не является критерием
эквивалентности.

## Cutover

Миграция `0006` необратима и образует одну breaking-границу:

1. остановить Inventory workers и Transformer;
2. сохранить резервную копию PostgreSQL и model artifacts;
3. применить миграцию `0006`;
4. развернуть только Inventory v4 + Transformer v4;
5. переобучить модели и получить новые `modelRef`.

Миграция удаляет v3 jobs, idempotency, attempts, inputs, outputs и recovery
state. Access tokens, model identities и aliases сохраняются. Старые models не
получают `mlContract` и остаются недоступными для v4 prediction. Неявной
сертификации checkpoint нет.

## Проверки приёмки

- изменение `target[i]` создаёт gradient для соответствующей public head;
- все шесть heads имеют direct gradient на stage 4;
- координаты 2, 3 и 5 публикуются после sigmoid;
- output проходит finite/range validation до публикации;
- per-target metrics не переставляют координаты;
- старый checkpoint не принимается prediction operation;
- `capabilities` и `model.describe` возвращают точные semantic IDs;
- closed и delayed-streaming input дают детерминированно эквивалентное ML-state;
- payload partitioning не меняет row order, score или checkpoint selection;
- recovery сохраняет objective и selection state;
- invalid selection score завершает обучение ошибкой;
- Inventory обязан выполнить новый fit и независимый per-target model test.

## Последствия

Consumer может сравнивать `prediction[i]` и `target[i]` без локального mapping.
Price этого решения — полный protocol cutover, новый worker/checkpoint/recovery
format и обязательное переобучение прежних моделей.
