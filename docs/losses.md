# Функция потерь

> Тип: справочник. Текущая семантика loss stages и model heads.

Нормативную public target identity задаёт текущий
[`Flight v5 contract`](../app/contracts/flight/v5/README.md). Внутри worker
модель возвращает семь значений: шесть public heads и один private Gaussian
scale. Flight boundary публикует только первые шесть в target-space.

| Индекс | Public semantic | Внутреннее представление | Диапазон prediction |
| --- | --- | --- | --- |
| `0` | `MeanReturn` | `tanh(meanHead)` | `[-1, 1]` |
| `1` | `SigmaReturn` | `sigmoid(sigmaHead)` | `[0, 1]` |
| `2` | `ProbTP` | logit; при публикации `sigmoid` | `[0, 1]` |
| `3` | `ProbSL` | logit; при публикации `sigmoid` | `[0, 1]` |
| `4` | `VolatilityNext` | `sigmoid(volatilityHead)` | `[0, 1]` |
| `5` | `HittingProbTP` | logit; при публикации `sigmoid` | `[0, 1]` |
| private | `returnScale` | `softplus(scaleHead) + 1e-6` | не публикуется |

`SigmaReturn` — нормализованный target Inventory, а не Gaussian scale.
`ProbTP` и `ProbSL` независимы и не обязаны давать сумму `1`.

## Прямые компоненты

Каждая target-координата непосредственно обучает одноимённую public head:

```text
L0 = mean(SmoothL1(MeanReturn, target[0]))
L1 = mean(SmoothL1(SigmaReturn, target[1]))
L2 = mean(BCEWithLogits(probTpLogit, target[2]))
L3 = mean(BCEWithLogits(probSlLogit, target[3]))
L4 = mean((log(VolatilityNext + 1e-6)
           - log(target[4] + 1e-6))²)
L5 = mean(BCEWithLogits(hittingProbTpLogit, target[5]))
```

На максимальном stage все шесть весов `w0…w5` строго положительны:

```text
directLoss = Σ wi × Li
```

`--direct-loss-weights` задаёт веса в порядке target vector.

## Вспомогательные компоненты

Private Gaussian NLL использует только `MeanReturn`, `target[0]` и отдельный
`returnScale`:

```text
variance = returnScale² + 1e-6
gaussianNll = mean(0.5 × (
  (target[0] - MeanReturn)² / variance + log(variance)
))
```

Значение Gaussian NLL может быть отрицательным при малой дисперсии. Это само по
себе не означает ошибку.

EV/risk regularizer использует независимые вероятности TP и SL:

```text
ev = sigmoid(probTpLogit) - sigmoid(probSlLogit)
risk = detach(returnScale) × abs(ev)
expectedValueLoss = -0.3 × mean(ev - 0.1 × risk)
```

Auxiliary losses влияют на `trainingLoss`, но не входят в checkpoint selection
score и не меняют публичную семантику.

## Этапы

`--loss-stage` зафиксирован в `4`. `--loss-schedule` управляет переходом к
максимальному stage:

| Stage | Активные компоненты |
| --- | --- |
| `1` | `L0`, `L1`, Gaussian NLL |
| `2` | stage 1 + `L2`, `L3`, `L5` |
| `3` | stage 2 + EV/risk |
| `4` | stage 3 + `L4` |

- `none` — сразу использовать stage 4;
- `epoch` — повышать stage каждые `--stage-size` epochs;
- `step` — повышать stage каждые `--stage-size` optimizer steps.

При defaults `--loss-schedule=epoch --stage-size=5` epochs `1..5` используют
stage 1, `6..10` — stage 2, `11..15` — stage 3, с epoch 16 — stage 4.
Обучение обязано завершить хотя бы одну полную epoch stage 4.

## Выбор checkpoint

Если включён `--select-best-checkpoint`, score вычисляется только для полной
epoch stage 4:

```text
selectionScore = Σ wi × globalRowMean(Li)
```

`globalRowMean` означает взвешивание batch-значений числом строк. Границы
payload и batch не меняют score. Auxiliary losses исключены. Нефинитный или
неполный score завершает обучение ошибкой.

Candidate заменяет best только при
`score < best - selectionMinDelta`; равенство сохраняет более ранний
checkpoint. `--selection-patience=0` отключает early stopping, но оставляет
выбор best. Если selection выключен, публикуется последний checkpoint
максимального stage.
