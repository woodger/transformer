# Функция потерь

> Тип: справочник. Текущая семантика loss stages и model heads.

Нормативную public target identity задаёт текущий
[`Flight v6 contract`](../app/contracts/flight/v6/README.md). Внутри worker
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

`SigmaReturn` — нормализованная public target-координата, а не Gaussian scale.
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

Stage composition определяет, какие компоненты входят в loss:

| Stage | Активные компоненты |
| --- | --- |
| `1` | `L0`, `L1`, Gaussian NLL |
| `2` | stage 1 + `L2`, `L3`, `L5` |
| `3` | stage 2 + EV/risk |
| `4` | stage 3 + `L4` |

Порядок перехода между stages, CLI modes и требование завершить stage 4 задаёт
[training runtime](./training-runtime.md#loss-schedule). Агрегация score, выбор
candidate и early stopping находятся в его разделе
[Selection](./training-runtime.md#selection-и-early-stopping).
