# Функция потерь

> Тип: справочник. Текущая семантика model heads и декларативного objective.

Нормативные wire schema и public target identity задаёт
[`Flight v9 contract`](../app/contracts/flight/v9/README.md). Consumer выбирает
каноническое непустое подмножество из следующего target universe:

| Public semantic | Внутреннее представление | Direct operator | Диапазон prediction |
| --- | --- | --- | --- |
| `MeanReturn` | `tanh(head)` | `SmoothL1` | `[-1, 1]` |
| `SigmaReturn` | `sigmoid(head)` | `SmoothL1` | `[0, 1]` |
| `ProbTP` | logit; при публикации `sigmoid` | `BinaryCrossEntropyWithLogits` | `[0, 1]` |
| `ProbSL` | logit; при публикации `sigmoid` | `BinaryCrossEntropyWithLogits` | `[0, 1]` |
| `VolatilityNext` | `sigmoid(head)` | `LogMSE` | `[0, 1]` |
| `HittingProbTP` | logit; при публикации `sigmoid` | `BinaryCrossEntropyWithLogits` | `[0, 1]` |

Порядок выбранных targets сохраняет порядок этой таблицы. Worker физически
создаёт только выбранные public heads; `tgt` и prediction имеют ту же ширину и
тот же порядок. `SigmaReturn` — public target-координата, а не private
Gaussian scale. `ProbTP` и `ProbSL` независимы и не обязаны давать сумму `1`.

## Declarative objective v1

Fit create передаёт отдельные поля `targets` и `objective`. Полный canonical
документ `{targets, objective}` хешируется по RFC 8785/JCS и сохраняется в
`mlContract`, checkpoint и model metadata.

Закрытая objective schema v1 фиксирует:

- `aggregation: "WeightedSum"`;
- `reduction: "GlobalRowMean"`;
- ровно один direct loss для каждого выбранного target в том же порядке;
- ноль или более совместимых auxiliary losses;
- строго положительные статические веса;
- `balancing: {"operator": "Static"}`.

Ни training policy, ни device/AMP, ни diagnostics в objective identity не
входят. Все объявленные компоненты активны с первого optimizer step; loss
stages и stage schedule отсутствуют.

## Прямые компоненты

Для выбранной координаты `i` Transformer применяет закреплённый за semantic
operator:

```text
SmoothL1:
  mean(SmoothL1(publicHead[i], target[i]))

BinaryCrossEntropyWithLogits:
  mean(BCEWithLogits(publicLogit[i], target[i]))

LogMSE:
  mean((log(publicHead[i] + 1e-6) - log(target[i] + 1e-6))²)
```

Итоговая прямая часть равна сумме `weight × directLoss`. Подмена operator для
конкретного target, пропуск direct loss или другой порядок отклоняются до
создания job.

## Вспомогательные компоненты

`GaussianNLL` требует `MeanReturn` и добавляет private `returnScale` head:

```text
variance = returnScale² + 1e-6
gaussianNll = mean(0.5 × (
  (target[MeanReturn] - prediction[MeanReturn])² / variance
  + log(variance)
))
```

Gaussian NLL может быть отрицательным при малой дисперсии; это само по себе не
означает ошибку.

`ExpectedValue` требует `ProbTP` и `ProbSL`:

```text
delta = prediction[ProbTP] - prediction[ProbSL]
expectedValueLoss = -mean(delta)
```

`RiskAdjustedExpectedValue` является отдельным operator. Он требует
`ProbTP`, `ProbSL` и `GaussianNLL`, использует private scale и объявленный
`riskPenalty`:

```text
delta = prediction[ProbTP] - prediction[ProbSL]
risk = detach(returnScale) × abs(delta)
riskAdjustedExpectedValueLoss = -mean(delta - riskPenalty × risk)
```

`ExpectedValue` и `RiskAdjustedExpectedValue` нельзя включить одновременно.
Вес каждого auxiliary component применяется внешней `WeightedSum`-агрегацией.

## Selection и diagnostics

Checkpoint selection является training policy. Если selection включён, score
равен сумме global-row-mean direct losses с весами из objective; auxiliary
components в score не входят. Параметры `minDelta` и `patience` не меняют
`objectiveConfigSha256`.

Gradient-interaction diagnostics — необязательное наблюдение, а не часть loss.
На выбранных optimizer steps Transformer вычисляет gradient каждого
target-task в общей representation model head, их нормы и попарные cosine.
`GaussianNLL` входит в компонент `target:MeanReturn`; совместные EV operators
представлены отдельным auxiliary component. Diagnostics не выполняет optimizer
step, не балансирует gradients и не меняет checkpoint compatibility.
