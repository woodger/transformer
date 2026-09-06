# Функция потерь

> Тип: справочник. Текущая семантика model outputs и consumer-neutral
> декларативного Objective.

Нормативную модель задают
[`semantic/v1`](../app/contracts/semantic/v1/README.md) и
[`Flight v12`](../app/contracts/flight/v12/README.md). Consumer materializer
передаёт self-contained ordered target contract и Objective. Transformer
считает target identities непрозрачными: ни transformation, ни operator, ни
private resource не выбираются по имени target.

## Target slots и model output

Каждый target slot независимо объявляет:

- opaque `identity`;
- `observedConstraint` для входного `y`;
- `lossInputTransformation` raw model coordinate;
- `publicPredictionTransformation` той же raw coordinate.

Runtime поддерживает constraints `Finite` и `ClosedInterval`, а также
transformations `Identity`, `Tanh` и `Sigmoid`. На Arrow boundary `y` и public
prediction имеют finite Float32. Внутренний tensor dtype остаётся деталью
Transformer и может меняться при AMP.

Model head выдаёт по одной raw coordinate на ordered target slot. Direct loss
получает результат `lossInputTransformation`; prediction — результат
`publicPredictionTransformation`. Поэтому direct operator не определяет public
representation: например, `SmoothL1` может работать как с `Tanh`, так и с
`Sigmoid` estimate.

## Objective language v1

Objective задаёт:

- `WeightedSum` aggregation и `GlobalRowMean` reduction;
- ровно один direct component для каждого target slot в том же порядке;
- ноль или более auxiliary components;
- положительный weight и стабильную identity каждого component;
- typed role references на target identities и private resource identities;
- отсортированные по ASCII abstract resource declarations.

Все components активны с первого optimizer step. Training policy, device/AMP,
checkpoint selection и diagnostics в Objective не входят. TargetContract,
Objective и полный ModelContract имеют отдельные D1 digests.

## Direct operators

Для slot `T` используются явно связанные `LossEstimate<T>` и `Observed<T>`:

```text
SmoothL1:
  mean(SmoothL1(lossEstimate[T], observed[T]))

BinaryCrossEntropyWithLogits:
  mean(BCEWithLogits(rawLogit[T], observedProbability[T]))

LogMSE:
  mean((log(positiveEstimate[T] + 1e-6)
        - log(nonNegativeObserved[T] + 1e-6))²)
```

`BinaryCrossEntropyWithLogits` требует `Identity` loss-input transformation и
observed interval внутри `[0, 1]`. `LogMSE` требует положительный estimate,
который в language v1 выражается `Sigmoid`, и неотрицательный observed target.
Итоговая direct часть — сумма `weight × component mean`.

## Private resources и auxiliary operators

Language v1 поддерживает resource kind `PositiveScalarPerObservation`. Это
private differentiable model output, принадлежащий checkpoint. Его identity
локальна Objective и позволяет нескольким operators использовать один и тот же
resource; способ PyTorch parameterization, tensor layout и хранения остаётся
внутренним устройством Transformer. Private values не входят в prediction.

Каждый объявленный resource должен быть использован и иметь структурный путь к
total loss через component с положительным weight и gradient-producing role.
Нулевой gradient на отдельном batch не является нарушением. В текущем языке
`GaussianNLL` обучает scale, а использование scale в
`RiskAdjustedExpectedValue` намеренно выполняется через stop-gradient.

`GaussianNLL` связывает location estimate, observed location одного slot и
positive scale resource:

```text
variance = scale² + 1e-6
gaussianNll = mean(0.5 × (
  (observedLocation - locationEstimate)² / variance + log(variance)
))
```

`ExpectedValue` связывает две разные public probability coordinates:

```text
delta = positiveOutcomeProbability - negativeOutcomeProbability
expectedValueLoss = -mean(delta)
```

`RiskAdjustedExpectedValue` дополнительно связывает uncertainty scale:

```text
delta = positiveOutcomeProbability - negativeOutcomeProbability
risk = detach(uncertaintyScale) × abs(delta)
riskAdjustedExpectedValueLoss = -mean(delta - riskPenalty × risk)
```

`ExpectedValue` и `RiskAdjustedExpectedValue` могут присутствовать одновременно
как независимые weighted components. Role bindings, weights, parameters,
resource declarations и sharing входят в Objective/model compatibility.

## Selection и diagnostics

Checkpoint selection является Transformer-owned training policy. Если она
включена, score равен weighted sum global-row-mean direct losses; auxiliary
components в score не входят.

Gradient-interaction diagnostics являются отдельным best-effort наблюдением.
Они используют стабильные component identities, а для direct components также
opaque target identity и производный physical index. Diagnostics не выполняет
optimizer step, не балансирует gradients и не меняет model compatibility.
