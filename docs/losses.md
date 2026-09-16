# Семантика objective и loss

> Тип: справочник. Числовая семантика языка objective Semantic v3.

Нормативный документ — [семантическая модель v3](../app/contracts/semantic/v3/README.md).
Этот справочник поясняет formulas, реализованные Transformer; он не вводит
поведение, специфичное для target-а.

## Coordinates target-ов

Каждый упорядоченный непрозрачный target slot объявляет loss-input
transformation и public-prediction transformation над одной raw coordinate
модели. Поддерживаемые transformations: `Identity`, `Tanh` и `Sigmoid`. `y` и
public predictions являются конечными Float32 на границе Arrow; промежуточный
dtype tensor-а принадлежит Transformer.

`observedConstraint` опускается для конечных values и задаётся явно для
закрытого interval. Direct operator указывается matching direct component и
никогда не выводится из target identity.

## Direct operators

Для одной target coordinate `T`:

```text
SmoothL1:
  mean(SmoothL1(lossEstimate[T], observed[T]))

BinaryCrossEntropyWithLogits:
  mean(BCEWithLogits(rawLogit[T], observedProbability[T]))

LogMSE:
  mean((log(positiveEstimate[T] + 1e-6)
        - log(nonNegativeObserved[T] + 1e-6))²)
```

`BinaryCrossEntropyWithLogits` требует identity loss input и observed values в
`[0, 1]`. `LogMSE` требует sigmoid loss estimate и non-negative closed
observation interval. У каждого slot ровно один direct component.

## Resources и auxiliary operators

`PositiveScalarPerObservation` — private positive differentiable output. Он
принадлежит checkpoint-у, может совместно использоваться через непрозрачную
resource identity и не включается в public prediction output. `GaussianNLL`
предоставляет его gradient-producing path. `RiskAdjustedExpectedValue`
намеренно потребляет scale через stop-gradient.

```text
GaussianNLL:
  variance = scale² + 1e-6
  mean(0.5 * ((observed - location)² / variance + log(variance)))

ExpectedValue:
  -mean(positiveProbability - negativeProbability)

RiskAdjustedExpectedValue:
  delta = positiveProbability - negativeProbability
  -mean(delta - riskPenalty * stopGradient(scale) * abs(delta))
```

`ExpectedValue` и `RiskAdjustedExpectedValue` могут сосуществовать. Weights
components положительны. Язык фиксирует global-row mean reduction и
weighted-sum aggregation; они не повторяются в каждом документе objective.

## Diagnostics

Selection — принадлежащая Transformer policy на основе weighted direct losses.
Gradient interactions — optional observations и не изменяют gradients, weights,
compatibility objective или output layout. Их публичная projection определена
Training Telemetry Query v3.
