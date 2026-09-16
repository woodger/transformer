# Objective and loss semantics

> Type: Reference. Numerical semantics of the Semantic v3 objective language.

The normative document is [Semantic v3](../app/contracts/semantic/v3/README.md).
This reference explains the formulas implemented by Transformer; it does not
introduce target-specific behavior.

## Target coordinates

Every ordered opaque target slot declares a loss-input transformation and a
public-prediction transformation over one raw model coordinate. Supported
transformations are `Identity`, `Tanh`, and `Sigmoid`. `y` and public
predictions are finite Float32 at the Arrow boundary; intermediate tensor dtype
is Transformer-owned.

`observedConstraint` is omitted for finite values and explicit for a closed
interval. The direct operator is stated by the matching direct component and
is never inferred from target identity.

## Direct operators

For one target coordinate `T`:

```text
SmoothL1:
  mean(SmoothL1(lossEstimate[T], observed[T]))

BinaryCrossEntropyWithLogits:
  mean(BCEWithLogits(rawLogit[T], observedProbability[T]))

LogMSE:
  mean((log(positiveEstimate[T] + 1e-6)
        - log(nonNegativeObserved[T] + 1e-6))²)
```

`BinaryCrossEntropyWithLogits` requires identity loss input and observed values
inside `[0, 1]`. `LogMSE` requires a sigmoid loss estimate and a non-negative
closed observation interval. Every slot has exactly one direct component.

## Resources and auxiliary operators

`PositiveScalarPerObservation` is a private positive differentiable output.
It is checkpoint-owned, may be shared through its opaque resource identity,
and is not included in public prediction output. `GaussianNLL` supplies its
gradient-producing path. `RiskAdjustedExpectedValue` deliberately consumes the
scale through stop-gradient.

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

`ExpectedValue` and `RiskAdjustedExpectedValue` may coexist. Component weights
are positive. The language fixes global-row mean reduction and weighted-sum
aggregation; they are not repeated in each objective document.

## Diagnostics

Selection is a Transformer-owned policy based on weighted direct losses.
Gradient interactions are optional observations and do not alter gradients,
weights, objective compatibility, or output layout. Their public projection is
defined by Training Telemetry Query v3.
