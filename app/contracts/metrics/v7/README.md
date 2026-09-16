# Transformer training metrics v7

> CONTRACT DOCUMENT. This package defines the provider-internal epoch metrics
> artifact and its OpenSearch point projection for Semantic v3 runs.

`transformer.training-metrics.v7` records immutable epoch observations with
ordered opaque target/component references, loss values, target MAE/RMSE,
gradient diagnostics, and optimizer health counters. The v7 point projection
uses `transformer.metrics-point.v7` in `metrics-points-v7`.

These documents are provider telemetry storage, not a Consumer query contract.
Consumer receives normalized reports only through Training Telemetry Query v3;
it does not receive index names, document IDs, or OpenSearch mappings.

`opensearch/metrics-points-v7.template.json` is the required strict index
template. Deployment must install it before the index is created. Earlier
metrics indices are incompatible with the v15 clean cut and are removed by the
release procedure rather than read by v7 runtime.
