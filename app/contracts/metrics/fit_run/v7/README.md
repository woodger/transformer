# Transformer terminal fit-run metrics v7

> CONTRACT DOCUMENT. This package defines the provider-internal terminal fit
> summary and its OpenSearch projection for Semantic v3 runs.

The immutable summary links a producing job and published model to its D1
identities, resolved initialization, training milestones, durations, input
counts, and physical artifact fences. It is the completion marker used by the
telemetry materializer; it is not a public model-registry record.

The v7 projection uses `transformer.metrics-fit-run.v7` in
`metrics-runs-v7`. `opensearch/metrics-runs-v7.template.json` must be installed
before index creation. Consumer reaches these observations only through
Training Telemetry Query v3, never by direct OpenSearch access.

Earlier fit-run documents are not read after the Flight v15 clean cut.
