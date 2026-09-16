# Интеграция Consumer с Transformer Flight v15

> Тип: руководство. Практический порядок работы Consumer с действующей
> provider boundary. JSON Schema в `app/contracts/` имеет приоритет над этим
> пояснением.

## Preconditions

Consumer calls Transformer only with its own authenticated bearer credential.
The browser never calls Transformer or OpenSearch. Owner scope is derived from
the authenticated subject; Consumer does not send an owner, checkpoint path,
or provider storage identity.

Before integration, use the exact current packages:

- [Semantic v3](../app/contracts/semantic/v3/README.md)
- [Flight v15](../app/contracts/flight/v15/README.md)
- [Model Catalog Query v3](../app/contracts/model_catalog/v3/README.md)
- [Training Telemetry Query v3](../app/contracts/training_telemetry/v3/README.md)

There is no v14 compatibility path. Existing generations from before migration
0027 are deleted and cannot be used for predict or warm start.

## Fit

1. Materialize a self-contained Semantic v3 `ModelContract` from Consumer
   target/catalog/profile semantics. Do not send profile paths, target lookup
   keys, or executable loss code.
2. Build `dataBinding` from opaque `dataContractSha256`, tensor geometry, and
   `inputLayout.featureBlocks`.
3. Call `transformer.v15.fit.create` with a UUID `requestId`, stable
   `idempotencyKey`, label, requested device, model contract, training and
   diagnostics configuration, and requested initialization.
4. Persist the returned `jobId`, `mutationLease`, and resolved definition.
5. Upload ordered compact Arrow payloads using the returned descriptor path and
   the public DoPut metadata schema.
6. Calculate the receipt manifest digest, call `job.input.close`, then poll
   `job.status` until terminal state.

The server calculates actual physical receipts and totals. Do not resend
provider schema IDs, data digest, client execution identity, fencing values, or
close totals in upload metadata. `expectedLogicalRows` is optional and is only
for early EOF detection.

Fit may begin after durable input availability according to service scheduling;
input close marks EOF and completes the input manifest. Consumer must still
wait for a terminal job result before treating a model as published.

## Predict

Call `transformer.v15.predict.create` with an exact `modelRef`, requested
device, and current `dataBinding`. Do not resend TargetContract, Objective, or
model tuning. The create result supplies `predictionDefinition` before upload:
`seqLen`, output width, and ordered opaque targets with public prediction
transformations. Use it to construct and decode the Arrow data plane.

Data/model-definition mismatch is rejected before input upload. Prediction
coordinates always follow the checkpoint-owned target order; no implicit
remapping is performed.

## Mutation, status, and errors

`job.acquire` replaces a stale opaque mutation lease. `job.cancel` and
`job.input.close` require the current lease. `job.status` is intentionally a
mutable projection; Consumer keeps the immutable create response instead of
expecting a provider job-detail API.

Branch on structured error `code` and `reason`, never on message text.
`MODEL_NOT_FOUND` is security-equivalent for unknown, foreign, and deleted
models. Stored corruption, incompatible requests, unavailable dependencies,
and validation errors have distinct structured outcomes.

## Model catalog and telemetry

Use `transformer.model-catalog.v3.list` for owner-scoped discovery and
`.detail` for an exact selected `modelRef`. List is bounded high-water/keyset
traversal; a concurrent deletion can make detail return `MODEL_NOT_FOUND`, in
which case Consumer refreshes its view.

Use `transformer.training-telemetry.v3.report` for completed training-pass
observations and request gradient interactions only when the UI expands that
section. Telemetry absence does not hide or invalidate a published model.

Both query packages define their own cursor schemas and outcome rules. Their
availability is advertised by Flight v15 capabilities, but their semantics are
not embedded in generic job actions.
