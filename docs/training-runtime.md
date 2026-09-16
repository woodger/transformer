# Training runtime and checkpoint

> Type: Reference. Provider-owned Worker v14 training, checkpoint v8, and
> recovery behavior behind Flight v15.

## Model definition and training

Flight v15 fit receives a Semantic v3 target/objective document and data
geometry. Transformer validates the closed language, materializes its internal
model configuration, and records D1 identities. Ordered opaque target slots
define output coordinates; no model/loss branch depends on Consumer target
names.

The Worker applies each slot's loss-input and public-prediction
transformations to the same raw output coordinate. Direct and auxiliary
operators, private resource classes, and gradient semantics are documented in
[losses](./losses.md) and Semantic v3. Training policy is separate from model
definition identity. Selection, when enabled, uses the weighted direct-loss
score; auxiliary components do not enter that score.

Feature values may contain NaN according to `missingValuePolicy`:

- `strict`: a timestep is masked when any feature is missing;
- `relaxed`: it is masked only when all features are missing, with missingness
  indicators supplied to the internal model.

Masking occurs before NaN-to-zero conversion. The model uses the last valid
timestep; an entirely masked sequence uses a safe zero placeholder.

## Input, epochs, and recovery

Worker reconstructs the compact Flight v15 `indexedFeatureBlocks` input into
bounded `[rows, seqLen, featureDim]` slices. Payload/chunk boundaries are not
optimizer batches, shuffle boundaries, or epoch boundaries. Durable fit may
start after input becomes available; close marks EOF and fixes the input
manifest for subsequent full epochs.

Recovery checkpoints are created only on completed global-epoch boundaries
after EOF. Checkpoint v8 stores model, optimizer, AMP scaler, RNG, shuffle,
selection, progress, semantic identities, resolved job configuration, and
input-manifest fences. Recovery verifies these before loading state. A
different data/model definition or manifest is rejected; no target remapping or
partial state loading is attempted.

Temporary attempt artifacts are service-managed. Startup reconciliation removes
only unreferenced managed artifacts under the service's own locked runtime
directory; it does not inspect or delete artifacts belonging to another
Transformer service instance.

## Telemetry

An epoch's telemetry is an observation of its training pass, before each
optimizer update. It includes objective values, training MAE/RMSE, health
counters, and optional gradient interactions. Its best-effort persistence does
not change optimizer execution, selection, fit success, or model publication.

OpenSearch receives a provider-owned v7 projection. Consumer retrieves a
validated, normalized report through Training Telemetry Query v3 rather than
directly from OpenSearch. A missing report does not invalidate a published
model.

## Clean cut

Checkpoint/recovery v8 has no reader for prior checkpoint state. Migration
0027 deletes old jobs and generations before Flight v15 activation; train new
generations after deployment.
