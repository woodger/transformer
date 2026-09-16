# Transformer Arrow Flight v15

> CONTRACT DOCUMENT. This directory defines the public, consumer-neutral Arrow
> Flight boundary. It is the only supported Flight workflow revision.

Flight v15 is a destructive clean cut. It has no v14 actions, aliases, or
reader. The public boundary is intentionally smaller: Consumer provides model
intent and data geometry; Transformer issues job identity, owns execution and
artifact details, and exposes only the information needed to upload, wait, and
consume results.

## Actions

```text
transformer.v15.capabilities
transformer.v15.health
transformer.v15.fit.create
transformer.v15.predict.create
transformer.v15.job.acquire
transformer.v15.job.status
transformer.v15.job.inputs.list
transformer.v15.job.input.close
transformer.v15.job.outputs.list
transformer.v15.job.cancel
transformer.model-catalog.v3.list
transformer.model-catalog.v3.detail
transformer.training-telemetry.v3.report
transformer.training-telemetry.v3.gradient-interactions
```

All Flight job requests carry a UUID `requestId`; create and mutation requests
also carry a Consumer-issued `idempotencyKey`. Fit and predict have separate
closed create schemas. A successful create returns a Transformer-issued
`jobId` and one opaque `mutationLease`, which are required for later mutation
actions and DoPut metadata. No client execution ID, fence counter, schema ID,
or runtime worker identity is supplied by Consumer.

`fit.create` receives a model label, requested device, `dataBinding`, semantic
v3 `modelContract`, training configuration, diagnostics, and requested
initialization (`source: random` or `source: publishedModel` with `modelRef`).
`predict.create` receives only an exact `modelRef`, requested device, and
`dataBinding`; it does not resend a model contract. Its result contains the
checkpoint-owned prediction definition: sequence length, target output width,
and ordered opaque target identities with their public transformations.

`dataBinding` contains opaque `dataContractSha256`, `tensorGeometry`, and
`inputLayout`. It contains no readable profile, data revision, feature
catalogue, or Consumer semantic metadata.

## Input lifecycle

`inputLayout.featureBlocks` declares each block by `windowRows` and
`nativeRowWidth`. Positions are derived by Transformer and all block widths
must sum to `featureDim`. `indexedFeatureBlocks` is the only v15 input layout;
there is no source-encoding discriminator.

Each DoPut is one physical payload. Upload metadata names only the issued job,
opaque lease, payload identity, ordinal, logical-row count, chunks, and native
row counts. Transformer derives and durably records physical schema identity,
schema fingerprint, receipts, and actual totals. `job.input.close` receives the
receipt-derived `manifestSha256` and may carry `expectedLogicalRows` solely for
early EOF detection; it does not repeat client-calculated totals.

The logical reconstruction is unchanged. For each block, Transformer resolves
local observation offsets over native rows, flattens the configured window,
and concatenates blocks into `[rows, seqLen, featureDim]`. Fit additionally
accepts finite target vectors whose width is derived from the ordered slots.
Predict output is a target-aligned finite Float32 vector with that same width.
This contract preserves logical tensor reconstruction, not a promise about
serialized Arrow IPC bytes.

## Execution and artifacts

`job.status` provides lifecycle state, input progress, selected device,
terminal error/result projection, and polling hint. It does not expose Worker
protocol versions, checkpoint byte counts or hashes, filesystem paths,
recovery descriptors, worker logs, or internal fencing. `job.outputs.list`
provides bounded output traversal for predict.

Transformer validates the semantic v3 document and computes its D1 identities
at create time. Fit and warm-start compatibility are exact at the data and
model-definition layers. Stored corruption, invalid request values, and
incompatible definitions use distinct structured error reasons. Physical
checkpoint validation remains a provider responsibility.

`capabilities` advertises available devices, upload limits, and availability
of the catalog and telemetry query surfaces. It does not publish architecture
literals, Worker/checkpoint versions, primitive catalogues, or storage
topology.

## Related query contracts

Model discovery/detail is defined by
[`model_catalog/v3`](../../model_catalog/v3/README.md). Training telemetry is
defined by [`training_telemetry/v3`](../../training_telemetry/v3/README.md).
Both are activated by the actions above but retain their own revision and
cursor semantics.

## Clean cut

Migration `0027_public_contract_simplification` refuses to run while a job is
non-terminal, then removes jobs, models, recovery records, and database-backed
telemetry that cannot be read by v15. Startup reconciliation removes their
unreferenced managed filesystem artifacts. OpenSearch v6 metric indices must
be removed through the release procedure and recreated from the v7 templates.
Existing generations are not available for predict, warm start, recovery, or
catalog queries; train new generations after deployment.

## Fixtures

`fixtures/` covers the active layout, initialization, and resolved job-config
forms. Its manifest hashes only the fixture bundle for offline cross-project
review. It is not part of runtime dispatch, request validation, or model
compatibility.
