# Owner-scoped Training Telemetry Query v3

> CONTRACT DOCUMENT. This package defines read-only telemetry for one
> published model generation. It is activated by Flight v15.

Telemetry describes the training pass that produced an epoch, not a repeat
evaluation of final checkpoint weights. For each batch, loss and target errors
are observed before its optimizer update and then aggregated for that epoch.
It is not model-test or out-of-sample quality data, and it is not evidence of
checkpoint integrity.

## Operations and owner scope

```text
transformer.training-telemetry.v3.report
transformer.training-telemetry.v3.gradient-interactions
```

Requests take an exact `modelRef`, never an owner or run identity. Transformer
resolves owner scope from authentication and obtains the authoritative
producing run. Unknown, foreign, and deleted models return the same
`MODEL_NOT_FOUND` result. A retained telemetry document cannot make a deleted
model visible.

Before returning a report, Transformer validates the registry-owned model
metadata, semantic identities, completion marker, epoch coverage, target and
component layout, numeric finiteness, and health-counter invariants. A partial
numeric report is never returned.

The outcome order is: model lookup/metadata validation; `pending` only while
materialization is demonstrably expected; `unavailable` when a complete report
will not arrive; telemetry-integrity failure when a completion marker conflicts
with observations; then `available`. Backend outage is an RPC error, not
`pending`.

## Report

The report header carries the model and run identities, data/model-definition
digests, and a single `layout` with target identities and direct/auxiliary
component identities. Epoch arrays then contain only numerical values in that
layout order. This avoids repeating semantic labels in every epoch.

An available report includes coverage, selection milestones, first/best/
published anchors, health totals, and one bounded page of epochs in ascending
order. `totalLoss`, `selectionScore`, and component means are authoritative
Worker observations that are accumulated independently. They are checked for
presence and finite representation but are not recomputed from one another
across languages.

`selectionScore` describes weighted direct losses when selection is enabled;
otherwise it is null. `bestEpoch` comes from checkpoint-owned selection state,
not a new argmin calculation over telemetry. With selection enabled, published
weights belong to the best epoch; otherwise they belong to the last epoch.

Each epoch's health counters satisfy:

```text
trainingBatchesCompleted
  = optimizerUpdatesApplied + optimizerUpdatesSkipped
  = finiteGradientBatches + nonFiniteGradientBatches
```

`ampOverflowBatches` is no greater than both skipped updates and non-finite
gradient batches. Health totals are exact component-wise sums over all epochs.

## Gradient interactions

Gradient interactions are fetched separately for one epoch. Components follow
objective execution order. A published pair has a unique orientation: its left
component precedes its right component in objective execution order. Self and
reversed pairs are absent. Pairs appear only when at least one finite cosine
observation exists; a zero-norm pair with no finite cosine observation is
therefore absent rather than represented by a sentinel. Pages sort pairs by
ASCII `(leftComponentIdentity, rightComponentIdentity)`.

The report says whether diagnostics are not configured, configured without
observations, or available. The default diagnostics epoch is published when
collected; otherwise the nearest collected epoch before it, then the earliest
one after it.

## Cursors, limits, and restart

Epoch pages allow at most 100 items; gradient pages allow at most 1,000 pairs.
Both use owner-bound, signed cursors with a 900-second TTL. A continuation
repeats the exact model, page size, and requested epoch where applicable.

Non-terminal pages retain an immutable validated snapshot in a shared bounded
process-local pool: at most 64 entries, 64 MiB total, and 16 MiB per snapshot.
Admission is atomic before a cursor is issued; unexpired entries are not
evicted. A terminal response requires no admission. If capacity cannot retain
a snapshot, the operation returns
`RESOURCE_EXHAUSTED / TELEMETRY_SNAPSHOT_CAPACITY_EXHAUSTED`.

Snapshots are intentionally process-local. A valid unexpired cursor from a
previous successful service start returns
`FAILED_PRECONDITION / TELEMETRY_CURSOR_INVALIDATED`; an expired cursor returns
`TELEMETRY_CURSOR_EXPIRED`. Model deletion has priority and returns
`MODEL_NOT_FOUND` even during traversal.

The response budget is 8 MiB. Error documents define structured invalid-query,
cursor, metadata, backend, integrity, capacity, and budget outcomes. Clients
branch on `code` and `reason`, not error text.

## Clean cut and fixtures

Revision 3 does not read v1/v2 telemetry. Flight v15 migration 0027 removes
old models and telemetry; new reports are produced only by new v15 fits and
the provider-owned metrics v7 projection.

`fixtures/` contains a compact pending/available report and lazy-gradient
examples. Its manifest hashes only the fixture bundle for offline review; it
does not enter telemetry, model, or checkpoint compatibility identity.
