# Owner-scoped Model Catalog Query v3

> CONTRACT DOCUMENT. This package defines read-only discovery and detail of
> published Transformer model generations. It is activated by Flight v15.

The registry is the catalog source of truth. Telemetry is neither a source of
existence nor a condition for model visibility. Requests never carry an owner:
Transformer derives scope from the authenticated subject. Unknown, foreign,
and deleted `modelRef` values are security-equivalent `MODEL_NOT_FOUND`.

## Operations

```text
transformer.model-catalog.v3.list
transformer.model-catalog.v3.detail
```

List is a bounded owner-scoped traversal ordered by `createdAt DESC,
modelRef ASC`. It uses a signed, owner-bound keyset cursor with a 900-second
TTL and a live high-water boundary. A concurrent deletion may make a later
detail lookup return `MODEL_NOT_FOUND`; Consumer should remove that entry and
refresh. The cursor does not preserve access to a deleted generation.

Each list summary contains only data required to select and compare a
generation: immutable `modelRef`, label, generation, creation time, opaque
data/model-definition identities, resolved model tuning, ordered opaque target
identities, catalog initialization summary, and producing-run identity. It
does not expose checkpoint bytes/hashes, filesystem paths, Worker versions,
or full target/objective documents.

Detail takes one exact `modelRef`. It returns the summary plus:

- data digest and tensor geometry;
- the full semantic v3 `ModelContract` and all D1 layers;
- resolved training/diagnostics settings;
- terminal progress and selection summary; and
- catalog initialization summary (`random`, or `publishedModel` with its
  visible parent model reference).

Requested initialization, checkpoint-resolved parent hashes, checkpoint paths,
and artifact byte counts are provider-internal. The detail response is a
canonical description of the generation, not an artifact administration API.

## Verification and errors

List reads registry metadata only. Detail validates stored metadata and D1
before checking the managed checkpoint artifact. A malformed stored definition
returns `MODEL_CORRUPT / STORED_MODEL_METADATA_INVALID`; an invalid artifact
returns `MODEL_CORRUPT / MODEL_CHECKPOINT_INVALID`; a verification work-budget
limit returns `RESOURCE_EXHAUSTED / MODEL_VERIFICATION_UNAVAILABLE`.

Other structured outcomes include invalid/expired cursor, invalid request,
registry unavailability, and response-budget exhaustion. Error documents are
defined by `schemas/error-detail.schema.json`; clients branch on `code` and
`reason`, never on human-readable text.

Maximum page size is 100, cursor TTL is 900 seconds, and a serialized result
is limited to 8 MiB. Detail verifies at most one checkpoint and refuses a
verification larger than 1 GiB before reading it.

## Clean cut and fixtures

Revision 3 has no reader for catalog v1/v2 entries. Flight v15 migration 0027
removes previous generations and their registry records, so deployment starts
the catalog empty and new models are trained under semantic v3.

`fixtures/` contains empty/list/detail examples for offline cross-project
review. Its manifest hashes only that fixture bundle and has no runtime or
compatibility meaning.
