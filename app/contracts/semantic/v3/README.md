# Consumer-neutral semantic model v3

> CONTRACT DOCUMENT. This directory defines the closed language for target
> slots, objectives, model tuning, and D1 semantic identities used by Flight
> v15 and the provider-owned Worker v14/checkpoint v8 runtime.

Schemas are the authoritative document shape. This file defines the semantic
rules that cannot be expressed by JSON Schema. Revision 3 is a clean cut: no
v1 or v2 document, digest, or compatibility reader is accepted.

## Boundary

`ModelContract` contains only Consumer-owned training intent:

```json
{
  "targetContract": {"slots": []},
  "objective": {"directComponents": []},
  "modelTuning": {}
}
```

The Consumer supplies an opaque data digest and tensor geometry in Flight
`dataBinding`. Transformer resolves its model definition from that geometry,
the closed language revision, target layout, objective, and tuning. It issues
the resulting model-definition digest. Architecture implementation, private
heads, parameterization, tensor layout, checkpoint bytes, and device policy
remain Transformer-owned.

Target identities are opaque. Transformer must not branch on a target name,
profile, FIGI, or other Consumer-owned semantic value.

## Targets and objective

`TargetContract.slots` is a non-empty ordered array of unique identities.
Its order is the physical target-vector order and determines output width.
Each slot has a loss-input transformation and a public-prediction
transformation. `observedConstraint` is optional: omission means finite
observations; an explicit closed interval has this form:

```json
{
  "identity": "ConsumerDefined.Probability",
  "observedConstraint": {
    "constraint": "ClosedInterval",
    "minimum": 0,
    "maximum": 1
  },
  "lossInputTransformation": "Identity",
  "publicPredictionTransformation": "Sigmoid"
}
```

The supported transformations are `Identity`, `Tanh`, and `Sigmoid`.
`ClosedInterval` is the only explicit constraint primitive. All JSON numbers
are finite IEEE-754 binary64 values. Arrow `y` values and public predictions
are finite Float32; Transformer-owned internal tensor dtype is not prescribed.

Each target slot has exactly one direct component in the same order as slots.
Direct components state `identity`, `operator`, `weight`, and
`targetIdentity`. Auxiliary components state their operator-specific named
roles. Target roles and resource roles are therefore structurally distinct
without a generic reference discriminator. Resources declare an opaque
identity and `resourceClass`:

```json
{"identity":"sharedScale","resourceClass":"PositiveScalarPerObservation"}
```

`PositiveScalarPerObservation` is a private, positive, differentiable scalar
per observation. It is checkpoint-owned, can be shared by repeated resource
identity, and is never a public prediction coordinate. Transformer owns its
implementation. A declared resource must be used and must have a structural
gradient-producing path to total loss; a stop-gradient-only use is invalid.

Direct operators are `SmoothL1`, `BinaryCrossEntropyWithLogits`, and `LogMSE`.
Auxiliary operators are `GaussianNLL`, `ExpectedValue`, and
`RiskAdjustedExpectedValue`. Their formulas, role constraints, constants, and
gradient semantics are implemented and validated by Transformer. Direct
operator selection is explicit in the document; it is never inferred by
Transformer from a target identity. `ExpectedValue` and
`RiskAdjustedExpectedValue` may coexist.

Components have strictly positive weights. Direct components are in target-slot
order; resources and auxiliary components are in ASCII identity order. The
language fixes `GlobalRowMean` reduction and `WeightedSum` aggregation, so
those redundant fields are not carried in each document.

## D1 identities

After structural and semantic validation, Transformer calculates SHA-256 over
RFC 8785/JCS UTF-8 preimages:

```text
targetContractSha256 = SHA256(JCS({
  objectiveLanguageRevision: 3,
  targetContract
}))

objectiveSha256 = SHA256(JCS({
  objectiveLanguageRevision: 3,
  objective
}))

```

`dataContractSha256` remains Consumer-issued and opaque to Transformer. The
full documents are the source of meaning; digests are derived identities.
`modelDefinitionSha256` is instead provider-issued after Transformer resolves
its internal model implementation and `ModelConfig`. It binds the target and
objective digests, but its preimage is not a Consumer contract and is not
recomputed across the boundary. There is no representation-independent hashing
and no conversion of older semantic revisions to v3. Input manifests, job
configuration hashes, and physical checkpoint hashes are separate operational
fences rather than D1 layers.

## Validation and capabilities

Validation order is: closed schema; finite JSON numbers; target layout and
constraints; component/resource ordering and uniqueness; direct coverage;
operator roles/domains; resource reachability; model-tuning validity; then D1
calculation. Incompatible stored or requested definitions are not remapped.

`language-capabilities.schema.json` declares revision 3, the closed language,
and semantic limits. A Consumer may introduce a new opaque target identity
using advertised primitives without causing target-name branching in
Transformer. Changing a primitive formula, type system, role model, or
resource lifecycle requires a new revision.

## Fixtures

`fixtures/` contains a small cross-project behavioral bundle: single-target,
multi-target shared-resource, new opaque target, and reordered-layout cases.
The fixture manifest hashes only files in that bundle; it is an offline review
aid, not a runtime input, capability, or compatibility fence.
