# Transformer checkpoint and recovery v8

> CONTRACT DOCUMENT. This package defines provider-internal checkpoint and
> recovery metadata for Semantic v3, Flight v15, and Worker v14.

`transformer-checkpoint-v8` embeds validated semantic v3 model intent,
Transformer-resolved model configuration, D1 identities, training and
diagnostics configuration, selection state, resolved initialization, job
configuration fence, input-manifest fence, and progress. Tensor parameters,
optimizer/scaler state, RNG state, checkpoint path, and physical artifact hash
remain implementation data.

Requested initialization and resolved initialization are different documents:

- Flight accepts `{ "source": "random" }` or
  `{ "source": "publishedModel", "modelRef": ... }` as requested intent.
- checkpoint metadata records the resolved source; a published-model source
  also records parent model/checkpoint and exact parent/current compatibility
  identities.
- Model Catalog exposes only the short catalog summary: random, or the visible
  parent model reference.

`transformer-recovery-v8` binds a recovery artifact to exact job, input
revision, semantic identities, resolved job configuration, input manifest, and
progress. Before loading tensors, Transformer verifies the managed artifact and
all metadata fences. Incompatibility is not repaired by target remapping or
partial state loading.

Checkpoint/recovery v8 has no v6/v7 reader. Flight v15 migration 0027 removes
old jobs and generations before v8 becomes active; new models are trained from
the new public boundary.
