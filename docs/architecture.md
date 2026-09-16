# Архитектура Transformer

> Тип: справочник. Текущие процессы, contracts и ownership данных.

Transformer is a provider-side training system. Inventory owns Consumer data
semantics and browser-facing behavior; Transformer owns generic tensor
execution, durable jobs, model generations, checkpoints, recovery, and
telemetry projection.

## Processes

```text
app/main.py
├── app/service/bootstrap       Arrow Flight service
├── app/worker/bootstrap        one training or prediction attempt
└── app/admin/bootstrap         token, model, and migration commands
```

The command dispatcher is lazy: an admin command does not initialize the
service, Worker, Torch, CUDA, or Flight runtime. There is no local
fit/predict/stream execution path.

```text
app/contracts/semantic/v3          consumer-neutral target/objective language
app/contracts/flight/v15           public Flight workflow
app/contracts/model_catalog/v3     owner-scoped model discovery/detail
app/contracts/training_telemetry/v3 owner-scoped telemetry report
app/contracts/worker/v14           internal service-to-worker protocol
app/contracts/checkpoint/v8        internal checkpoint/recovery metadata
app/contracts/metrics/v7           internal epoch metrics/OpenSearch points
app/contracts/metrics/fit_run/v7   internal terminal run summary
```

## Flight service

```text
service/adapters/inbound/flight
              │
              ▼
service/application/{commands,queries,services,ports,telemetry}
              │
              ▼
service/domain

service/bootstrap ── assembles adapters and resources
service/adapters/outbound/{postgres,artifacts,worker,cuda,opensearch}
```

The domain owns job states, errors, and lifecycle rules. Application owns use
cases and capability-oriented ports. Inbound Flight validates public wire
documents and presents results/errors. Outbound adapters own PostgreSQL,
artifact storage, worker process supervision, CUDA inventory, and OpenSearch.

Flight v15 accepts Consumer-owned data binding, semantic model intent,
training intent, and requested initialization. It issues job identity and an
opaque mutation lease. Provider resolved values—model implementation,
operational fences, schema fingerprints, worker/checkpoint versions, and
artifact details—do not cross the public boundary.

## Worker and tensor data plane

One Worker process executes one service-owned attempt. It receives an immutable
Worker v14 manifest, writes only attempt workspace artifacts, and returns
bounded events. It has no PostgreSQL, Flight, or public identity dependency.

Flight input uses compact `indexedFeatureBlocks`. Consumer supplies ordered
block dimensions and local offsets. Worker derives block positions, maps
native rows, and reconstructs logical `[rows, seqLen, featureDim]` Float32
tensors in bounded slices before batching. Physical payload/chunk boundaries
do not change logical order, training rows, or target coordinates.

Semantic v3 target identities are opaque. Ordered slots determine target and
prediction width. Transformer interprets generic transformations, operators,
typed roles, and private resource classes, but never Consumer target names,
profiles, FIGIs, or feature formulas.

## Storage and lifecycle

PostgreSQL is the control-plane source of truth for jobs, idempotency, owners,
attempts, and published-model metadata. Managed filesystem storage holds inputs,
attempt artifacts, recovery artifacts, model checkpoints, and temporary
telemetry artifacts. OpenSearch is a best-effort telemetry projection, not a
registry or job state source.

Model Catalog Query v3 reads owner-scoped registry state. Training Telemetry
Query v3 validates a complete projection against checkpoint-owned model
metadata before exposing it. Neither query lets Consumer access filesystem
paths, checkpoint bytes, or OpenSearch topology.

Migration 0027 is deliberately destructive: it refuses active jobs and removes
prior public-boundary state. Startup reconciliation removes managed artifacts
left unreferenced by the clean cut. The OpenSearch v6-to-v7 index replacement
is an explicit release operation.

## Dependency rules

Detailed direction and placement rules live in
[architecture policy](./policy/architecture.md). Historical rationale lives in
[ADRs](./adr/index.md). Public schemas are defined only in the active contract
packages; this document does not override them.
