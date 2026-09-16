# Transformer Arrow Flight service

Transformer is a Python service for durable PyTorch training and prediction
over authenticated Apache Arrow Flight jobs. Inventory is the browser-facing
Consumer; Transformer owns model execution, storage, checkpoints, recovery,
and telemetry projection.

## Current boundary

- Flight v15 is the only public job workflow.
- Semantic v3 carries ordered opaque targets, explicit objective bindings, and
  model tuning without Consumer-owned architecture literals.
- Model Catalog Query v3 and Training Telemetry Query v3 are owner-scoped
  read-only surfaces activated by Flight v15.
- Worker v14, checkpoint/recovery v8, and metrics v7 are provider-internal.
- `indexedFeatureBlocks` remains the compact Arrow input representation; its
  logical reconstruction semantics are unchanged.

There is no local fit/predict CLI and no compatibility reader for earlier
Flight, semantic, checkpoint, model-catalog, or telemetry revisions.

## Install and inspect

From the repository root:

```sh
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app/main.py --help
```

The project interpreter is `.venv/bin/python`. Do not install application
dependencies into the system Python or user site.

## Commands

```text
flight serve
auth tokens issue|list|revoke
models list|delete
db migrations status|apply|rollback
```

`flight serve` runs the remote service. The other commands are local
operational commands for database-backed access tokens, published generations,
and Alembic state. Exact options come from `transformer <command> --help`.

## Documentation

- [Flight v15](./app/contracts/flight/v15/README.md)
- [Semantic v3](./app/contracts/semantic/v3/README.md)
- [Model Catalog Query v3](./app/contracts/model_catalog/v3/README.md)
- [Training Telemetry Query v3](./app/contracts/training_telemetry/v3/README.md)
- [Worker v14](./app/contracts/worker/v14/README.md)
- [Checkpoint/recovery v8](./app/contracts/checkpoint/v8/README.md)
- [Architecture](./docs/architecture.md)
- [Consumer integration](./docs/consumer-flight-integration.md)
- [Flight operations](./docs/operations/flight-service.md)
- [Deployment](./docs/deployment/systemd.md)
- [Architecture decisions](./docs/adr/index.md)
