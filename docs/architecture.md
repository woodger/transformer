# Архитектура Transformer

> Тип: справочник. Текущие процессы, contracts и ownership данных.

Transformer — training system на стороне provider-а. Inventory владеет
предметной семантикой данных и поведением, ориентированным на браузер;
Transformer владеет generic tensor execution, durable jobs, generations
моделей, checkpoints, recovery и projection telemetry.

## Процессы

```text
app/main.py
├── app/service/bootstrap       Arrow Flight service
├── app/worker/bootstrap        one training or prediction attempt
└── app/admin/bootstrap         token, model, and migration commands
```

Command dispatcher ленивый: admin command не инициализирует service, Worker,
Torch, CUDA или runtime Flight. Локального пути исполнения fit/predict/stream
нет.

```text
app/contracts/semantic/v6          semantic target/objective language
app/contracts/flight/v23           public Flight workflow
app/contracts/model_catalog/v8     owner-scoped model discovery/detail
app/contracts/model_topology/v4    owner-scoped public model topology
app/contracts/training_telemetry/v4 owner-scoped telemetry report
app/contracts/target_head_diagnostics/v5 owner-scoped target head diagnostics
app/contracts/worker/v21           internal service-to-worker protocol
app/contracts/checkpoint/v13       internal checkpoint/recovery metadata
app/contracts/metrics/v12          internal epoch metrics/OpenSearch points
app/contracts/metrics/fit_run/v12  internal terminal run summary
```

## Сервис Flight

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

Domain владеет состояниями job, errors и правилами lifecycle. Application
владеет use cases и capability-oriented ports. Inbound Flight валидирует
публичные wire documents и представляет results/errors. Outbound adapters
владеют PostgreSQL, storage artifacts, supervision worker process, inventory
CUDA и OpenSearch.

Flight v23 принимает предоставленные вызывающей системой data binding, semantic
model intent, training intent и запрошенную initialization. Он выпускает
identity job и непрозрачный mutation lease. Разрешённые provider-ом values —
реализация модели, operational fences, schema fingerprints, версии
worker/checkpoint и детали artifacts — не пересекают публичную границу.

## Worker и tensor data plane

Один process Worker исполняет один принадлежащий service attempt. Он получает
immutable manifest Worker v21, записывает только artifacts workspace attempt и
возвращает bounded events. Он не зависит от PostgreSQL, Flight или public
identity.

Input Flight использует компактный `indexedFeatureBlocks`. Вызывающая система
передаёт упорядоченные dimensions blocks и local offsets. Worker выводит
positions blocks, отображает native rows и восстанавливает логические tensors
Float32 `[rows, seqLen, featureDim]` ограниченными slices до batching. Границы
физических payload/chunk не меняют логический порядок, training rows или target
coordinates.

Target identities Semantic v6 непрозрачны. Упорядоченные slots определяют
ширину target и prediction. Transformer интерпретирует generic transformations,
operators, typed roles и private resource classes, но никогда не имена target
внешней предметной области, profiles, FIGIs или feature formulas.

## Хранение и lifecycle

PostgreSQL — source of truth control plane для jobs, idempotency, owners,
attempts и metadata опубликованных моделей. Managed filesystem storage хранит
inputs, attempt artifacts, recovery artifacts, checkpoints моделей и временные
artifacts telemetry. OpenSearch — best-effort projection telemetry, а не source
registry или состояния job.

Model Catalog Query v8 читает owner-scoped state registry. Model Topology Query
v4 строит browser-safe статическую projection из metadata generation. Training
Telemetry Query v4 валидирует полную projection относительно принадлежащих
checkpoint-у metadata модели до раскрытия. Target Head Diagnostics Query v5
раскрывает opt-in наблюдения выходных головок и границы encoder, сохранённые
Worker после эпох.
Ни один query не позволяет вызывающей системе получить filesystem paths, bytes
checkpoint-а или topology OpenSearch.

Migration 0030 намеренно разрушительна: она отклоняет active jobs и удаляет
предыдущее состояние public boundary. Startup reconciliation удаляет managed
artifacts, оставшиеся без references после clean cut. Release procedure заменяет
индексы OpenSearch metrics v11 на v12 до запуска Flight v23.

## Правила зависимостей

Подробные правила направлений и размещения находятся в
[policy архитектуры](./policy/architecture.md). Историческое обоснование — в
[ADR](./adr/index.md). Публичные schemas определяются только активными
contract packages; этот документ не имеет над ними приоритета.
