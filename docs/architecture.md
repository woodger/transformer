# Архитектура Transformer

> Тип: справочник. Текущие процессы, компоненты, contracts и ownership данных
> Transformer.

Этот документ описывает, как система устроена сейчас. Обязательные правила
направления зависимостей и размещения нового кода задаёт
[архитектурная политика](./policy/architecture.md). Историческое обоснование
границ хранится в [ADR](./adr/index.md), а wire- и process-форматы — в
соответствующих каталогах `app/contracts/`.

## Процессы и composition roots

```text
app/main.py                         ленивый CLI dispatcher
├── app/local                      локальные file/stream commands и gmark
├── app/service/bootstrap          Arrow Flight service
├── app/worker/bootstrap           один ML execution attempt
└── app/admin/bootstrap            auth, database и model commands

app/contracts/semantic/v1               consumer-neutral ML language
app/contracts/flight/v13                 публичный Flight contract
app/contracts/model_catalog/v1           owner-scoped model query contract
app/contracts/training_telemetry/v1      owner-scoped telemetry query contract
app/contracts/worker/v12                внутренний process contract
app/contracts/checkpoint/v6             checkpoint/recovery metadata
app/contracts/indexed_feature_blocks.py  общие compact-input invariants
app/contracts/metrics/v5                epoch artifact и OpenSearch points
app/contracts/metrics/fit_run/v5        terminal fit summary и lineage
```

Каждый исполняемый процесс имеет собственный composition root. Service
запускает worker как установленный executable по версионированному process
contract. Admin-команды создают только необходимые им короткоживущие
resources. Ленивый CLI dispatcher не загружает runtime остальных процессов.

## Flight service

```text
service/adapters/inbound/flight
              │
              ▼
service/application/{commands,queries,services,ports,telemetry}
              │
              ▼
service/domain

service/bootstrap ── собирает inbound и outbound adapters
service/adapters/outbound/{postgres,artifacts,worker,cuda,opensearch}
```

- `service/domain` содержит job states, error codes, immutable records и pure
  lifecycle policies.
- `service/application` содержит commands, queries, нейтральные results,
  scheduler orchestration и capability-oriented ports.
- `service/application/telemetry` содержит best-effort records и доставку
  наблюдений отдельно от job ledger.
- inbound Flight adapter валидирует wire DTO, выполняет semantic mapping и
  преобразует application results и errors в Flight documents и Arrow status.
- Public Flight и local fit/predict CLI используют device class `gpu`;
  inbound/local adapters преобразуют его во внутренний CUDA runtime, а наружу
  не публикуют backend-specific device identity.
- outbound adapters реализуют PostgreSQL, artifact storage, worker process,
  CUDA inventory и OpenSearch boundaries.
- `service/bootstrap` собирает конкретные service adapters.

Application использует capability-oriented ports, включая
`JobLifecycleStore`, `JobQueryStore`, `InputUploadStore`, `OutputAccessStore`,
`ArtifactPublisher`, `ExecutionPlanBuilder`, `AttemptProcess`,
`WorkerCapabilities`, `DeviceLeaseManager`, `ModelCatalogStore` и
`TrainingTelemetrySource`, и `AccessTokenAuthenticator`.

## Worker

`app/worker/` владеет Arrow-to-tensor data path, model, training, telemetry,
device/reproducibility runtime и checkpoint staging. Один процесс обслуживает
один execution attempt. Он читает service-owned immutable command manifest,
пишет в attempt workspace и передаёт bounded NDJSON events через stdout, а
diagnostics — через stderr.

Flight input хранится как compact indexed feature blocks. Worker memory-map-ит
immutable artifacts и восстанавливает dense `[rows, seqLen, featureDim]`
ограниченными срезами непосредственно перед batching. Полный развёрнутый
dataset не материализуется; physical chunk/payload boundaries не меняют
логический порядок, bounded shuffle или optimizer batches.

Worker не подключается к PostgreSQL и не управляет Flight identity, public job
state, artifact publication, recovery generation или model generation. Его
внутренний executor напрямую использует PyArrow, Torch и filesystem как части
одного runtime.

Канонический ML-код находится в `app/worker/`. Checkpoints принадлежат
`app/worker/checkpoints/`, подготовка batch —
`app/worker/training/batching.py`, core результат global epoch —
`app/worker/training/epoch.py`, а необязательные наблюдения —
`app/worker/telemetry/`. Подробную training semantics описывает
[training reference](./training-runtime.md).

## Local CLI и admin

`app/local/` владеет локальными file/stream командами, `plot-metrics` и
`gmark`. Этот путь выполняется в локальном процессе и может напрямую
использовать worker-код.

Общие identity и путь корня проекта находятся в `app/project.py`, встроенные
operational defaults — в `app/config.py`. Runtime-владельцы сохраняют
configuration types, загрузку и валидацию. Версионируемые worker defaults
остаются в `app/contracts/worker/v12/config.py`.

`app/config.py` не является adapter или provider boundary. Его immutable
defaults могут использовать разные процессы, а понятия Transformer остаются в
domain либо внутренних contracts; adapters преобразуют их в типы конкретной
runtime library. Сейчас `RUNTIME_DIR_DEFAULT` является отдельным
platform-derived значением: оно вычисляется через `tempfile.gettempdir()` и
потребляется bootstrap, а не service domain/application.

`app/admin/cli` отвечает за presentation. `app/admin/bootstrap` создаёт
короткоживущие PostgreSQL resources для access-token и model use cases.
Alembic-команды имеют отдельный короткоживущий SQLAlchemy lifecycle.

## Contracts

- `app/contracts/semantic/v1/` — закрытый mathematical language, canonical
  TargetContract/Objective/ModelContract и D1 digests;
- `app/contracts/flight/v13/` — нормативные schemas и fixtures публичного API;
- `app/contracts/model_catalog/v1/` — independently versioned list/detail
  contract owner-scoped model registry;
- `app/contracts/training_telemetry/v1/` — independently versioned complete
  report и lazy gradient-interaction query для published generation;
- `app/contracts/indexed_feature_blocks.py` — общая pure-валидация ordered
  block geometry между service и worker без зависимости от Flight adapter;
- `app/contracts/worker/v12/` — command/result manifests, capability document,
  Arrow artifact manifests, events и exit semantics;
- `app/contracts/checkpoint/v6/` — embedded checkpoint metadata и recovery
  fences;
- `app/contracts/metrics/v5/` — immutable epoch artifact, OpenSearch
  projection, golden identity и index templates;
- `app/contracts/metrics/fit_run/v5/` — terminal fit summary, initialization
  lineage, lifecycle durations и counters.

Consumer materializer передаёт self-contained ModelContract с ordered opaque
target slots, numerical constraints, loss/public transformations, Objective и
model configuration. Transformer не интерпретирует target identities, profile
или data identity. Он валидирует generic geometry и typed bindings, вычисляет
D1 digests и владеет tensor-семантикой operators, private resource kinds,
model heads и autograd. Ordered slots физически задают ширину `tgt`, public
heads и prediction output. Objective, training policy и diagnostics остаются
разными contract sections; checkpoint навсегда связан с data, target,
objective и model digest layers.

`sourceEncoding` описывает только физическое compact-представление
Consumer-owned features. Оно входит в immutable job configuration и recovery
fencing, но не в data-contract, objective, checkpoint или model identity при
численно эквивалентном logical tensor.

Fit initialization также является отдельной частью job identity.
`publishedModel` требует точного совпадения data, target, objective и model
digests. Service разрешает owner-scoped immutable parent
`modelRef` и передаёт worker-у проверенный checkpoint artifact. Worker повторно
проверяет digest, загружает полный parent `state_dict` и создаёт новое training
state. Lineage с parent reference, checkpoint digest и совпадающими
parent/current data-contract digests сохраняется в job, checkpoint, model
metadata и terminal fit telemetry.

Эти contracts версионируются независимо. Worker `attemptId` является UUID
execution identity и equality fence; публичный `attempt` — положительный
job-local ordinal. Retry создаёт новый ordinal и новый `attemptId`. Любая
worker mutation проверяет `jobId`, текущий `attemptId` и допустимое
non-terminal state; ownership не передаётся при неизменном `attemptId`.

## Данные и хранение

PostgreSQL является источником истины для job lifecycle, revision,
idempotency, active attempt, API access tokens, owner-scoped state и published
model metadata. Отдельный telemetry slice владеет epoch intervals, run artifact
metadata и metrics outbox. OpenSearch — best-effort аналитическая проекция.

Owner-scoped Model Catalog читает только `AVAILABLE` generations из registry.
List использует bounded keyset pagination с publication high-water boundary и
подписанным owner-bound cursor; checkpoint bytes при этом не читаются. Single
detail проверяет canonical metadata и полный SHA-256 checkpoint в пределах
contract budget. Unknown, foreign и deleted references неразличимы для
Consumer-а. Catalog membership не зависит от OpenSearch telemetry.

Owner-scoped Training Telemetry Query сначала разрешает exact
`modelRef` через registry, затем читает metrics v5 проекцию через
application port. Он возвращает только полный проверенный report;
отсутствие или повреждение telemetry не меняет model registry и не
влияет на fit, predict, warm start и model lifecycle.

Filesystem artifacts проходят staged lifecycle до появления ссылки на них в
PostgreSQL. Prediction artifacts и attempt workspaces являются runtime-данными;
compact fit inputs и completed-epoch checkpoints обеспечивают recovery;
опубликованные model directories содержат только неизменяемый binary
checkpoint, а отдельная control-plane metadata хранится только в PostgreSQL.
Checkpoint сохраняет собственную process-contract metadata для проверки
worker-ом. Незавершённый warm-start fit удерживает parent generation от явного
удаления. Точные каталоги, failure semantics и процедуры reconciliation задаёт
[операционное руководство Flight](./operations/flight-service.md).

Process-local memory хранит bounded queues, token verification cache и handles
активных процессов. Authentication model и границы согласованности описывает
[справочник аутентификации](./authentication.md), а observability boundary —
[политика metrics](./policy/metrics-policy.md).
