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

app/contracts/flight/v5            публичный Flight contract
app/contracts/worker/v7            внутренний process contract
app/contracts/metrics/v3           epoch artifact и OpenSearch points
app/contracts/metrics/fit_run/v2   terminal fit summary
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
- outbound adapters реализуют PostgreSQL, artifact storage, worker process,
  CUDA inventory и OpenSearch boundaries.
- `service/bootstrap` собирает конкретные service adapters.

Application использует capability-oriented ports, включая
`JobLifecycleStore`, `JobQueryStore`, `InputUploadStore`, `OutputAccessStore`,
`ArtifactPublisher`, `ExecutionPlanBuilder`, `AttemptProcess`,
`WorkerCapabilities`, `DeviceLeaseManager` и `AccessTokenAuthenticator`.

## Worker

`app/worker/` владеет Arrow-to-tensor data path, model, training, telemetry,
device/reproducibility runtime и checkpoint staging. Один процесс обслуживает
один execution attempt. Он читает service-owned immutable command manifest,
пишет в attempt workspace и передаёт bounded NDJSON events через stdout, а
diagnostics — через stderr.

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
остаются в `app/contracts/worker/v7/config.py`.

`app/admin/cli` отвечает за presentation. `app/admin/bootstrap` создаёт
короткоживущие PostgreSQL resources для access-token и model use cases.
Alembic-команды имеют отдельный короткоживущий SQLAlchemy lifecycle.

## Contracts

- `app/contracts/flight/v5/` — нормативные schemas и fixtures публичного API;
- `app/contracts/ml.py` — единая Python identity target, ML-контракта,
  checkpoint и recovery formats;
- `app/contracts/worker/v7/` — command/result manifests, capability document,
  Arrow artifact manifests, events и exit semantics;
- `app/contracts/metrics/v3/` — immutable epoch artifact, OpenSearch
  projection, golden identity и index templates;
- `app/contracts/metrics/fit_run/v2/` — terminal fit summary, lifecycle
  durations и counters.

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

Filesystem artifacts проходят staged lifecycle до появления ссылки на них в
PostgreSQL. Prediction artifacts и attempt workspaces являются runtime-данными;
fit inputs и completed-epoch checkpoints обеспечивают recovery; опубликованные
model generations неизменяемы. Точные каталоги, failure semantics и процедуры
reconciliation задаёт
[операционное руководство Flight](./operations/flight-service.md).

Process-local memory хранит bounded queues, token verification cache и handles
активных процессов. Authentication model и границы согласованности описывает
[справочник аутентификации](./authentication.md), а observability boundary —
[политика metrics](./policy/metrics-policy.md).
