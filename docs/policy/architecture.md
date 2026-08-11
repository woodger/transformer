# Архитектурная политика

> Type: Policy. Документ фиксирует process boundaries, направление зависимостей
> и ownership данных Transformer Arrow Flight service.

Проект использует Clean Architecture отдельно для каждого исполняемого
процесса. Нормативное решение и его причины зафиксированы в
[ADR 0004](../adr/0004-clean-architecture-process-boundaries.md).

## Процессы и composition roots

```text
app/main.py                         ленивый CLI dispatcher
├── app/service/bootstrap          Arrow Flight service
├── app/worker/bootstrap           один ML execution attempt
└── app/admin/bootstrap            auth и database commands

app/contracts/flight/v3            публичный Flight contract
app/contracts/worker/v2            внутренний process contract
```

Единого bootstrap, импортирующего весь проект, нет. Service запускает worker
как установленный executable по версионированному process contract и не
импортирует его Python implementation. Admin-команды не загружают Flight
server, Arrow/Torch worker runtime или модель.

## Service

```text
service/adapters/inbound/flight
              │
              ▼
service/application/{commands,queries,services,ports}
              │
              ▼
service/domain

service/bootstrap ── собирает inbound и outbound adapters
service/adapters/outbound/{postgres,artifact_storage,worker_process,worker_probe}
```

- `service/domain` содержит job states, error codes, immutable records и pure
  lifecycle policies. Он не знает о Flight, SQLAlchemy, PyArrow, filesystem,
  subprocess и Torch.
- `service/application` содержит use cases, scheduler orchestration и
  capability-oriented ports. Он зависит только от domain и нейтральных
  contracts.
- inbound Flight adapter валидирует wire DTO, выполняет mapping и преобразует
  application errors в Flight/Arrow status.
- outbound adapters реализуют PostgreSQL, artifact storage, worker process и
  worker capability boundaries. Inbound и outbound adapters не импортируют
  друг друга.
- `service/bootstrap` — единственное место сборки конкретных service adapters.

Ports называются по возможностям: `JobRepository`, `ArtifactPublisher`,
`ExecutionPlanBuilder`, `AttemptProcess`, `WorkerExecutor`,
`WorkerCapabilities`, `DeviceLeaseManager`. Имена технологий в application
ports и generic `Repository[T]` не допускаются. Каждый port имеет текущего
runtime consumer и adapter; интерфейсы «на будущее» не создаются.

## Worker

`app/worker/` владеет Arrow-to-tensor data path, model, training, metrics,
device/reproducibility runtime и checkpoint staging. Один процесс обслуживает
ровно один execution attempt. Worker:

- читает только service-owned immutable command manifest;
- пишет только в attempt workspace;
- отправляет bounded NDJSON events в stdout и diagnostics в stderr;
- не подключается к PostgreSQL и не знает о Flight, bearer auth, idempotency,
  public job state или `modelRef`;
- не публикует output, recovery generation или model generation.

`app/data`, `app/model`, `app/training`, `app/storage`, `app/runtime` и
`app/metrics` временно сохраняются только как compatibility import facades.
Новый production-код размещается непосредственно в `app/worker/`.

## Admin

`app/admin/cli` отвечает только за presentation. `app/admin/bootstrap`
создаёт короткоживущие PostgreSQL resources и вызывает те же application use
cases, которые определяют операции с access tokens. Alembic-команды имеют
отдельный короткоживущий SQLAlchemy lifecycle и не запускают service или worker.

## Contracts

- `app/contracts/flight/v3/` — нормативные schemas и fixtures публичного API;
- Flight v3 является текущей штатной архитектурой remote API; дальнейшие
  изменения проектируются от его lifecycle, durability и fencing semantics;
- `app/contracts/worker/v2/` — command/result manifests, capability document,
  Arrow artifact manifests, events и exit semantics;
- эти contracts версионируются независимо;
- worker `attemptId` — UUID execution identity и equality fence; публичный
  `attempt` остаётся положительным job-local ordinal;
- ownership никогда не передаётся при неизменном `attemptId`. Retry создаёт
  новый ordinal и новый `attemptId`.

Любая worker mutation атомарно проверяет `jobId`, текущий `attemptId` и
разрешённое non-terminal state. Отдельный случайный fencing token не добавляется
до появления lifecycle, где ownership меняется внутри одного attempt.

## PostgreSQL и artifacts

PostgreSQL adapter, ORM и Alembic находятся в
`app/service/adapters/outbound/postgres/`. PostgreSQL является единственным
источником истины для job lifecycle, revision, idempotency, active attempt,
tokens и published metadata. SQLite и dual-write запрещены.

PostgreSQL-транзакция не охватывает filesystem или subprocess. Artifact
lifecycle всегда staged:

```text
temporary write → close/fsync → atomic rename → validation
→ immutable publication/fsync → PostgreSQL reference/state transaction
```

После crash допустим непривязанный staged artifact, но не запись БД на
недописанный файл. Orphans удаляет reconciliation.

Ownership хранения:

- `/tmp/transformer` — prediction inputs/outputs, attempt workspaces, runtime
  epoch и boot-scoped CUDA quarantine;
- `recovery/` — persistent fit inputs и completed-global-epoch checkpoints;
- `models/` — только успешно опубликованные immutable model generations;
- RAM — FIFO queues, token digest cache и active process handles.

## Обязательные dependency rules

- domain не зависит от application, adapters или bootstrap;
- application не зависит от adapters или bootstrap;
- adapters зависят от application/domain, но не от другого направления
  transport-а;
- service не импортирует `app.worker` implementation;
- worker не импортирует service, Flight или database implementation;
- admin не импортирует worker или Flight server;
- shared service/worker данные находятся только в `app/contracts/worker/v2`;
- import graph не содержит циклов;
- environment, connections, CUDA initialization и filesystem mutation не
  выполняются при import.

Правила закреплены AST- и process-import тестами в
`tests/test_architecture_boundaries.py`.

## Размещение нового кода

- lifecycle rule или record — `app/service/domain/`;
- command/query и capability port — `app/service/application/`;
- Flight parser/presenter — `app/service/adapters/inbound/flight/`;
- ORM/repository/Alembic — `app/service/adapters/outbound/postgres/`;
- spool/publication — `app/service/adapters/outbound/artifact_storage/`;
- subprocess supervision — `app/service/adapters/outbound/worker_process/`;
- model/loss/trainer/Arrow tensor/checkpoint — профильный пакет в `app/worker/`;
- wire/process schema — соответствующий versioned package в `app/contracts/`;
- runtime wiring — composition root конкретного процесса.

Новые `common`, `helpers`, `lib` и `misc` без одного ясного owner не создаются.
Compatibility facade не становится владельцем новой логики.
