# Архитектурная политика

> Type: Policy. Документ фиксирует process boundaries, направление зависимостей
> и ownership данных Transformer Arrow Flight service.

Проект использует Clean Architecture отдельно для каждого исполняемого
процесса. Этот документ является источником текущих process, dependency и data
ownership boundaries. Нормативный remote protocol находится в
[`app/contracts/flight/v5`](../../app/contracts/flight/v5/README.md),
training telemetry policy — в [`docs/metrics.md`](../metrics.md), а credential
model и security boundary — в
[`docs/authentication.md`](../authentication.md).
Историческое обоснование service и process boundaries находится в
[ADR 0001](../adr/0001-arrow-flight-job-service.md) и
[ADR 0004](../adr/0004-clean-architecture-process-boundaries.md); ADR не
переопределяет правила этого документа.

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

Единого bootstrap, импортирующего весь проект, нет. Service запускает worker
как установленный executable по версионированному process contract и не
импортирует его Python implementation. Admin-команды не загружают Flight
server, Arrow/Torch worker runtime или модель.

## Service

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
  lifecycle policies. Он не знает о Flight, SQLAlchemy, PyArrow, filesystem,
  subprocess и Torch.
- `service/application` содержит типизированные commands, queries, нейтральные
  results, scheduler orchestration и capability-oriented ports. Он зависит
  только от domain и внутренних worker/metrics contracts.
- `service/application/telemetry` содержит только best-effort records и
  доставку наблюдений. Telemetry records не размещаются в domain, а её
  persistence API не добавляется в общий job ledger.
- inbound Flight adapter проверяет структуру wire DTO нормативными JSON Schema
  Draft 2020-12, затем выполняет семантическую валидацию и mapping и
  преобразует application results и errors в Flight documents и Arrow status.
  Action names, descriptor paths, schema IDs и wire casing не попадают в
  application.
- outbound adapters реализуют PostgreSQL, artifact storage, worker process и
  worker capability boundaries. PostgreSQL adapter владеет транзакциями,
  idempotency, row locks и mapping database projections. Inbound и outbound
  adapters не импортируют друг друга.
- `service/bootstrap` — единственное место сборки конкретных service adapters.

Ports называются по возможностям: `JobLifecycleStore`, `JobQueryStore`,
`InputUploadStore`, `OutputAccessStore`, `JobRepository`,
`ArtifactPublisher`, `ExecutionPlanBuilder`, `AttemptProcess`,
`WorkerExecutor`, `WorkerCapabilities`, `DeviceLeaseManager`,
`AccessTokenAuthenticator`. Имена технологий
в application ports и generic `Repository[T]` не допускаются. Каждый port имеет
текущего runtime consumer и adapter; интерфейсы «на будущее» не создаются.

## Worker

`app/worker/` владеет Arrow-to-tensor data path, model, training, telemetry,
device/reproducibility runtime и checkpoint staging. Один процесс обслуживает
ровно один execution attempt. Worker:

- читает только service-owned immutable command manifest;
- пишет только в attempt workspace;
- отправляет bounded NDJSON events в stdout и diagnostics в stderr;
- не подключается к PostgreSQL и не знает о Flight, bearer auth, idempotency,
  public job state или `modelRef`;
- не публикует output, recovery generation или model generation.

Worker является осознанным исключением из внутреннего послойного разделения:
его executor может напрямую использовать PyArrow, Torch и filesystem как части
одного принадлежащего worker runtime. Ports вокруг tensors, Arrow replay и
checkpoint storage не вводятся без измеримой проблемы или второго
implementation.

ML-код имеет единственный канонический import path в `app/worker/`. Старые
параллельные пакеты `app/data`, `app/model`, `app/training`, `app/storage`,
`app/runtime` и `app/metrics` отсутствуют и повторно не вводятся.

Checkpoints принадлежат `app/worker/checkpoints/`, а не generic runtime-пакету.
Подготовка batch и prefetch принадлежат `app/worker/training/batching.py`;
исполнитель attempt только выбирает fit/predict use case и не содержит их
реализацию целиком.

Core результат global epoch находится в `app/worker/training/epoch.py` и не
зависит от telemetry. AMP/gradient counters, phase timings,
JSONL и plots принадлежат `app/worker/telemetry/`. Job progress хранит только
checkpoint-aligned `epoch`, `step`, `loss_stage`, `loss`; полный metrics
document является необязательным наблюдением.

## Local CLI

`app/local/` владеет локальными file/stream командами, `plot-metrics` и
`gmark`. Этот execution path может напрямую использовать worker-код: он
работает в локальном процессе и не является частью Flight service. Пакет
`app/commands` отсутствует, чтобы слово `commands` не обозначало одновременно
local CLI и application use cases сервиса.

Общие identity и путь корня проекта находятся в `app/project.py`. Встроенные
operational defaults находятся в `app/config.py`; runtime-владельцы сохраняют
configuration types, загрузку, валидацию и технологические преобразования.
Версионируемые worker contract defaults остаются в
`app/contracts/worker/v7/config.py`. Общий модуль не загружает environment или
adapters и не содержит mutable configuration state.

## Admin

`app/admin/cli` отвечает только за presentation. `app/admin/bootstrap`
создаёт короткоживущие PostgreSQL resources и вызывает application use cases
для access tokens и published models. Alembic-команды имеют отдельный
короткоживущий SQLAlchemy lifecycle. Admin-команды не запускают service или
worker.

## Contracts

- `app/contracts/flight/v5/` — нормативные schemas и fixtures публичного API;
- Flight v5 является текущей штатной архитектурой remote API; дальнейшие
  изменения проектируются от его lifecycle, durability и fencing semantics;
- `app/contracts/ml.py` — единая Python identity target, ML-контракта,
  checkpoint и recovery formats для Flight, worker и telemetry contracts;
- `app/contracts/worker/v7/` — command/result manifests, capability document,
  Arrow artifact manifests, events и exit semantics;
- `app/contracts/metrics/v3/` — текущий immutable epoch artifact, закрытая
  OpenSearch projection, golden identity и strict index templates;
- `app/contracts/metrics/fit_run/v2/` — terminal fit summary, lifecycle
  durations и counters;
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
tokens, owner-scoped state и published metadata. Owner устанавливается точным
subject API credential. Отдельный telemetry slice владеет epoch intervals, run
artifact metadata и metrics outbox. OpenSearch является
best-effort аналитической проекцией, а не частью model/job lifecycle.
SQLite и dual-write запрещены.

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
- `telemetry/` — run-owned best-effort artifacts до завершения outbox
  retention;
- RAM — FIFO queues, token digest cache и active process handles.

## Обязательные dependency rules

- domain не зависит от application, adapters или bootstrap;
- application не зависит от adapters или bootstrap;
- application не зависит от публичного Flight contract;
- application может зависеть от внутреннего metrics contract для
  детерминированной outbox projection;
- service domain и общий PostgreSQL ledger не содержат telemetry records и
  telemetry capabilities;
- core training epoch и core model publication не импортируют telemetry;
- adapters зависят от application/domain, но не от другого направления
  transport-а;
- service не импортирует `app.worker` implementation;
- worker не импортирует service, Flight или database implementation;
- admin не импортирует worker или Flight server;
- shared service/worker данные находятся только в `app/contracts/worker/v7`;
- import graph не содержит циклов;
- environment, connections, CUDA initialization и filesystem mutation не
  выполняются при import.

Правила закреплены AST- и process-import тестами в
`tests/architecture/test_boundaries.py`.

## Размещение нового кода

- lifecycle rule или record — `app/service/domain/`;
- command/query и capability port — `app/service/application/`;
- Flight parser/presenter — `app/service/adapters/inbound/flight/`;
- wire documents, descriptors и validation разделяются по этим
  ответственностям внутри Flight adapter;
- ORM/repository/Alembic — `app/service/adapters/outbound/postgres/`;
- транзакционные ledger capabilities — профильный модуль в
  `app/service/adapters/outbound/postgres/ledger/`;
- spool/publication — `app/service/adapters/outbound/artifacts/`;
- subprocess supervision — `app/service/adapters/outbound/worker/`;
- CUDA inventory — `app/service/adapters/outbound/cuda/`;
- OpenSearch transport — `app/service/adapters/outbound/opensearch/`;
- metrics artifacts —
  `app/service/adapters/outbound/artifacts/telemetry/`;
- metrics persistence и outbox —
  `app/service/adapters/outbound/postgres/telemetry/`;
- telemetry records и delivery orchestration —
  `app/service/application/telemetry/`;
- model/loss/trainer/Arrow tensor/checkpoint — профильный пакет в `app/worker/`;
- worker runtime observations, JSONL и plots — `app/worker/telemetry/`;
- local file/stream command — `app/local/`;
- wire/process schema — соответствующий versioned package в `app/contracts/`;
- runtime wiring — composition root конкретного процесса.

Новые `common`, `helpers`, `lib` и `misc` без одного ясного owner не создаются.
Compatibility facade не становится владельцем новой логики.
