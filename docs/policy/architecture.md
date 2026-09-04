# Архитектурная политика

> Тип: политика. Обязательные направления зависимостей и правила размещения
> кода Transformer.

Текущие процессы, компоненты, contracts и data ownership описывает
[архитектурный справочник](../architecture.md). Этот документ задаёт правила,
которые должны сохраняться при изменении системы. Историческое обоснование
process boundaries находится в
[ADR 0001](../adr/0001-arrow-flight-job-service.md) и
[ADR 0004](../adr/0004-clean-architecture-process-boundaries.md).

## Границы процессов и слоёв

Clean Architecture применяется отдельно к каждому исполняемому процессу.
Единый bootstrap, импортирующий весь проект, не создаётся. Composition root
конкретного процесса является единственным местом сборки его adapters и
resources.

Внутри Flight service:

- domain не зависит от application, adapters или bootstrap;
- application зависит только от domain и внутренних process/metrics contracts;
- application не зависит от adapters, bootstrap или публичного Flight
  contract;
- adapters зависят от application/domain, но inbound и outbound adapters не
  импортируют друг друга;
- service domain и общий PostgreSQL ledger не содержат telemetry records или
  telemetry capabilities;
- core training epoch и model publication не импортируют telemetry.

Между процессами:

- service не импортирует implementation из `app.worker`;
- worker не импортирует service, Flight или database implementation;
- admin не импортирует worker или Flight server;
- service/worker обмениваются данными только через
  `app/contracts/worker/v11`;
- local CLI может напрямую использовать worker-код, потому что это один
  локальный execution path, а не Flight service boundary.

Import graph не содержит циклов. Environment, connections, CUDA
initialization и filesystem mutation не выполняются при import. Эти правила
проверяются AST- и process-import тестами в
`tests/architecture/test_boundaries.py`.

## Ports и ownership

Application ports называются по возможностям, а не по технологиям. Generic
`Repository[T]` и technology-specific port names не используются. Каждый port
имеет текущего runtime consumer и adapter; интерфейсы без действующего
потребителя не добавляются.

PostgreSQL adapter владеет транзакциями, idempotency, row locks и mapping
database projections. PostgreSQL-транзакция не охватывает filesystem или
subprocess; artifact publication выполняется staged до фиксации database
reference. OpenSearch не используется как источник состояния приложения.
SQLite и dual-write persistence не добавляются.

Worker остаётся цельным Arrow/Torch runtime. Ports вокруг tensors, Arrow replay
или checkpoint storage добавляются только при измеримой проблеме или втором
implementation. Параллельные ML-пакеты, конкурирующие с каноническим
`app/worker/`, не создаются.

Положение configuration module определяется его зависимостями и поведением, а
не именем файла. Общий модуль чистых статических defaults не является внешним
слоем: зависимости domain/application на такой модуль и его зависимости на
domain-типы допустимы. Конкретный import graph при этом остаётся ацикличным.

Общий чистый configuration module содержит только deterministic immutable
values. Он не читает environment или credentials, не выполняет I/O и runtime
initialization, не импортирует adapters, bootstrap, provider SDK или другие
runtime dependencies и не хранит mutable state. Configuration types, parsing,
validation и wiring остаются у соответствующих runtime-владельцев.

Тип, выражающий понятие Transformer, принадлежит domain или профильному
внутреннему versioned contract. Domain/application и внутренние contracts не
зависят от типов внешнего provider SDK или runtime library; преобразование в
provider-owned representation выполняет infrastructure adapter на границе.

Generic `app/commands` не вводится как второй владелец рядом с local CLI и
application use cases.

## Размещение нового кода

- lifecycle rule или record — `app/service/domain/`;
- command/query и capability port — `app/service/application/`;
- Flight parser/presenter — `app/service/adapters/inbound/flight/`;
- wire documents, descriptors и validation разделяются по ответственностям
  внутри Flight adapter;
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
- model/loss/trainer/Arrow tensor/checkpoint — профильный пакет в
  `app/worker/`;
- worker runtime observations, JSONL и plots — `app/worker/telemetry/`;
- local file/stream command — `app/local/`;
- wire/process schema — соответствующий versioned package в `app/contracts/`;
- runtime wiring — composition root конкретного процесса.

Новые `common`, `helpers`, `lib` и `misc` без одного ясного owner не создаются.
Compatibility facade не становится владельцем новой логики.
