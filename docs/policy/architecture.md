# Архитектурная политика

> Type: Policy. Этот документ фиксирует существующие package boundaries,
> направление зависимостей и ownership данных Transformer.

Проект использует package-oriented архитектуру. В нём нет искусственного
деления на `domain/application/infrastructure`, и такая структура не должна
добавляться без отдельного архитектурного решения.

## Общая схема

```text
app/main.py
├── app/cli → app/commands → data/model/training/metrics/storage/runtime
└── app/flight → app/database + runtime spool + recovery store + scheduler
                                                        └── subprocess
                                                            → fit-stream|predict-stream

contracts/flight/v2 ← нормативный внешний контракт
migrations/         ← эволюция PostgreSQL schema
```

`app/main.py` является composition root и CLI dispatcher. Flight RPC не
выполняет Torch внутри handler: worker запускает существующий CLI в отдельной
process group, как зафиксировано в
[ADR 0001](../adr/0001-arrow-flight-job-service.md).

## Роли пакетов

### Core ML и data path

- `app/model/` — PyTorch modules, context masking и positional encoding;
- `app/training/` — training config, losses, scheduler, early stopping и
  trainer lifecycle;
- `app/data/` — Arrow IO, framed protocol, tensor reshape и shape validation;
- `app/storage/` — atomic file operations и checkpoint formats;
- `app/metrics/` — metrics records, JSONL и plots;
- `app/runtime/` — device selection, reproducibility и version;
- `app/config.py` — встроенные defaults проекта.

Эти пакеты не должны зависеть от CLI presentation, Flight server или
PostgreSQL control plane. Допустимы направленные зависимости между профильными
core packages, если они соответствуют текущему data/training flow.

### CLI

- `app/cli/` определяет parser, help и пользовательскую форму команд;
- `app/commands/` оркестрирует конкретные CLI-сценарии;
- `app/main.py` выбирает handler и инициализирует только нужные ресурсы.

Создание parser или вывод `--help` не должны открывать PostgreSQL connections,
создавать runtime spool или запускать training.

### PostgreSQL

`app/database/` владеет configuration, SQLAlchemy models, sessions, migrations
и token persistence. Он не должен зависеть от `app/flight/` или CLI handlers.

Изменение ORM-модели, которое меняет schema, требует Alembic migration.
SQLite, локальный ledger и дублирующее durable storage не допускаются.

### Arrow Flight service

`app/flight/` владеет:

- authentication и transport boundary;
- action contract validation;
- job coordination и state transitions;
- PostgreSQL ledger;
- `/tmp/transformer` runtime spool и persistent `recovery/`;
- physical CUDA inventory, worker queues и subprocess lifecycle;
- cancellation, recovery, maintenance и observability.

RPC handler выполняет только bounded validation, IO и control-plane mutation.
Model training и prediction остаются в worker subprocess. Network request не
может передавать произвольный filesystem path или CLI argument.

### Внешние contracts и migrations

- `contracts/flight/v2/` содержит нормативные JSON Schemas и golden fixtures;
- `migrations/` содержит Alembic environment и последовательность revisions;
- `docs/adr/` фиксирует принятые архитектурные решения.

Эти файлы могут не иметь обычного Python import path, но являются частью
production contract и tool-driven runtime.

## Ownership и durability

Архитектурное разделение данных является обязательным:

- PostgreSQL — jobs, attempts, idempotency, tickets, recovery metadata, model
  metadata и tokens;
- `/tmp/transformer` — prediction payloads, attempt artifacts, runtime storage
  epoch и boot-scoped CUDA quarantine;
- `recovery/` — persistent fit payloads и внутренние completed-epoch
  checkpoints без `modelRef`;
- `models/` — только успешно опубликованные immutable checkpoints;
- RAM — worker queues, token digest cache и активное process state.

Потеря `/tmp` инвалидирует prediction jobs и attempt-local artifacts, но fit с
целыми persistent inputs переходит в `RETRYING`. Потеря зарегистрированного
файла из `recovery/` является явной ошибкой и не разрешает silent restart
обучения. Уже опубликованные models и access tokens не затрагиваются.
PostgreSQL нельзя использовать как blob/payload storage или idle queue polling
mechanism.

## Направление зависимостей

Запрещено:

- импортировать `app/flight` или `app/database` из core ML packages;
- переносить training execution в Flight RPC thread;
- помещать CLI formatting в model, training, database или Flight state logic;
- читать environment или открывать resources при import модуля;
- дублировать wire validation независимо от normative contract;
- смешивать published models, internal recovery artifacts и ephemeral runtime
  spool;
- обходить ORM/session boundary случайными SQL-запросами в других packages.

Допустимо:

- `app/commands` зависит от профильных core/database packages;
- `app/flight` переиспользует Arrow, checkpoint и immutable training config;
- `app/flight` зависит от `app/database`;
- `app/main.py` и `app/flight/application.py` собирают runtime components;
- integration adapter преобразует внешний contract в внутренние records на
  границе package.

## Размещение файлов

Новый файл помещается в пакет, который владеет его основной ответственностью.

Примеры:

- новый loss или scheduler — `app/training/`;
- новый Arrow validator — `app/data/` или `app/flight/arrow.py`, в зависимости
  от того, является ли правило общим или transport-specific;
- checkpoint serialization — `app/storage/`;
- PostgreSQL repository — `app/database/`;
- Flight action parsing — `app/flight/`;
- CLI handler — `app/commands/`;
- parser/help metadata — `app/cli/`.

Новые generic directories `common`, `shared`, `helpers`, `lib` и `misc` без
чёткой роли не создаются. Новая top-level directory или package boundary
требует обновления архитектурной документации.

## Минимальность архитектурных изменений

Перенос файла не должен одновременно менять его public API и runtime
поведение. Массовое переименование, новая package hierarchy или смена
persistence ownership выполняются отдельным решением с тестами и документацией.

Если placement меняет ownership данных, направление зависимостей или внешний
контракт, это не локальный cleanup и требует явного согласования.
