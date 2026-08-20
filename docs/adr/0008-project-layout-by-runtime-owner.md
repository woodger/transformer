# ADR 0008: структура проекта по runtime-владельцам

- Статус: принято
- Дата: 2026-08-13
- Уточняет: ADR 0004 и ADR 0006
- Структурная граница telemetry уточнена в ADR 0013

## Контекст

После выделения процессов service, worker и admin production-код уже соблюдал
направление зависимостей Clean Architecture. При этом рядом оставались прежние
пути `app/commands`, общий `app/config.py`, крупный `app/cli/help.py`, legacy
названия outbound adapters и плоский каталог тестов. Несколько путей обозначали
одну ответственность, а тесты и документация поддерживали обе карты проекта.

Проблема была навигационной и архитектурной, а не runtime-функциональной:
разработчик не мог определить канонического владельца по пути файла.

## Решение

Проект остаётся модульным монолитом с отдельными процессами service и worker.
Python package не переименовывается и сохраняет корень `app/`, но внутри него
каждая ответственность получает один канонический путь:

```text
app/
├── cli/                         parser и presentation общего CLI
├── local/                       file/stream commands и gmark
├── contracts/                   Flight v4 и worker v3
├── service/
│   ├── domain/
│   ├── application/
│   │   ├── commands/
│   │   ├── queries/
│   │   ├── services/
│   │   ├── ports/
│   │   └── messages/
│   ├── adapters/
│   │   ├── inbound/flight/
│   │   └── outbound/{postgres,artifacts,worker,cuda}/
│   └── bootstrap/
├── worker/
│   ├── application/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── checkpoints/
│   ├── telemetry/
│   ├── runtime/
│   └── bootstrap/
├── admin/
└── project.py
```

Flight adapter разделяет wire documents, descriptor paths и validation.
PostgreSQL ledger сгруппирован по durable capabilities внутри собственного
пакета. Worker executor только диспетчеризует fit/predict; artifact boundary и
оба use case находятся в профильных модулях. Batch construction, shuffle и
prefetch принадлежат `worker/training/batching.py`.

Конфигурация не имеет общего mutable или generic владельца. Identity и путь
корня находятся в `app/project.py`; local, service, PostgreSQL и worker
defaults находятся в соответствующих пакетах.

Тесты организованы по уровню проверки:

```text
tests/
├── unit/{cli,local,service,worker}/
├── contract/{flight_v4,worker_v3}/
├── integration/{flight,postgres,worker_process}/
├── architecture/
└── support/
```

Compatibility facades и прежние source paths удаляются после одновременной
миграции production-кода, тестов и документации. Поддерживаемого внутреннего
Python API у них нет.

## Что не меняется

- Flight v4, worker v3, Arrow schemas и persisted formats;
- CLI commands, options и entrypoint `app/main.py`;
- PostgreSQL schema, Alembic chain и filesystem layout;
- process isolation и порядок runtime initialization;
- worker как осознанное цельное runtime-исключение из Clean Architecture.

Перенос в `src/transformer` не выполняется: он затронул бы каждый import,
subprocess entrypoint и deployment command, не устраняя дополнительной
архитектурной проблемы после удаления параллельных путей. Реорганизация всего
`docs/` также не выполняется: роли документов уже заданы policy, а массовое
изменение ссылок не улучшает runtime ownership.

## Контроль

Architecture tests проверяют направление импортов, отсутствие циклов,
process isolation, канонические contracts/composition roots и отсутствие
прежних source paths. Ruff, Pyright и полный pytest остаются обязательной
проверкой широкого рефакторинга.

## Последствия

- путь файла однозначно указывает runtime-владельца и слой;
- production-код и тесты используют одну карту imports;
- удалены generic `config`, `utils` и неоднозначный `commands` package;
- крупные модули CLI, worker executor и trainer разделены по устойчивым
  обязанностям;
- переносы создают большой, но behavior-preserving diff без compatibility
  shims.
