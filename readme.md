# Transformer Arrow Flight service

Python-сервис для обучения и инференса PyTorch Transformer на датасетах Apache
Arrow. Он поддерживает локальный CLI и durable remote jobs по Arrow Flight.

## Что есть в проекте

- local CLI для обучения и prediction из Arrow IPC files;
- stream CLI для framed Arrow payloads через standard streams;
- single-instance Arrow Flight v6 service с durable streaming jobs,
  PostgreSQL state, cross-system fencing, recovery, API tokens и GPU
  scheduling;
- versioned public Flight и internal worker contracts;
- документацию текущего состояния, исторические decision records и политики
  изменения в `docs/`.

## Режимы работы

| Режим | Вход | Результат | Основной справочник |
| --- | --- | --- | --- |
| File CLI | Arrow IPC file | checkpoint или Arrow prediction file | [CLI](./docs/cli/index.md) |
| Stream CLI | framed Arrow stdin | checkpoint или framed Arrow stdout | [local Arrow protocol](./docs/local-arrow-protocol.md) |
| Arrow Flight v6 | authenticated Flight RPC | durable streaming job, `modelRef` или output ticket | [Flight contract](./app/contracts/flight/v6/README.md) |

## Быстрый старт

Из корня проекта создайте чистое virtual environment и установите
зафиксированные зависимости:

```sh
/usr/bin/python3 -m venv --clear .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./app/main.py --help
```

Project `.venv` — единственное окружение приложения. Подходящую версию
системного `/usr/bin/python3` обеспечивает владелец development или production
среды; не устанавливайте application dependencies в system Python или
user-site. Полный локальный сценарий находится в
[начале работы](./docs/getting-started.md).

## CLI команды

Справка:

```sh
./.venv/bin/python ./app/main.py --help
./.venv/bin/python ./app/main.py --version
./.venv/bin/python ./app/main.py <command> --help
```

- [`fit INPUT` и `predict INPUT`](./docs/cli/index.md) — обучение и prediction
  из Arrow files;
- [`fit-stream` и `predict-stream`](./docs/local-arrow-protocol.md) —
  обучение и prediction через framed standard streams;
- [`gmark`](./docs/cli/index.md#gpu-stress-test) — CUDA training, AMP и
  integrity stress test;
- [`plot-metrics METRICS_FILE`](./docs/training-runtime.md) — SVG-графики по
  training metrics JSONL;
- [`flight serve`](./docs/operations/flight-service.md) — durable Arrow Flight job
  service;
- [`auth tokens issue|list|revoke`](./docs/operations/api-access-tokens.md) —
  lifecycle API access tokens;
- [`models list|delete`](./docs/operations/published-models.md) — lifecycle
  опубликованных model generations;
- [`db migrations`](./docs/operations/database-migrations.md) — schema
  PostgreSQL.

Точные options, defaults, aliases и side effects описывает help leaf-команды;
поведение local commands и paths — [справочник CLI](./docs/cli/index.md).

## Документация

- [Начало работы](./docs/getting-started.md)
- [Справочник CLI](./docs/cli/index.md)
- [Локальный Arrow и stream contract](./docs/local-arrow-protocol.md)
- [Training runtime и checkpoint](./docs/training-runtime.md)
- [Функция потерь](./docs/losses.md)
- [Arrow Flight v6 contract](./app/contracts/flight/v6/README.md)
- [Worker process contract v7](./app/contracts/worker/v7/README.md)
- [Training metrics contract v3](./app/contracts/metrics/v3/README.md)
- [Аутентификация Flight](./docs/authentication.md)
- [Операционные руководства](./docs/operations/index.md)
- [Flight runbook](./docs/operations/flight-service.md)
- [Развёртывание через systemd](./docs/deployment/systemd.md)
- [Доставка training metrics в OpenSearch](./docs/deployment/opensearch.md)
- [Архитектура Transformer](./docs/architecture.md)
- [Журнал архитектурных решений](./docs/adr/index.md)
- [Политики проекта](./docs/policy/index.md)

## Структура проекта

```text
app/main.py          # тонкий CLI entrypoint
app/config.py        # единый источник встроенных operational defaults
app/cli/             # parser, help formatting и command-group parsers
app/local/           # локальные file/stream commands и GPU diagnostics
app/contracts/       # public Flight v6, internal worker v7 и metrics v3
app/service/         # domain/application, Flight/outbound adapters, bootstrap
app/worker/          # Arrow/Torch model, training, checkpoints и process root
app/admin/           # auth/database CLI и composition roots
app/project.py       # identity и путь корня проекта
docs/                # пользовательская документация, ADR и политики
recovery/            # runtime-created persistent fit inputs/checkpoints
tests/               # unit, contract, integration и architecture tests
```

## Развертывание

Production-запуск на Fedora через systemd описан в
[docs/deployment/systemd.md](./docs/deployment/systemd.md). Schema PostgreSQL,
tokens и published models управляются по
[операционным руководствам](./docs/operations/index.md), а recovery, storage
lifecycle и TLS/mTLS описаны в [Flight runbook](./docs/operations/flight-service.md).

Bearer authentication обязательна при любом transport. Полная пара
`--tls-cert-file`/`--tls-key-file` включает TLS; без неё endpoint использует
plaintext, поэтому не открывайте его в недоверенную сеть.
