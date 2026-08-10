# Transformer Arrow Flight service

Python-сервис для обучения и инференса PyTorch Transformer на датасетах Apache
Arrow. Он поддерживает локальный CLI и durable remote jobs по Arrow Flight.

## Что есть в проекте

- local CLI для обучения и prediction из Arrow IPC files;
- stream CLI для framed Arrow payloads через standard streams;
- single-instance Arrow Flight v3 service с durable streaming jobs,
  PostgreSQL state, cross-system fencing, recovery, API tokens и CUDA
  scheduling;
- versioned public Flight и internal worker contracts;
- documentation, ADR и политики изменения в `docs/`.

## Режимы работы

| Режим | Вход | Результат | Основной справочник |
| --- | --- | --- | --- |
| File CLI | Arrow IPC file | checkpoint или Arrow prediction file | [CLI](./docs/cli/index.md) |
| Stream CLI | framed Arrow stdin | checkpoint или framed Arrow stdout | [local Arrow protocol](./docs/local-arrow-protocol.md) |
| Arrow Flight v3 | authenticated Flight RPC | durable streaming job, `modelRef` или output ticket | [Flight contract](./app/contracts/flight/v3/README.md) |

## Быстрый старт

Из корня проекта создайте чистое virtual environment и установите
зафиксированные зависимости:

```bash
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

```bash
./.venv/bin/python ./app/main.py --help
./.venv/bin/python ./app/main.py --version
./.venv/bin/python ./app/main.py <command> --help
```

- [`fit INPUT` и `predict INPUT`](./docs/cli/index.md) — обучение и prediction
  из Arrow files;
- [`fit-stream` и `predict-stream`](./docs/local-arrow-protocol.md) —
  обучение и prediction через framed standard streams;
- [`plot-metrics METRICS_FILE`](./docs/training-runtime.md) — SVG-графики по
  training metrics JSONL;
- [`flight serve`](./docs/flight-operations.md) — durable Arrow Flight job
  service;
- [`auth tokens issue|list|revoke` и `db migrations`](./docs/flight-operations.md)
  — access tokens и schema PostgreSQL.

Точные options, defaults, aliases и side effects описывает help leaf-команды;
поведение local commands и paths — [справочник CLI](./docs/cli/index.md).

## Документация

- [Начало работы](./docs/getting-started.md)
- [Справочник CLI](./docs/cli/index.md)
- [Локальный Arrow и stream contract](./docs/local-arrow-protocol.md)
- [Training runtime и checkpoint](./docs/training-runtime.md)
- [Функция потерь](./docs/losses.md)
- [Arrow Flight v3 contract](./app/contracts/flight/v3/README.md)
- [Worker process contract v2](./app/contracts/worker/v2/README.md)
- [Flight runbook](./docs/flight-operations.md)
- [Развёртывание через systemd](./docs/deployment/systemd.md)
- [Архитектурные решения](./docs/adr/)
- [Политики проекта](./docs/policy/index.md)

## Структура проекта

```text
app/main.py          # тонкий CLI entrypoint
app/cli/             # argparse и форматированный --help/--version
app/contracts/       # public Flight v3 и internal worker v2 contracts
app/service/         # domain/application, Flight/outbound adapters, bootstrap
app/worker/          # Arrow/Torch model, training, checkpoints и process root
app/admin/           # auth/database CLI и composition roots
app/flight/          # временные compatibility imports старого service API
app/database/        # временные compatibility imports PostgreSQL adapter
docs/                # пользовательская документация, ADR и политики
recovery/            # runtime-created persistent fit inputs/checkpoints
app/config.py        # project defaults
```

## Развертывание

Production-запуск на Fedora через systemd описан в
[docs/deployment/systemd.md](./docs/deployment/systemd.md). PostgreSQL,
tokens, recovery, storage lifecycle и TLS/mTLS собраны в
[Flight runbook](./docs/flight-operations.md).

Bearer authentication обязательна при любом transport. Plaintext разрешается
только явным `--allow-plaintext`; не открывайте такой endpoint в недоверенную
сеть.
