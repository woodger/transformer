# Политика ведения документации

> Тип: политика. Этот документ задаёт роли документов и правила их обновления.

Документация должна помогать запустить Transformer, понять его contracts и
безопасно изменить систему. Один факт должен иметь один основной источник
правды; в остальных местах используется краткое резюме и ссылка.

Документация проекта ведётся на русском языке. Имена API, wire fields, code
identifiers, команды, пути, значения enum и другие элементы машинного
контракта сохраняются в исходной форме. Английский текст допустим внутри
дословных примеров и там, где перевод изменил бы значение внешнего термина.

## Текущая структура

| Тема | Основной источник |
| --- | --- |
| Назначение, навигация и общий CLI | `readme.md` |
| Локальный quick start | `docs/getting-started.md` |
| Local CLI commands, options и artifact paths | `docs/cli/index.md` |
| Локальные Arrow IPC columns и framed stream protocol | `docs/local-arrow-protocol.md` |
| Local training runtime, checkpoint и metrics | `docs/training-runtime.md` |
| Версии Python-пакетов проекта | `requirements.txt` |
| Конфигурация Ruff, Pyright, pytest и Alembic | `pyproject.toml` |
| Python types и tensor runtime contracts | `docs/policy/typing-policy.md` |
| Команды и порядок проверки изменений | `docs/policy/testing-policy.md` |
| Best-effort границы metrics и OpenSearch | `docs/metrics.md` |
| Python runtime, `.venv` и установка package dependencies | `docs/policy/python-runtime-policy.md` |
| Environment example | `.env.example` |
| ML architecture и training decisions | профильные документы в `docs/` |
| Flight service behavior и operations | `docs/flight-operations.md` |
| Inventory integration | `docs/inventory-flight-handoff.md` |
| Архитектурные решения | `docs/adr/` |
| Ручное production deployment | `docs/deployment/` |
| Правила разработки | `docs/policy/` |
| Нормативный Flight v4 contract | `app/contracts/flight/v4/` |
| Нормативный worker v6 contract | `app/contracts/worker/v6/` |
| Нормативные training metrics contracts | `app/contracts/metrics/` |
| История релизов | `CHANGELOG.md` |

В проекте нет отдельного `docs/index.md`; навигационной входной точкой остаётся
`readme.md`, а для политик — `docs/policy/index.md`.

## README

`readme.md` отвечает на вопросы:

- что делает проект;
- какие режимы работы поддержаны;
- как создать локальное окружение и увидеть основные команды;
- где находятся подробные contracts и operations.

README использует навигационную структуру: что есть в проекте, режимы, быстрый
старт, CLI, документация, структура и deployment. Длинные объяснения recovery,
deployment, protocol edge cases, training semantics и архитектурных решений
следует помещать в профильный документ, оставляя в README краткую ссылку.

## CLI help

CLI help является публичным контрактом.

Источники:

- `app/cli/parser.py` и `app/cli/parsers/` — parser tree и command groups;
- `app/cli/formatting.py` и `app/cli/options.py` — rendering и общие options;
- `app/cli/args.py` — parsing entry;
- `app/local/` и `app/admin/` — command behavior;
- process roots в `app/service/bootstrap/`, `app/worker/bootstrap/` и
  `app/admin/bootstrap/` — resource lifecycle.

Глобальный help остаётся компактным: Usage, global options и список command
groups. Полные arguments/options находятся в leaf command help:

```text
transformer <command> --help
```

Блок `Examples` добавляется только когда показывает неочевидную комбинацию,
формат значения, transport mode или side effect. Пример, дублирующий `Usage`,
не нужен.

При изменении command path, option, default или output одновременно
обновляются help tests и относящаяся пользовательская документация.

## Environment

`.env.example` содержит безопасный рабочий образец и русские комментарии к
группам переменных. Фактические правила parsing и defaults находятся у
владельцев runtime: `app/local/config.py`,
`app/service/bootstrap/config.py`, `app/contracts/worker/v6/config.py` и
`app/service/adapters/outbound/postgres/config.py`.

Документация не должна:

- помещать реальные credentials в repository;
- обещать default, которого нет в коде;
- дублировать удалённые environment names;
- добавлять `POSTGRES_SSLMODE`, локальную SQLite или token file, которых нет в
  текущем contract.

## Нормативные contracts

JSON Schemas и golden fixtures в `app/contracts/flight/v4/` нормативны для wire
format. README или operations guide не могут переопределять их. Flight v4 —
единственный текущий remote API contract и базовая точка для дальнейших
изменений.

Изменение Flight contract требует синхронно проверить:

- schemas и fixtures;
- parser/serializer;
- contract tests;
- version/compatibility policy;
- Inventory handoff.

Для local CLI checkpoint, Arrow columns и framed protocol имеют единственные
основные источники: `docs/training-runtime.md` и
`docs/local-arrow-protocol.md`. Flight v4 schemas и fixtures остаются
нормативными для remote API. Эти contracts проверяются тестами.

## ADR

ADR фиксирует значимое принятое решение, альтернативы и последствия. ADR нужен
для изменения service boundary, persistence ownership, durability model,
protocol или multi-process architecture.

ADR не используется как пошаговый runbook и не переписывается так, будто
предыдущее решение никогда не существовало. Новое решение оформляется новым
ADR или явным изменением статуса.

## Deployment и operations

`docs/deployment/` содержит короткие ручные инструкции установки и запуска.
`docs/flight-operations.md` содержит подробные lifecycle, recovery, security и
storage semantics.

Deployment guide не должен дублировать всю архитектуру сервиса или
автоматизировать изменение production-хоста. Для target Fedora deployment он
фиксирует один production root `/home/nerv/transformer`, один runtime layout и
способ запуска; альтернативные или предположительные варианты в
reference-инструкцию не добавляются.

## Policy documents

Документы в `docs/policy/` должны быть прямыми и применимыми на review.
Примеры обязаны использовать Python и реальные concepts Transformer, но не
должны превращать общую policy в описание конкретной модели или consumer.

## Имена и оформление

- постоянные filenames — английский `kebab-case`, если путь уже не закреплён;
- H1 — понятный русский заголовок;
- metadata `> Тип: политика|справочник` рекомендуется для policy/reference;
- code identifiers и wire fields сохраняются в исходной форме;
- relative links должны разрешаться из текущего файла;
- путь документа считается стабильным контрактом и не меняется без причины.

## Добавление документа

Перед созданием файла нужно ответить:

1. Какой вопрос он закрывает?
2. Почему существующий источник не подходит?
3. Где на него будет ссылка?
4. Не дублирует ли он contract, ADR или README?

Временная идея без устойчивой роли остаётся issue/plan, а не постоянным
документом.

## Обновление

Если code change меняет запуск, CLI, environment, schema, persistence,
protocol, checkpoint или архитектурную границу, соответствующая документация
обновляется в том же change set.

Устаревший текст удаляется, а не сохраняется «на всякий случай». Перед
завершением проверяются links, команды, paths, version markers и отсутствие
секретов. Deployment-команда считается документированной только после
успешной проверки на целевом host environment.

Команды Ruff, Pyright и pytest для проверки изменений публикуются только в
`docs/policy/testing-policy.md`. Guides и другие policies ссылаются на него и
не создают сокращённую или альтернативную последовательность.
