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
| Быстрый старт Flight service | `docs/getting-started.md` |
| Training runtime, checkpoint и metrics | `docs/training-runtime.md` |
| Версии Python-пакетов проекта | `requirements.txt` |
| Конфигурация Ruff, Pyright, pytest и Alembic | `pyproject.toml` |
| Python types и tensor runtime contracts | `docs/policy/typing-policy.md` |
| Команды и порядок проверки изменений | `docs/policy/testing-policy.md` |
| Best-effort границы metrics и OpenSearch | `docs/policy/metrics-policy.md` |
| Authentication model и security boundary | `docs/authentication.md` |
| Выдача, передача, ротация и отзыв API access tokens | `docs/operations/api-access-tokens.md` |
| Проверка и изменение PostgreSQL schema | `docs/operations/database-migrations.md` |
| Просмотр и удаление published models | `docs/operations/published-models.md` |
| Python runtime, `.venv` и установка package dependencies | `docs/policy/python-runtime-policy.md` |
| Environment example | `.env.example` |
| ML behavior и training reference | профильные документы в `docs/` |
| Flight service behavior и operations | `docs/operations/flight-service.md` |
| Интеграция с Flight | `docs/flight-integration.md` |
| Текущие процессы, компоненты, contracts и data ownership | `docs/architecture.md` |
| Dependency boundaries и размещение кода | `docs/policy/architecture.md` |
| Активные ненормативные проектные предложения | `docs/design/index.md` |
| Исторический rationale архитектурных решений | `docs/adr/index.md` |
| Ручное production deployment | `docs/deployment/` |
| Правила разработки | `docs/policy/` |
| Нормативный semantic language v4 | `app/contracts/semantic/v4/` |
| Нормативный Flight v19 contract | `app/contracts/flight/v19/` |
| Нормативный worker v17 contract | `app/contracts/worker/v17/` |
| Нормативный checkpoint/recovery v10 contract | `app/contracts/checkpoint/v10/` |
| Нормативные training metrics contracts | `app/contracts/metrics/` |
| Нормативный Model Catalog Query v5 | `app/contracts/model_catalog/v5/` |
| Нормативный Model Topology Query v2 | `app/contracts/model_topology/v2/` |
| Нормативный Training Telemetry Query v4 | `app/contracts/training_telemetry/v4/` |
| Нормативная диагностика выходных головок v2 | `app/contracts/target_head_diagnostics/v2/` |
| Нормативный package weighted binary BCE | `app/contracts/semantic/v4/`, `app/contracts/checkpoint/v10/`, `app/contracts/worker/v17/`, `app/contracts/metrics/v9/`, `app/contracts/metrics/fit_run/v9/`, `app/contracts/model_catalog/v5/`, `app/contracts/model_topology/v2/`, `app/contracts/training_telemetry/v4/`, `app/contracts/flight/v19/` |
| История релизов | `CHANGELOG.md` |

В проекте нет отдельного `docs/index.md`; навигационной входной точкой остаётся
`readme.md`, для operational procedures — `docs/operations/index.md`, а для
политик — `docs/policy/index.md`.

## README

`readme.md` отвечает на вопросы:

- что делает проект;
- какие режимы работы поддержаны;
- как создать локальное окружение и увидеть основные команды;
- где находятся подробные contracts и operations.

README использует навигационную структуру: что есть в проекте, режимы, быстрый
старт, CLI, документация, структура и deployment. Длинные объяснения recovery,
deployment, protocol edge cases, training semantics и архитектурных границ
следует помещать в профильный документ, оставляя в README краткую ссылку.

## CLI help

CLI help является публичным контрактом.

Источники:

- `app/cli/parser.py` и `app/cli/parsers/` — parser tree и command groups;
- `app/cli/formatting.py` и `app/cli/options.py` — rendering и общие options;
- `app/cli/args.py` — parsing entry;
- `app/admin/` — command behavior;
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
группам переменных. Встроенные operational defaults находятся в
`app/config.py`. Правила parsing и validation остаются у runtime-владельцев:
`app/service/bootstrap/config.py`, `app/contracts/worker/v17/config.py`,
PostgreSQL и OpenSearch adapters.

Документация не должна:

- помещать реальные credentials в repository;
- обещать default, которого нет в коде;
- дублировать удалённые environment names;
- добавлять `POSTGRES_SSLMODE`, локальную SQLite или token file, которых нет в
  текущем contract.

## Нормативные contracts

JSON Schemas и golden fixtures semantic language v4, Flight v19, worker v17,
checkpoint/recovery v10, metrics v9, Model Catalog Query v5, Model Topology
Query v2, Training Telemetry Query v4 и Target Head Diagnostics Query v2
нормативны для текущего runtime. README или operations guide не могут
переопределять их. Flight v19 — единственный
текущий remote API contract; legacy aliases отсутствуют.

Semantic v4, Flight v19, worker v17, checkpoint/recovery v10, metrics v9 и
активные query revisions составляют единый package текущего runtime.

`app/contracts/model_catalog/v5/` имеет независимую revision и активирован
двумя Flight v19 actions. Изменение его query language не обязано синхронно
менять job workflow Flight.

`app/contracts/training_telemetry/v4/` имеет независимую revision и активирован
двумя Flight v19 actions. Изменение query language не обязано
синхронно менять job workflow Flight.

`app/contracts/model_topology/v2/` имеет независимую revision и активирован
одним Flight v19 action. Изменение topology language не обязано синхронно
менять job workflow Flight.

`app/contracts/target_head_diagnostics/v2/` имеет независимую revision и
активирован одним Flight v19 action. Он описывает opt-in runtime observations,
а не semantic language или checkpoint compatibility.

Изменение Flight contract требует синхронно проверить:

- schemas и fixtures;
- parser/serializer;
- contract tests;
- version/compatibility policy;
- Руководство по интеграции с Flight.

Flight v19 schemas и fixtures остаются нормативными для remote API. Эти
contracts проверяются тестами.

## Документация текущего состояния

Living documentation описывает только действующую систему. Architecture,
security, persistence, protocol, training и operations facts размещаются в
профильном источнике с конкретным читателем.

ADR хранит historical rationale отдельного архитектурного решения, но не
является system reference или нормативным источником текущего состояния.
Admission, immutable lifecycle и допустимое содержание ADR задаёт
[`docs/adr/index.md`](../adr/index.md). Решение, не проходящее admission,
остаётся в issue, merge request или активном Design Note.

После реализации release history сохраняет `CHANGELOG.md`, а точную историю
изменений — Git. Текущий PostgreSQL baseline и последующую эволюцию schema
задают Alembic migrations. Если опубликованная цепочка заменена baseline,
удалённые revisions остаются в release tag и Git history, а compatibility
boundary явно фиксируется в operations. Эти исторические источники не
заменяют документацию текущего состояния.

При замене решения профильный документ обновляется в том же change set:
устаревшая и отменённая семантика удаляется, а не переносится в новый
постоянный справочник. Сводный `decisions.md`, дублирующий ADR или current
documentation, в проекте не ведётся.

## Design Notes

`docs/design/` содержит только активные ненормативные предложения, для которых
ещё не принято архитектурное решение. Design Note может инвентаризировать
текущее состояние и сравнивать варианты, но не определяет current behavior,
contract или обязательную реализацию.

После завершения обсуждения Design Note удаляется из текущей документации.
Принятое долгосрочное решение при необходимости получает ADR, а реализованное
состояние — профильную living documentation. История предложения остаётся в
Git. Поэтому `docs/design/` не становится параллельным справочником системы или
архивом отклонённых идей.

## Deployment и operations

`docs/deployment/` содержит короткие ручные инструкции установки и запуска.
`docs/operations/` содержит operator-facing lifecycle procedures для service и
application-managed ресурсов. Их состав и границы перечисляет
`docs/operations/index.md`; каждый resource lifecycle имеет один основной
Operational Guide.
`docs/operations/flight-service.md` содержит подробные lifecycle, recovery,
security и storage semantics Flight service.

Deployment guide не должен дублировать всю архитектуру сервиса или
автоматизировать изменение production-хоста. Для target Fedora deployment он
фиксирует один production root `/home/nerv/transformer`, один runtime layout и
способ запуска; альтернативные или предположительные варианты в
reference-инструкцию не добавляются.

## Policy documents

Документы в `docs/policy/` должны быть прямыми и применимыми на review.
Примеры обязаны использовать Python и реальные concepts Transformer, но не
должны превращать общую policy в описание конкретной модели или внешней
системы.

## Имена и оформление

- постоянные filenames — английский `kebab-case`, если путь уже не закреплён;
- H1 — понятный русский заголовок;
- metadata `> Тип: ...` называет фактическую роль файла: справочник, политика,
  указатель, Design Note, руководство, операционное руководство или контракт;
- code identifiers и wire fields сохраняются в исходной форме;
- relative links должны разрешаться из текущего файла;
- путь документа считается стабильным контрактом и не меняется без причины.

## Терминология границ

В документации не используется общий ярлык `Consumer` для разных понятий.
Называйте аутентифицированную сторону Flight вызывающей системой, источник
предметной семантики — внешней предметной стороной, а зависимый код —
использующим кодом. Эти роли не становятся префиксами schemas, fixtures,
sample identities или package names.

## Добавление документа

Перед созданием файла нужно ответить:

1. Какой вопрос он закрывает?
2. Почему существующий источник не подходит?
3. Где на него будет ссылка?
4. Не дублирует ли он contract, профильный источник, ADR или README?

Временная идея без устойчивой роли остаётся issue/plan, а не постоянным
документом. Design Note допустим только как активный материал конкретного
архитектурного обсуждения с lifecycle, заданным выше.

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
