# Политика жизненности кода

> Type: Policy. Этот документ задаёт правила deletion audit и cleanup pass.

Файл считается живым не потому, что импортируется тестом. У него должна быть
подтверждённая роль в runtime graph, tool graph, внешнем contract или
осознанном test harness.

## Графы использования

### Runtime graph

Основные roots:

- `app/main.py` и зарегистрированные CLI commands;
- `app/service/bootstrap/application.py` и Flight server wiring;
- `app/worker/bootstrap/` и один worker process на attempt;
- `app/admin/bootstrap/` для auth/database commands;
- package imports и `__init__.py` exports;
- systemd entrypoint и documented operator commands.

### Tool и contract graph

Некоторые живые файлы не импортируются обычным production-кодом:

- `app/service/adapters/outbound/postgres/alembic/env.py` и
  `app/service/adapters/outbound/postgres/alembic/versions/*.py`,
  загружаемые Alembic;
- `app/contracts/flight/v4/`, используемый внешними consumers и contract tests;
- `app/contracts/worker/v6/`, используемый service и worker processes;
- `.env.example`, `pyproject.toml` и deployment reference;
- golden JSON/Arrow fixtures;
- files, найденные по dynamic string path или reflection.

Отсутствие Python import не делает такие файлы мёртвыми.

### Test graph

Test graph начинается с `tests/{unit,contract,integration,architecture}/`,
`tests/conftest.py`, `tests/support/` и pytest fixtures. Тест подтверждает
проверяемое поведение, но сам по себе не доказывает production usage.

## Категории

### Active runtime code

Код достижим из runtime root, command dispatch, Flight composition или worker
subprocess flow.

Решение: `keep`.

### Tool/contract file

Файл загружается Alembic, описывает нормативный wire contract, deployment или
release metadata.

Решение: `keep`, пока соответствующий tool или внешний contract поддерживается.

### Unused file

Нет imports, exports, dynamic references, tests, docs, tool usage или внешней
contract role.

Решение: `delete candidate` после проверки всего связанного island.

### Placeholder

Файл содержит только TODO, stub или будущую идею без runtime consumer.

Решение: перенести намерение в ADR/issue/documentation или удалить. Пустая
архитектурная заготовка не должна жить в `app/`.

### Test-only implementation

Реальный reusable code импортируется только тестами.

Решение: `needs owner decision`. Это может быть забытый production path,
устаревшая реализация или намеренная test utility.

### Package-export only

Symbol re-exported из `__init__.py`, но не используется внутри runtime graph.

Решение: проверить, является ли export поддерживаемым internal/public API.
Re-export не доказывает жизненность автоматически.

### Orphaned island

Группа modules импортирует друг друга, но не имеет входящего runtime/tool
reference.

Решение: анализировать и удалять island целиком только после owner decision.

## Audit checklist

Перед удалением проверяются:

- `rg` по пути, module name и exported symbols;
- прямые и package imports;
- `__all__` и re-exports;
- CLI dispatch и argparse registration;
- subprocess argv, filesystem paths и string-based loading;
- Alembic configuration и revision chain;
- SQLAlchemy model metadata;
- tests, fixtures и `conftest.py`;
- README, docs, ADR и normative contracts;
- systemd unit, environment names и release files;
- compatibility с checkpoint/database/wire artifacts.

Нужно учитывать reflection: например, environment names строятся из dataclass
fields, а migrations обнаруживаются инструментом по directory structure.

## Generated и runtime artifacts

`__pycache__`, `*.pyc`, test cache, generated plots, local checkpoints и
runtime spool не являются source code. Они не должны попадать в repository и
не используются как доказательство актуальности исходников.

Удаление production artifacts (`models/`, `recovery/`, `/tmp/transformer`,
PostgreSQL data) не является code cleanup и требует отдельного явного решения.

## Правило удаления

Deletion pass ограничивается заранее перечисленными paths.

Запрещено одновременно:

- менять runtime behavior;
- перестраивать package architecture;
- обновлять зависимости;
- менять public contract;
- удалять owner-decision candidates;
- добавлять broad cleanup automation.

После удаления запускаются относящиеся тесты, затем при необходимости полный
pytest и `git diff --check`. При сомнении файл остаётся до отдельного решения.
