# Политика именования

> Type: Policy. Этот документ задаёт правила именования Python-сущностей,
> файлов и внешних contracts.

Имена должны отражать роль и ownership сущности. Новое имя следует ближайшей
устойчивой конвенции проекта, а не личному стилю автора.

## Python

Используется стиль PEP 8:

- `snake_case` — modules, functions, methods, variables и parameters;
- `PascalCase` — classes, exceptions, enums и значимые type aliases;
- `UPPER_SNAKE_CASE` — module-level constants;
- `_leading_underscore` — внутренние symbols без публичного контракта;
- `test_<behavior>` — pytest test functions;
- `test_<subject>.py` — test modules.

Допустимо:

```python
DEFAULT_DEVICE = "cpu"
MAX_MANIFEST_ITEMS = 400


class FlightServiceConfig:
    pass


def validate_feature_dim(actual_dim, expected_dim):
    pass
```

`UPPER_SNAKE_CASE` используется для действительно постоянных значений,
contract identifiers и immutable lookup tables. Обычная локальная переменная
не становится константой только потому, что сейчас не переназначается.

Встроенные magic methods и сторонние API могут требовать имена с двойным
подчёркиванием. Собственные «псевдомагические» имена создавать не следует.

## Файлы и директории

Python modules и packages используют `snake_case`:

```text
app/worker/training/losses.py
app/service/adapters/outbound/postgres/token_cache.py
tests/unit/worker/test_checkpoint_storage.py
```

Markdown-документы используют устойчивые существующие имена, для новых
постоянных документов предпочтителен английский `kebab-case`.

Имя `utils.py`, `helpers.py`, `common.py` или аналогичное не описывает
ответственность. Новый generic container не создаётся без архитектурного
обоснования.

## Внешние контракты

Внешнее имя сохраняется дословно, даже если оно не соответствует Python style:

- environment variables — `POSTGRES_HOST`, `TRANSFORMER_MAX_PAYLOAD_BYTES`;
- CLI commands/options — `flight serve`, `fit-stream`, `--tls-cert-file`;
- Flight JSON fields — `requestId`, `modelRef`, `maxPayloadBytes`;
- machine error/state codes — `INVALID_ARGUMENT`, `RUNNING`;
- SQL tables и columns — `snake_case`;
- Arrow columns и schema IDs — в форме нормативного contract.

Внутри Python external field не нужно распространять дальше boundary. Если
полезно, transport mapping преобразует `camelCase` в `snake_case`, сохраняя
wire representation неизменным.

Пример:

```python
response = {
    "requestId": request_id,
    "modelRef": model_ref,
}
```

Здесь ключи являются Flight contract, а `request_id` и `model_ref` —
внутренними Python names.

## Семантическая точность

- `host` и `port` используются последовательно для listen endpoint;
- `runtime_dir` не называется `state_dir`, если содержит payload и spool;
- `model_ref` означает opaque published model identifier, а не filesystem path;
- `payload`, `batch`, `frame` и `job` не взаимозаменяются;
- `checkpoint`, `model metadata` и `model alias` называются отдельно;
- `config`, `data`, `result`, `item` допустимы только при ясном локальном
  контексте.

Имена функций должны выражать действие, boolean values — проверяемое свойство,
exceptions — причину отказа.

## Переименования

Политика не является основанием для массового переименования существующего
кода. Перед переименованием проверяются:

- Python imports и package exports;
- CLI, environment, JSON, SQL и Arrow contracts;
- subprocess arguments и dynamic string references;
- tests, fixtures и documentation;
- checkpoint и database compatibility.

Переименование выполняется отдельной минимальной правкой, если оно затрагивает
публичное или persisted имя.
