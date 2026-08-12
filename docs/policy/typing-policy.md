# Политика типов и runtime-контрактов

> Тип: политика. Документ задаёт правила Python type hints, внутренних
> структур данных и tensor-контрактов Transformer.

Типы должны сокращать число предположений о данных между слоями и процессами.
Аннотация не заменяет runtime validation внешнего input, а runtime validation
не заменяет статическую проверку внутренних вызовов.

## Python type hints

Новая или изменяемая public/reusable функция production-кода указывает типы
параметров и результата. Для private функции аннотации обязательны, если она:

- принимает callback, iterator или вложенную структуру;
- пересекает module boundary;
- возвращает несколько семантически разных значений;
- участвует в model, training, Arrow, checkpoint или process contract.

Bare `dict`, `list`, `tuple` и `set` в аннотациях не используются. Тип должен
описывать элементы, например `dict[str, object]` или `tuple[float, ...]`.
`Any`, `cast`, `# type: ignore` и `# pyright: ignore` допустимы только на
неизбежной dynamic boundary с узкой причиной. Они не применяются для скрытия
ошибок или быстрого прохождения checker-а.

## Boundary и внутренние структуры

JSON, Arrow, CLI и checkpoint input сначала проверяется нормативным parser,
schema или validator. После boundary часто используемые значения переводятся
в типизированную внутреннюю форму.

- `dataclass(frozen=True, slots=True)` — immutable configuration и records;
- `NamedTuple` — компактный именованный tuple, когда требуется unpacking;
- `Protocol` — capability port, callback или iterator behavior;
- `Enum` — закрытое множество внутренних состояний;
- `TypedDict` — небольшая устойчивая Python-структура без отдельной
  нормативной schema.

Полный Flight или worker JSON Schema не дублируется вручную в `TypedDict`.
Versioned schema остаётся источником истины, а внутренний record создаётся
после её проверки.

Безымянный tuple допустим внутри короткого математического выражения. Если
значения проходят через несколько функций или имеют разную семантику, им
нужны именованные поля.

## Tensor-контракты

Public/reusable tensor API фиксирует в type hint и docstring:

- логическую форму, например `[batch, sequence, features]`;
- dtype;
- значение каждой неочевидной оси;
- device requirements, если они ограничены;
- mask semantics, включая смысл `True`;
- форму и семантику результата.

Python type checker видит `torch.Tensor`, но не доказывает shape, dtype или
значения. Поэтому на trust boundary выполняется явная runtime validation.
Некорректный внешний input отклоняется `ValueError` либо профильной contract
error. Обычный `assert` для этого не используется: он может быть удалён
режимом `python -O`.

Validation размещается в ближайшей точке, где данные становятся доверенными.
Она не должна без необходимости повторять полный NaN/range scan в каждом
batch или добавлять CUDA→CPU synchronization. Shape и dtype, уже доказанные
immutable Arrow receipt/checkpoint contract, повторно проверяются только там,
где это защищает самостоятельный reusable API.

Для target и prediction нормативны текущие semantic names из
[ADR 0007](../adr/0007-target-aligned-flight-v4.md). Короткие `x`, `y` и `p`
допустимы в локальной формуле; orchestration и boundary code использует
`features`, `targets`, `predictions`, `padding_mask` и другие смысловые имена.

## Pyright

Конфигурация находится только в `[tool.pyright]` файла `pyproject.toml`.
Текущий `strict` scope перечислен там явно и расширяется постепенно после
устранения diagnostics в следующем устойчивом module boundary.

Правила ratchet:

- новый strict scope добавляется только с нулевым baseline;
- существующий strict path не удаляется ради прохождения проверки;
- severity не ослабляется без отдельного обоснованного изменения политики;
- legacy compatibility facade не становится первым владельцем type contract;
- отсутствие module в strict scope не разрешает ухудшать его типы.

Стандартная команда:

```bash
./.venv/bin/pyright
```

Pyright запускается после Ruff и до относящихся pytest tests. Изменение только
документации не требует type check, если оно не меняет Python/config examples.

## Review

- сигнатура позволяет понять допустимые значения без чтения всей реализации;
- runtime boundary проверяет то, чего не может доказать type checker;
- tensor docstring согласован с Arrow/model/checkpoint contract;
- имена раскрывают семантику вне локальной формулы;
- новый record имеет одного владельца и не дублирует versioned schema;
- checker проходит без широкого `Any` и необоснованных suppressions.
