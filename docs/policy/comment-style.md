# Политика комментариев

> Type: Policy. Этот документ задаёт правила для Python comments и docstrings.

Комментарий объясняет причину решения, инвариант или внешнее ограничение.
Он не должен пересказывать код, хранить историю изменений или компенсировать
неясное имя.

Основной вопрос:

> Почему это решение существует именно в таком виде?

## Docstrings

Docstring нужен, когда public или reusable component имеет неочевидный
контракт:

- lifecycle и ownership ресурса;
- side effects;
- допустимые состояния и ошибки;
- filesystem, database или subprocess guarantees;
- wire/checkpoint compatibility;
- shape, dtype или numerical constraints.

Docstring обычно не нужен для простого function/class, смысл которого полностью
виден из имени, type hints и короткой реализации.

Хорошо:

```python
class RuntimeDirectoryLock:
    """Exclusive process-lifetime ownership of the Flight runtime directory."""
```

Плохо:

```python
def load_config():
    """Loads config."""
```

Module-level docstring добавляется, только если модуль задаёт самостоятельную
границу, которую нельзя понять из имени и структуры. Обязательные
«архитектурные шапки» для каждого production-файла не используются.

## Inline comments

Комментарий оправдан, если фиксирует:

- порядок операций, важный для atomicity или race safety;
- причину unusual validation или quota;
- ограничение PyArrow, PyTorch, PostgreSQL или ОС;
- compatibility с legacy checkpoint или external contract;
- причину намеренно пустой ветки;
- безопасный shutdown/recovery порядок;
- numerical stability или masking invariant.

Хорошо:

```python
# Revision pagination keeps input listing below the action-document limit.
# A fixed page bound also makes consumer traversal memory predictable.
MAX_PAGE_ITEMS = 100
```

Хорошо:

```python
# Close the queue-claim boundary before RPC shutdown so a racing start can
# remain durable QUEUED but cannot become RUNNING.
worker.stop_claiming()
```

Хорошо:

```python
# Publish the checkpoint before committing model metadata. A modelRef must
# never resolve to a file that failed to reach persistent storage.
publish_checkpoint()
commit_model_metadata()
```

Плохо:

```python
# Increment revision.
revision += 1
```

Плохо:

```python
# Check whether CUDA is available.
if torch.cuda.is_available():
    ...
```

## Комментарий не расширяет контракт

Гарантия в комментарии должна подтверждаться кодом и, для существенного
инварианта, тестом. Нельзя обещать:

- retry safety без idempotency record;
- durability для данных в `/tmp`;
- deterministic training без соответствующей настройки PyTorch;
- atomic publication без `fsync`/rename/transaction sequence;
- поддержку checkpoint, которую reader не проверяет.

Если контракт изменился, комментарий обновляется в том же change set.

## TODO

TODO допустим, только если содержит:

- текущее ограничение;
- причину отложенного изменения;
- условие возврата к задаче.

Хорошо:

```python
# TODO: configure a transport receive limit when the PyArrow server binding
# exposes it. Until then the application rejects oversized batches itself.
```

Плохо:

```python
# TODO: fix later
```

Placeholder без runtime consumer лучше хранить в issue или roadmap, а не в
пустом Python module.

## Workarounds

Workaround должен называть внешнюю причину и границу действия:

```python
# FlightServerBase.serve() blocks inside a C extension, so it runs in a thread
# to let the Python main thread dispatch SIGTERM promptly.
server_thread.start()
```

Комментарий `# Run in thread` в этом месте не объясняет решение.

## Язык и терминология

В пределах файла используется один основной язык. Stable external names
(`modelRef`, `DoPut`, `SIGTERM`, `storage epoch`) сохраняются без искусственного
перевода. Формулировки должны соответствовать
[Политике именования](./naming-policy.md).

## Комментарии в тестах

В тесте комментарий нужен только для неочевидного production-risk, fixture или
expected value. Он не должен размечать `arrange/act/assert` и дублировать имя
теста.

Сначала следует улучшить имя теста и данные. Комментарий добавляется только
если причина всё равно не видна.

## Проверка при review

- можно ли удалить комментарий без потери причины решения;
- соответствует ли он текущему коду;
- защищена ли существенная гарантия тестом;
- не хранит ли он историю, план или очевидную механику;
- не скрывает ли слишком широкую ответственность функции.
