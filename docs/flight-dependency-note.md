# Ограничения зависимости PyArrow Flight 24

## Область действия

Эта заметка фиксирует два подтверждённых ограничения `pyarrow==24.0.0`,
влияющих на контракт Transformer Flight v5. Это ограничения Python binding
сервера Flight, а не дефекты `arrow-flight-client@0.0.8` или Inventory.

Окружение, в котором воспроизведено поведение:

```text
pyarrow 24.0.0
Arrow C++ 24.0.0
bundled gRPC C++ 1.71.0
```

## Отсутствующие server status codes

### Проявление

Python binding может отправить следующие точные transport statuses:

| Python exception | gRPC status |
| --- | --- |
| `ValueError` / `pyarrow.ArrowInvalid` | `INVALID_ARGUMENT` |
| `KeyError` / `pyarrow.ArrowKeyError` | `NOT_FOUND` |
| `NotImplementedError` | `UNIMPLEMENTED` |
| `FlightUnauthenticatedError` | `UNAUTHENTICATED` |
| `FlightUnauthorizedError` | `PERMISSION_DENIED` |
| `FlightCancelledError` | `CANCELLED` |
| `FlightTimedOutError` | `DEADLINE_EXCEEDED` |
| `FlightInternalError` | `INTERNAL` |
| `FlightUnavailableError` | `UNAVAILABLE` |

Он не предоставляет server exception или status constructor для
`ALREADY_EXISTS`, `FAILED_PRECONDITION` и `RESOURCE_EXHAUSTED`.
`FileExistsError`, `ArrowCapacityError` и `ArrowMemoryError` передаются как
общий `FlightServerError` (`UNKNOWN`). Mapping failed precondition или conflict
в `ArrowInvalid` даёт `INVALID_ARGUMENT`, а не требуемый code.

### Минимальное воспроизведение

```python
import pyarrow as pa
import pyarrow.flight as flight

errors = {
    "capacity": pa.ArrowCapacityError("full"),
    "exists": FileExistsError("exists"),
}

class Server(flight.FlightServerBase):
    def do_action(self, context, action):
        raise errors[action.type]

with Server(("127.0.0.1", 0)) as server:
    with flight.connect(("127.0.0.1", server.port)) as client:
        for name in errors:
            try:
                list(client.do_action(flight.Action(name, b"")))
            except Exception as error:
                print(name, type(error).__name__)
```

Фактический output:

```text
capacity FlightServerError
exists FlightServerError
```

В `pyarrow.flight` также отсутствуют `FlightAlreadyExistsError`, exception для
failed precondition и exception для resource exhausted.

### Ожидаемый результат и влияние на contract

Flight v5 требует, чтобы ошибочный RPC содержал нормативный gRPC code;
кодирование ошибки внутри успешного JSON result неприемлемо. Поэтому service
на чистом Python не может пройти wire-status gate с точными
`ALREADY_EXISTS`, `FAILED_PRECONDITION` и `RESOURCE_EXHAUSTED`. Стабильные
application codes по-прежнему можно включать в безопасные сообщения и terminal
status job, но они не заменяют transport code.

### Безопасное временное поведение

- Завершать RPC ошибкой, не возвращая ошибку внутри успешного result.
- Сохранять стабильный application code в безопасном сообщении exception.
- Использовать ближайший неуспешный PyArrow exception и документировать
  расхождение wire status.
- Не использовать зарезервированные gRPC trailers, private Cython symbols или
  второй in-process runtime `grpcio` для подделки status.

### Предлагаемый upstream API

PyArrow должен предоставить отдельные exceptions для всех стандартных Flight
transport statuses либо публичный constructor, например:

```python
flight.FlightStatusError(
    code=flight.TransportStatusCode.FAILED_PRECONDITION,
    message="job input is not closed",
    extra_info=b"",
)
```

API должен покрывать как минимум `ALREADY_EXISTS`, `FAILED_PRECONDITION` и
`RESOURCE_EXHAUSTED` и сохранять точный gRPC status для клиентов не на PyArrow.

## Лимит принимаемого server message

### Проявление

Server gRPC из Arrow Flight 24 инициализирует transport так:

```cpp
builder.SetMaxReceiveMessageSize(-1);
```

В C++ `FlightServerOptions::builder_hook` позволяет настроить server builder,
но `pyarrow.flight.FlightServerBase` не предоставляет `builder_hook`,
`generic_options`, `maxReceiveMessageLength` или эквивалентный option.

Минимальное воспроизведение на Python:

```python
import pyarrow.flight as flight

flight.FlightServerBase(
    ("127.0.0.1", 0),
    generic_options=[("grpc.max_receive_message_length", 16 * 1024 * 1024)],
)
```

Фактический результат:

```text
TypeError: __init__() got an unexpected keyword argument 'generic_options'
```

### Ожидаемый результат и влияние на contract

Transformer должен иметь возможность применить и сообщить transport limit
приёма 16 MiB, принимая клиентские RecordBatches примерно по 8 MiB. В PyArrow
24 payload-ы больше исторического значения gRPC 4 MiB работают, поскольку
receive size не ограничен, но application validation выполняется только после
того, как gRPC уже выделил память под входящее сообщение. Поэтому квоты batch,
logical payload, rows, job и queue на уровне приложения не дают такую же
защиту до выделения памяти.

### Безопасное временное поведение

- Inventory настраивает `maxSendMessageLength` и `maxReceiveMessageLength` и
  формирует RecordBatches примерно по 8 MiB.
- Transformer применяет лимиты batch, logical payload, total job и rows при
  чтении каждого chunk.
- Capabilities описывают 16 MiB как цель interoperability, а не как жёсткий
  transport limit server-а PyArrow.
- Доверенная граница TLS/network может добавить независимую policy размера
  request, но её нельзя представлять как гарантию handler-а PyArrow.

### Предлагаемый upstream API

Предоставить server options узкой области действия, например:

```python
flight.FlightServerBase(
    location,
    max_receive_message_bytes=16 * 1024 * 1024,
    max_send_message_bytes=16 * 1024 * 1024,
)
```

Альтернативой является поддерживаемый binding `generic_options` или
`builder_hook`. Выбранные значения должны быть наблюдаемыми, чтобы capabilities
сообщали фактические, а не только настроенные limits.

## Влияние на compatibility и scope

Для закрытия любого из gaps требуется upstream change binding PyArrow,
отдельно поддерживаемое native extension или будущее обновление dependency,
поведение которого проверено с Node client. Private local patch или custom
Cython shim существенно расширили бы scope Transformer и deployment matrix.
Изменения Inventory или `arrow-flight-client@0.0.8` не предлагаются.
