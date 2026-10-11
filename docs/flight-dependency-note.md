# Ограничения PyArrow Flight

> Тип: справочник. Неподдерживаемые статусы ошибок и настройка размера сообщений.

PyArrow не позволяет серверу передать некоторые статусы ошибок и задать
максимальный размер входящего сообщения. Здесь описано, как эти ограничения
влияют на Transformer и что учитывать при подключении клиента. Поведение
подтверждено для `pyarrow==24.0.0`. Ограничения относятся к Python-интерфейсу
сервера Flight; исправления в `arrow-flight-client@0.0.8` или вызывающей
системе для них не предлагаются.

Окружение, в котором воспроизведено поведение:

```text
pyarrow 24.0.0
Arrow C++ 24.0.0
bundled gRPC C++ 1.71.0
```

## Неподдерживаемые статусы ошибок

### Проявление

Сервер на Python может передать следующие статусы gRPC:

| Исключение Python | Статус gRPC |
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

Но PyArrow не предоставляет серверные исключения или конструктор статуса для
`ALREADY_EXISTS`, `FAILED_PRECONDITION` и `RESOURCE_EXHAUSTED`.
`FileExistsError`, `ArrowCapacityError` и `ArrowMemoryError` передаются как
общий `FlightServerError` (`UNKNOWN`). Замена ошибки конфликта или невыполненного
условия на `ArrowInvalid` даёт `INVALID_ARGUMENT`, а не требуемый статус.

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

Фактический вывод:

```text
capacity FlightServerError
exists FlightServerError
```

В `pyarrow.flight` также отсутствуют `FlightAlreadyExistsError` и исключения
для невыполненного условия или исчерпания ресурсов.

### Что это означает для клиента

По спецификации Flight v24 ошибочный RPC завершается ошибкой. Сообщение
содержит стабильный код приложения, а ошибки семантики запросов, независимые
от предметной области, — также структурированное описание в
`FlightError.extra_info`. Точные статусы gRPC `ALREADY_EXISTS`,
`FAILED_PRECONDITION` и `RESOURCE_EXHAUSTED` на чистом Python недоступны.
Клиент выбирает обработку по коду приложения и структурированному описанию,
а не пытается определить недоступный статус по классу исключения PyArrow.

### Безопасное временное поведение

- Завершать RPC ошибкой; не помещать её внутрь успешного результата.
- Сохранять стабильный код приложения в сообщении исключения без секретов.
- Использовать ближайшее подходящее исключение PyArrow и описывать расхождение
  с требуемым статусом gRPC.
- Не подделывать статус через зарезервированные служебные поля gRPC, приватные
  символы Cython или параллельный запуск `grpcio` внутри процесса.

### Какое изменение нужно в PyArrow

Нужны отдельные исключения для всех стандартных статусов Flight или публичный
конструктор статуса. Например, такой API:

```python
flight.FlightStatusError(
    code=flight.TransportStatusCode.FAILED_PRECONDITION,
    message="job input is not closed",
    extra_info=b"",
)
```

Он должен покрывать как минимум `ALREADY_EXISTS`, `FAILED_PRECONDITION` и
`RESOURCE_EXHAUSTED` и сохранять точный статус gRPC для клиентов, которые
не используют PyArrow.

## Размер входящего сообщения

### Проявление

Сервер gRPC в Arrow Flight 24 задаёт размер входящего сообщения так:

```cpp
builder.SetMaxReceiveMessageSize(-1);
```

В C++ `FlightServerOptions::builder_hook` позволяет настроить сервер,
но `pyarrow.flight.FlightServerBase` не предоставляет `builder_hook`,
`generic_options`, `maxReceiveMessageLength` или аналогичный параметр.

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

### Что это означает для защиты памяти

Transformer должен иметь возможность ограничить приём сообщения размером
16 MiB и сообщить этот предел клиенту, принимая его RecordBatches примерно
по 8 MiB. В PyArrow 24 сообщения больше прежнего лимита gRPC 4 MiB проходят,
поскольку размер приёма не ограничен. Проверки приложения выполняются уже
после того, как gRPC выделил память под сообщение. Поэтому лимиты батча,
переданного пакета данных, логических строк, задания и очереди не защищают
от выделения этой памяти.

### Безопасное временное поведение

- Клиент настраивает `maxSendMessageLength` и `maxReceiveMessageLength` и
  формирует RecordBatches примерно по 8 MiB.
- Transformer проверяет размер батча, переданного пакета данных, всего задания
  и число логических строк при чтении каждого чанка.
- Capabilities описывают 16 MiB как размер для согласованной работы клиента
  и сервиса. Это не жёсткое ограничение приёма на сервере PyArrow.
- Доверенная сетевая инфраструктура или узел завершения TLS могут независимо
  ограничивать размер запроса. Такая настройка не становится гарантией
  обработчика PyArrow.

### Какое изменение нужно в PyArrow

Нужны параметры сервера для ограничения размера сообщений. Например:

```python
flight.FlightServerBase(
    location,
    max_receive_message_bytes=16 * 1024 * 1024,
    max_send_message_bytes=16 * 1024 * 1024,
)
```

Другой вариант — поддержка `generic_options` или `builder_hook` в Python.
Сервер должен позволять проверить применённые значения, чтобы capabilities
сообщали фактические ограничения.

## Что потребуется для исправления

Каждое из этих ограничений требует изменения в PyArrow, отдельно
поддерживаемого нативного расширения или будущего обновления зависимости,
проверенного с клиентом на Node.js. Собственная заплатка PyArrow или обёртка
на Cython добавили бы Transformer работу по сопровождению и проверке новых
вариантов развёртывания. Изменения вызывающих приложений или
`arrow-flight-client@0.0.8` здесь не предлагаются.
