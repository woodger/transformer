# PyArrow Flight 24 dependency note

## Scope

This note records two confirmed limitations of `pyarrow==24.0.0` that affect
the Transformer Flight v2 contract. They are limitations of the Python Flight
server binding, not defects in `arrow-flight-client@0.0.8` or Inventory.

Environment used to reproduce the behavior:

```text
pyarrow 24.0.0
Arrow C++ 24.0.0
bundled gRPC C++ 1.71.0
```

## Missing server status codes

### Symptom

The Python binding can emit these exact transport statuses:

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

It does not expose a server exception or status constructor for
`ALREADY_EXISTS`, `FAILED_PRECONDITION`, or `RESOURCE_EXHAUSTED`.
`FileExistsError`, `ArrowCapacityError`, and `ArrowMemoryError` are transported
as a generic `FlightServerError` (`UNKNOWN`). Mapping a failed precondition or
conflict to `ArrowInvalid` produces `INVALID_ARGUMENT`, not the required code.

### Minimal reproduction

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

Actual output:

```text
capacity FlightServerError
exists FlightServerError
```

There is also no `FlightAlreadyExistsError`, failed-precondition exception, or
resource-exhausted exception in `pyarrow.flight`.

### Expected result and migration impact

Flight v2 requires the actual failing RPC to carry the normative gRPC code; an
error encoded in a successful JSON result is not acceptable. Consequently the
pure-Python service cannot pass the exact `ALREADY_EXISTS`,
`FAILED_PRECONDITION`, and `RESOURCE_EXHAUSTED` wire-status gate. Stable
application codes can still be included in safe error messages and in terminal
job status, but they do not replace the transport code.

### Safe temporary behavior

- Continue to fail the RPC; never return an error inside a successful result.
- Preserve the stable application code in the safe exception message.
- Use the closest non-success PyArrow exception and document the resulting
  wire-status mismatch.
- Do not use reserved gRPC trailers, private Cython symbols, or an in-process
  second `grpcio` runtime to forge a status.

### Proposed upstream API

PyArrow should expose either dedicated exceptions for all standard Flight
transport statuses or a public constructor such as:

```python
flight.FlightStatusError(
    code=flight.TransportStatusCode.FAILED_PRECONDITION,
    message="job is not sealed",
    extra_info=b"",
)
```

The API must cover at least `ALREADY_EXISTS`, `FAILED_PRECONDITION`, and
`RESOURCE_EXHAUSTED` and preserve the exact gRPC status for non-PyArrow clients.

## Server receive-message limit

### Symptom

Arrow Flight 24's gRPC server initializes the transport with:

```cpp
builder.SetMaxReceiveMessageSize(-1);
```

The C++ `FlightServerOptions::builder_hook` can customize the server builder,
but `pyarrow.flight.FlightServerBase` does not expose `builder_hook`,
`generic_options`, `maxReceiveMessageLength`, or an equivalent option.

Minimal Python reproduction:

```python
import pyarrow.flight as flight

flight.FlightServerBase(
    ("127.0.0.1", 0),
    generic_options=[("grpc.max_receive_message_length", 16 * 1024 * 1024)],
)
```

Actual result:

```text
TypeError: __init__() got an unexpected keyword argument 'generic_options'
```

### Expected result and migration impact

Transformer should be able to enforce and advertise a 16 MiB transport receive
limit while accepting approximately 8 MiB client RecordBatches. With PyArrow
24, payloads larger than gRPC's historical 4 MiB default work because receive
size is unlimited, but application validation runs only after gRPC has already
allocated the incoming message. Application batch, payload, row, job and queue
quotas therefore do not provide the same pre-allocation protection.

### Safe temporary behavior

- Inventory configures `maxSendMessageLength` and `maxReceiveMessageLength` and
  targets RecordBatches of about 8 MiB.
- Transformer enforces batch, logical payload, total job and row limits while
  reading each chunk.
- Capabilities describe 16 MiB as the interoperability target, not as a hard
  PyArrow server transport limit.
- A trusted TLS/network boundary may add an independent request-size policy,
  but it must not be presented as a PyArrow handler guarantee.

### Proposed upstream API

Expose narrowly scoped server options, for example:

```python
flight.FlightServerBase(
    location,
    max_receive_message_bytes=16 * 1024 * 1024,
    max_send_message_bytes=16 * 1024 * 1024,
)
```

Alternatively expose a supported `generic_options` or `builder_hook` binding.
The selected values must be observable so capabilities can report effective,
not merely configured, limits.

## Compatibility and scope impact

Closing either gap requires an upstream PyArrow binding change, a separately
maintained native extension, or a future dependency upgrade whose behavior is
verified against the Node client. A private local patch or custom Cython shim
would materially expand the Transformer migration scope and deployment matrix.
No change to Inventory or `arrow-flight-client@0.0.8` is proposed.
