import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch

from app.storage.atomic import atomic_output_path


FRAME_HEADER_BYTES = 8
DEFAULT_MAX_FRAME_BYTES = 512 * 1024 * 1024
TARGET_WIDTH = 6
FLOAT32_MAX = float(np.finfo(np.float32).max)


def table_to_tensors(table):
    source_values, target_values = _validated_arrow_columns(
        table,
        require_target=True,
    )
    X = _list_values_to_tensor(source_values)
    Y = _list_values_to_tensor(target_values)

    return X, Y


def table_to_source_tensor(table):
    source_values, _ = _validated_arrow_columns(table, require_target=False)
    return _list_values_to_tensor(source_values)


def validate_arrow_table(table, require_target: bool = False):
    _validated_arrow_columns(table, require_target=require_target)


def _validated_arrow_columns(table, require_target: bool):
    source_values = _validate_list_column(table, "src", allow_nan=True)
    target_values = None
    if require_target:
        target_values = _validate_list_column(
            table,
            "tgt",
            allow_nan=False,
            expected_width=TARGET_WIDTH,
        )
        _validate_target_values(target_values)

    return source_values, target_values


def read_arrow(path):
    with open(path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    return table_to_tensors(table)


def read_source_arrow(path):
    with open(path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    return table_to_source_tensor(table)


def _validate_list_column(
    table,
    name: str,
    *,
    allow_nan: bool,
    expected_width: int | None = None,
):
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Arrow table must contain '{name}' column")

    column_type = table.schema.field(column_index).type
    if not (
        pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
        or pa.types.is_fixed_size_list(column_type)
    ):
        raise ValueError(
            f"Arrow column '{name}' must be a list<float32> or list<float64> column"
        )

    if column_type.value_type not in (pa.float32(), pa.float64()):
        raise ValueError(
            f"Arrow column '{name}' must be a list<float32> or list<float64> column"
        )

    values = table.column(name).to_pylist()
    width = column_type.list_size if pa.types.is_fixed_size_list(column_type) else None

    for row_index, row in enumerate(values, start=1):
        if row is None:
            raise ValueError(f"Arrow column '{name}' has null row at index {row_index}")

        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"Arrow column '{name}' has inconsistent list length at row {row_index}"
            )

        for value_index, value in enumerate(row, start=1):
            if value is None:
                raise ValueError(
                    f"Arrow column '{name}' has null element at row {row_index}, "
                    f"position {value_index}"
                )

            if np.isinf(value) or (not allow_nan and np.isnan(value)):
                raise ValueError(
                    f"Arrow column '{name}' has non-finite value at row {row_index}, "
                    f"position {value_index}"
                )
            if not np.isnan(value) and abs(value) > FLOAT32_MAX:
                raise ValueError(
                    f"Arrow column '{name}' has value outside float32 range at "
                    f"row {row_index}, position {value_index}"
                )

    if expected_width is not None and width is not None and width != expected_width:
        raise ValueError(
            f"Arrow column '{name}' must have list length {expected_width}, got {width}"
        )
    if name == "src" and values and width == 0:
        raise ValueError("Arrow column 'src' must have a positive list length")

    return values


def _list_values_to_tensor(values):
    if not values:
        return torch.empty((0, 0), dtype=torch.float32)

    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _validate_target_values(values):
    for row_index, row in enumerate(values, start=1):
        if row[4] < 0.0:
            raise ValueError(
                f"Arrow column 'tgt' has negative volatility at row {row_index}"
            )
        if row[5] < 0.0 or row[5] > 1.0:
            raise ValueError(
                f"Arrow column 'tgt' has hit probability outside [0, 1] "
                f"at row {row_index}"
            )


def _read_exact(stream, size):
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Unexpected end of framed Arrow stream")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def iter_framed_arrow(stream, max_frame_bytes=DEFAULT_MAX_FRAME_BYTES):
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be greater than zero")

    while True:
        header = stream.read(FRAME_HEADER_BYTES)

        if header == b"":
            return

        if len(header) != FRAME_HEADER_BYTES:
            raise EOFError("Incomplete framed Arrow header")

        payload_size = int.from_bytes(header, byteorder="big", signed=False)
        if payload_size == 0:
            return
        if payload_size > max_frame_bytes:
            raise ValueError(
                f"Framed Arrow payload size {payload_size} exceeds maximum "
                f"{max_frame_bytes} bytes"
            )

        payload = _read_exact(stream, payload_size)
        reader = ipc.RecordBatchFileReader(pa.BufferReader(payload))

        yield reader.read_all()


def write_arrow(
    path: str,
    preds: torch.Tensor,
    col_name: str,
    expected_rows: int | None = None,
):
    table = predictions_to_table(preds, col_name, expected_rows=expected_rows)

    with atomic_output_path(path) as temporary_path:
        with pa.OSFile(temporary_path, "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)


def predictions_to_table(
    preds: torch.Tensor,
    col_name: str,
    expected_rows: int | None = None,
):
    if preds.ndim != 2 or preds.shape[1] != TARGET_WIDTH:
        raise ValueError(
            f"Predictions must have shape [rows, {TARGET_WIDTH}], "
            f"got {list(preds.shape)}"
        )
    if expected_rows is not None and preds.shape[0] != expected_rows:
        raise ValueError(
            f"Predictions row count {preds.shape[0]} does not match input row "
            f"count {expected_rows}"
        )
    arr = preds.detach().cpu().to(dtype=torch.float32).numpy()
    if not np.isfinite(arr).all():
        raise ValueError("Predictions must contain only finite values")
    col = pa.array(arr.tolist(), type=pa.list_(pa.float32()))

    return pa.table({col_name: col})


def empty_predictions_table(col_name: str):
    return pa.table({col_name: pa.array([], type=pa.list_(pa.float32()))})


def write_framed_arrow(stream, table):
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    payload = sink.getvalue().to_pybytes()
    stream.write(
        len(payload).to_bytes(
            FRAME_HEADER_BYTES,
            byteorder="big",
            signed=False,
        )
    )
    stream.write(payload)
    stream.flush()
