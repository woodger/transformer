import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch


FRAME_HEADER_BYTES = 8


def table_to_tensors(table):
    validate_arrow_table(table, require_target=True)
    X = _list_column_to_tensor(table, "src")
    Y = _list_column_to_tensor(table, "tgt")

    return X, Y


def table_to_source_tensor(table):
    validate_arrow_table(table, require_target=False)
    return _list_column_to_tensor(table, "src")


def validate_arrow_table(table, require_target: bool = False):
    _validate_list_column(table, "src")
    if require_target:
        _validate_list_column(table, "tgt")


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


def _validate_list_column(table, name: str):
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Arrow table must contain '{name}' column")

    column_type = table.schema.field(column_index).type
    if not (
        pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
        or pa.types.is_fixed_size_list(column_type)
    ):
        raise ValueError(f"Arrow column '{name}' must be a list<float> column")


def _list_column_to_tensor(table, name: str):
    values = table.column(name).to_pylist()
    if not values:
        return torch.empty((0, 0), dtype=torch.float32)

    width = None
    for row_index, row in enumerate(values, start=1):
        if row is None:
            raise ValueError(f"Arrow column '{name}' has null row at index {row_index}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"Arrow column '{name}' has inconsistent list length at row {row_index}"
            )

    return torch.from_numpy(np.asarray(values, dtype=np.float32))


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


def iter_framed_arrow(stream):
    while True:
        header = stream.read(FRAME_HEADER_BYTES)

        if header == b"":
            return

        if len(header) != FRAME_HEADER_BYTES:
            raise EOFError("Incomplete framed Arrow header")

        payload_size = int.from_bytes(header, byteorder="big", signed=False)
        if payload_size == 0:
            return

        payload = _read_exact(stream, payload_size)
        reader = ipc.RecordBatchFileReader(pa.BufferReader(payload))

        yield reader.read_all()


def write_arrow(path: str, preds: torch.Tensor, col_name: str):
    table = predictions_to_table(preds, col_name)

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def predictions_to_table(preds: torch.Tensor, col_name: str):
    arr = preds.cpu().numpy()
    col = arr.tolist()

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
