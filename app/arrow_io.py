import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch


FRAME_HEADER_BYTES = 8


def table_to_tensors(table):
    X = np.stack(table.column("src").to_pylist()).astype(np.float32)
    Y = np.stack(table.column("tgt").to_pylist()).astype(np.float32)

    return torch.from_numpy(X), torch.from_numpy(Y)


def read_arrow(path):
    with open(path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    return table_to_tensors(table)


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
    # preds: (N, k) → list of lists
    arr = preds.cpu().numpy()
    col = arr.tolist()
    table = pa.table({col_name: col})

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
