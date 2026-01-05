import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch


def read_arrow(path):
    with open(path, "rb") as f:
        reader = ipc.RecordBatchFileReader(f)
        table = reader.read_all()

    X = np.stack(table.column("src").to_pylist()).astype(np.float32)
    Y = np.stack(table.column("tgt").to_pylist()).astype(np.float32)

    return torch.from_numpy(X), torch.from_numpy(Y)


def write_arrow(path: str, preds: torch.Tensor, col_name: str):
    # preds: (N, k) → list of lists
    arr = preds.cpu().numpy()
    col = arr.tolist()
    table = pa.table({col_name: col})

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
