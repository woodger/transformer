import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import io

from app.arrow_io import (
    iter_framed_arrow,
    read_arrow,
    read_source_arrow,
    table_to_source_tensor,
    table_to_tensors,
    write_arrow,
)


def test_arrow_read_write(tmp_path):
    # ---- create fake arrow file ----
    X = [[1.0, 2.0], [3.0, 4.0]]
    Y = [[0.5], [1.5]]

    table = pa.table({"src": X, "tgt": Y})
    path = tmp_path / "data.arrow"

    with pa.OSFile(str(path), "wb") as sink:  # <--- str(path)
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    X_t, Y_t = read_arrow(str(path))  # read_arrow тоже должен принимать str

    assert isinstance(X_t, torch.Tensor)
    assert isinstance(Y_t, torch.Tensor)
    assert X_t.shape == (2, 2)
    assert Y_t.shape == (2, 1)

    # ---- write predictions ----
    preds = torch.randn(2, 1)
    out_path = tmp_path / "preds.arrow"

    write_arrow(str(out_path), preds, "out")  # <--- str(out_path)
    assert out_path.exists()


def test_iter_framed_arrow_reads_multiple_payloads():
    tables = [
        pa.table({"src": [[1.0, 2.0]], "tgt": [[0.5]]}),
        pa.table({"src": [[3.0, 4.0]], "tgt": [[1.5]]}),
    ]

    stream = io.BytesIO()
    for table in tables:
        sink = pa.BufferOutputStream()
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

        payload = sink.getvalue().to_pybytes()
        stream.write(len(payload).to_bytes(8, byteorder="big", signed=False))
        stream.write(payload)

    stream.seek(0)
    result = list(iter_framed_arrow(stream))

    assert len(result) == 2
    X_t, Y_t = table_to_tensors(result[1])
    assert isinstance(X_t, torch.Tensor)
    assert X_t.tolist() == [[3.0, 4.0]]
    assert Y_t.tolist() == [[1.5]]


def test_read_source_arrow_does_not_require_target(tmp_path):
    table = pa.table({"src": [[1.0, 2.0], [3.0, 4.0]]})
    path = tmp_path / "predict.arrow"

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    X_t = read_source_arrow(str(path))

    assert X_t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_table_to_source_tensor_rejects_inconsistent_src_width():
    table = pa.table({"src": [[1.0, 2.0], [3.0]]})

    try:
        table_to_source_tensor(table)
    except ValueError as exc:
        assert "inconsistent list length" in str(exc)
    else:
        raise AssertionError("accepted inconsistent src width")
