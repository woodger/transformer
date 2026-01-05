import torch
import pyarrow as pa
import pyarrow.ipc as ipc

from app.arrow_io import read_arrow, write_arrow


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
