import io

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.data.arrow import (
    iter_framed_arrow,
    read_arrow,
    read_source_arrow,
    table_to_source_tensor,
    table_to_tensors,
    write_arrow,
)
from app.worker.data.arrow import (
    read_committed_fit_arrow,
    read_committed_source_arrow,
)


def test_arrow_file_io_preserves_tensor_values(tmp_path):
    source_rows = [[1.0, 2.0], [3.0, 4.0]]
    target_rows = [
        [0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
        [-0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
    ]

    table = pa.table({"src": source_rows, "tgt": target_rows})
    path = tmp_path / "data.arrow"

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    features, targets = read_arrow(str(path))

    assert isinstance(features, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert features.tolist() == source_rows
    assert torch.allclose(
        targets,
        torch.tensor(target_rows, dtype=torch.float32),
    )

    predictions = torch.tensor([
        [1.0, 0.2, 0.3, 0.4, 0.5, 0.6],
        [-1.0, 0.5, 0.4, 0.3, 0.2, 0.25],
    ])
    out_path = tmp_path / "preds.arrow"

    write_arrow(str(out_path), predictions, "out")

    with pa.memory_map(str(out_path), "r") as source:
        written = ipc.RecordBatchFileReader(source).read_all()
    assert written.column("out").to_pylist() == predictions.tolist()


def test_iter_framed_arrow_reads_multiple_payloads():
    tables = [
        pa.table({
            "src": [[1.0, 2.0]],
            "tgt": [[0.5, 0.0, 0.0, 0.0, 1.0, 1.0]],
        }),
        pa.table({
            "src": [[3.0, 4.0]],
            "tgt": [[-0.5, 0.1, 0.2, 0.3, 0.4, 0.5]],
        }),
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
    assert torch.allclose(
        Y_t,
        torch.tensor([[-0.5, 0.1, 0.2, 0.3, 0.4, 0.5]]),
    )


def test_read_source_arrow_does_not_require_target(tmp_path):
    table = pa.table({"src": [[1.0, 2.0], [3.0, 4.0]]})
    path = tmp_path / "predict.arrow"

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    X_t = read_source_arrow(str(path))

    assert X_t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_committed_arrow_replay_preserves_validated_fit_values(tmp_path):
    schema = pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("tgt", pa.list_(pa.float32(), 6), nullable=False),
    ])
    batches = [
        pa.record_batch(
            [
                pa.array([[1.0, float("nan"), 3.0, 4.0]], type=schema.field(0).type),
                pa.array([[0.5, 0.0, 0.0, 0.0, 1.0, 1.0]], type=schema.field(1).type),
            ],
            schema=schema,
        ),
        pa.record_batch(
            [
                pa.array([[5.0, 6.0, 7.0, 8.0]], type=schema.field(0).type),
                pa.array(
                    [[-0.5, 0.1, 0.2, 0.3, 0.4, 0.0]],
                    type=schema.field(1).type,
                ),
            ],
            schema=schema,
        ),
    ]
    path = tmp_path / "committed-fit.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)

    source, target = read_committed_fit_arrow(
        str(path),
        expected_rows=2,
        source_width=4,
    )

    assert source.shape == (2, 4)
    assert torch.isnan(source[0, 1])
    assert source[1].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert torch.allclose(target, torch.tensor([
        [0.5, 0.0, 0.0, 0.0, 1.0, 1.0],
        [-0.5, 0.1, 0.2, 0.3, 0.4, 0.0],
    ]))


def test_committed_arrow_replay_rechecks_receipt_shape(tmp_path):
    schema = pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
    ])
    table = pa.Table.from_arrays(
        [pa.array([[1.0, 2.0, 3.0, 4.0]], type=schema.field(0).type)],
        schema=schema,
    )
    path = tmp_path / "committed-predict.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)

    with pytest.raises(ValueError, match=r"row count 1.*receipt row count 2"):
        read_committed_source_arrow(
            str(path),
            expected_rows=2,
            source_width=4,
        )

    with pytest.raises(ValueError, match="physical schema"):
        read_committed_source_arrow(
            str(path),
            expected_rows=1,
            source_width=5,
        )


def test_table_to_source_tensor_rejects_inconsistent_src_width():
    table = pa.table({"src": [[1.0, 2.0], [3.0]]})

    try:
        table_to_source_tensor(table)
    except ValueError as exc:
        assert "inconsistent list length" in str(exc)
    else:
        raise AssertionError("accepted inconsistent src width")
