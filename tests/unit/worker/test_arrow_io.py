import io

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.contracts.flight.v13.arrow import canonical_input_schema
from app.worker.data.arrow import (
    iter_committed_fit_arrow,
    iter_committed_source_arrow,
    iter_framed_arrow,
    read_arrow,
    read_source_arrow,
    table_to_source_tensor,
    table_to_tensors,
    write_arrow,
)
from tests.support.consumer_neutral import model_contract

SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 2, "nativeRowWidth": 2},
    ],
}
TARGET_CONTRACT = model_contract("single-regression").target_contract


def test_arrow_file_io_preserves_tensor_values(tmp_path):
    source_rows = [[1.0, 2.0], [3.0, 4.0]]
    target_rows = [
        [0.5],
        [-0.5],
    ]

    table = pa.table({"src": source_rows, "tgt": target_rows})
    path = tmp_path / "data.arrow"

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    batch = read_arrow(str(path), TARGET_CONTRACT)

    assert isinstance(batch.features, torch.Tensor)
    assert isinstance(batch.targets, torch.Tensor)
    assert batch.features.tolist() == source_rows
    assert torch.allclose(
        batch.targets,
        torch.tensor(target_rows, dtype=torch.float32),
    )

    predictions = torch.tensor([
        [1.0],
        [-1.0],
    ])
    out_path = tmp_path / "preds.arrow"

    write_arrow(str(out_path), predictions, "out", TARGET_CONTRACT)

    with pa.memory_map(str(out_path), "r") as source:
        written = ipc.RecordBatchFileReader(source).read_all()
    assert written.column("out").to_pylist() == predictions.tolist()


def test_iter_framed_arrow_reads_multiple_payloads():
    tables = [
        pa.table({
            "src": [[1.0, 2.0]],
            "tgt": [[0.5]],
        }),
        pa.table({
            "src": [[3.0, 4.0]],
            "tgt": [[-0.5]],
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
    batch = table_to_tensors(result[1], TARGET_CONTRACT)
    assert isinstance(batch.features, torch.Tensor)
    assert batch.features.tolist() == [[3.0, 4.0]]
    assert torch.allclose(
        batch.targets,
        torch.tensor([[-0.5]]),
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
    schema = canonical_input_schema(
        "fit",
        SOURCE_ENCODING,
        seq_len=1,
        feature_dim=4,
        target_contract=TARGET_CONTRACT,
    )
    batches = [
        pa.record_batch(
            [
                pa.array([0], type=schema.field(0).type),
                pa.array([0], type=schema.field(1).type),
                pa.array([{
                    "b0": {
                        "nativeRows": [[1.0, float("nan")], [3.0, 4.0]],
                        "observationOffsets": [[0]],
                    },
                }], type=schema.field(2).type),
                pa.array([[[0.5]]], type=schema.field(3).type),
            ],
            schema=schema,
        ),
        pa.record_batch(
            [
                pa.array([0], type=schema.field(0).type),
                pa.array([1], type=schema.field(1).type),
                pa.array([{
                    "b0": {
                        "nativeRows": [[5.0, 6.0], [7.0, 8.0]],
                        "observationOffsets": [[0]],
                    },
                }], type=schema.field(2).type),
                pa.array([[[-0.5]]], type=schema.field(3).type),
            ],
            schema=schema,
        ),
    ]
    path = tmp_path / "committed-fit.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)

    batches = list(iter_committed_fit_arrow(
        str(path),
        expected_rows=2,
        expected_chunks=2,
        expected_native_rows=(4,),
        source_encoding=SOURCE_ENCODING,
        seq_len=1,
        feature_dim=4,
        target_contract=TARGET_CONTRACT,
    ))
    features = torch.cat([batch.features for batch in batches])
    targets = torch.cat([batch.targets for batch in batches])

    assert features.shape == (2, 1, 4)
    assert torch.isnan(features[0, 0, 1])
    assert features[1, 0].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert torch.allclose(targets, torch.tensor([
        [0.5],
        [-0.5],
    ]))


def test_committed_arrow_replay_rechecks_receipt_shape(tmp_path):
    schema = canonical_input_schema(
        "predict",
        SOURCE_ENCODING,
        seq_len=1,
        feature_dim=4,
        target_contract=TARGET_CONTRACT,
    )
    table = pa.Table.from_arrays([
        pa.array([0], type=schema.field(0).type),
        pa.array([0], type=schema.field(1).type),
        pa.array([{
            "b0": {
                "nativeRows": [[1.0, 2.0], [3.0, 4.0]],
                "observationOffsets": [[0]],
            },
        }], type=schema.field(2).type),
    ],
        schema=schema,
    )
    path = tmp_path / "committed-predict.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)

    with pytest.raises(ValueError, match=r"row count 1.*receipt row count 2"):
        list(iter_committed_source_arrow(
            str(path),
            expected_rows=2,
            expected_chunks=1,
            expected_native_rows=(2,),
            source_encoding=SOURCE_ENCODING,
            seq_len=1,
            feature_dim=4,
            target_contract=TARGET_CONTRACT,
        ))

    with pytest.raises(ValueError, match="physical schema"):
        list(iter_committed_source_arrow(
            str(path),
            expected_rows=1,
            expected_chunks=1,
            expected_native_rows=(1,),
            source_encoding={
                "kind": "indexedFeatureBlocks",
                "featureBlocks": [
                    {"position": 0, "windowRows": 1, "nativeRowWidth": 5},
                ],
            },
            seq_len=1,
            feature_dim=5,
            target_contract=TARGET_CONTRACT,
        ))


def test_table_to_source_tensor_rejects_inconsistent_src_width():
    table = pa.table({"src": [[1.0, 2.0], [3.0]]})

    try:
        table_to_source_tensor(table)
    except ValueError as exc:
        assert "inconsistent list length" in str(exc)
    else:
        raise AssertionError("accepted inconsistent src width")
