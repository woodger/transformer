import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.contracts.flight.v21.arrow import canonical_input_schema
from app.contracts.flight.v21.target_value_error import TargetValueError
from app.contracts.semantic.v4 import ModelContract
from app.worker.data.arrow import (
    iter_committed_fit_arrow,
    iter_committed_source_arrow,
    write_arrow,
)
from tests.fixture_documents import semantic_fixture_document

SOURCE_ENCODING = {
    "featureBlocks": [
        {"windowRows": 2, "nativeRowWidth": 2},
    ],
}
TARGET_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("single-regression")["modelContract"],
).target_contract


def test_prediction_artifact_preserves_public_values(tmp_path):
    predictions = torch.tensor([
        [1.0],
        [-1.0],
    ])
    out_path = tmp_path / "preds.arrow"

    write_arrow(str(out_path), predictions, "out", TARGET_CONTRACT)

    with pa.memory_map(str(out_path), "r") as source:
        written = ipc.RecordBatchFileReader(source).read_all()
    assert written.column("out").to_pylist() == predictions.tolist()


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


def test_committed_arrow_replay_rechecks_binary_target_values(tmp_path):
    contract = ModelContract.from_document(
        semantic_fixture_document("positive-class-weighted-binary-w1")[
            "modelContract"
        ]
    )
    schema = canonical_input_schema(
        "fit",
        SOURCE_ENCODING,
        seq_len=1,
        feature_dim=4,
        target_contract=contract.target_contract,
    )
    batch = pa.record_batch(
        [
            pa.array([0], type=schema.field(0).type),
            pa.array([0], type=schema.field(1).type),
            pa.array([{
                "b0": {
                    "nativeRows": [[1.0, 2.0], [3.0, 4.0]],
                    "observationOffsets": [[0]],
                },
            }], type=schema.field(2).type),
            pa.array([[[0.5]]], type=schema.field(3).type),
        ],
        schema=schema,
    )
    path = tmp_path / "committed-binary-fit.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_batch(batch)

    with pytest.raises(TargetValueError) as error:
        list(iter_committed_fit_arrow(
            str(path),
            expected_rows=1,
            expected_chunks=1,
            expected_native_rows=(2,),
            source_encoding=SOURCE_ENCODING,
            seq_len=1,
            feature_dim=4,
            target_contract=contract.target_contract,
            binary_target_indices=contract.weighted_binary_target_indices,
        ))

    assert error.value.expected_domain == "Binary"


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
                "featureBlocks": [
                    {"windowRows": 1, "nativeRowWidth": 5},
                ],
            },
            seq_len=1,
            feature_dim=5,
            target_contract=TARGET_CONTRACT,
        ))
