import math

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from app.flight.arrow import (
    InputBatchValidator,
    schema_fingerprint,
    validate_prediction_file,
)
from app.flight.errors import ServiceError


def fit_schema():
    return pa.schema([
        ("src", pa.list_(pa.float32())),
        ("tgt", pa.list_(pa.float32())),
    ])


def fit_batch(source_rows):
    targets = [[0.0, 0.1, 0.0, 0.0, 0.2, 1.0] for _ in source_rows]
    return pa.record_batch(
        [
            pa.array(source_rows, type=pa.list_(pa.float32())),
            pa.array(targets, type=pa.list_(pa.float32())),
        ],
        schema=fit_schema(),
    )


def test_multi_batch_input_is_validated_as_one_logical_payload():
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        seq_len=2,
        expected_feature_dim=None,
        max_batch_bytes=1024 * 1024,
        max_payload_bytes=2 * 1024 * 1024,
        max_rows=10,
    )

    validator.validate_batch(fit_batch([[1.0, 2.0, 3.0, 4.0]]))
    validator.validate_batch(fit_batch([[5.0, 6.0, 7.0, 8.0]]))

    stats = validator.stats()
    assert stats.rows == 2
    assert stats.batches == 2
    assert stats.src_width == 4
    assert stats.tgt_width == 6


def test_width_is_consistent_between_record_batches():
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        seq_len=2,
        expected_feature_dim=None,
        max_batch_bytes=1024 * 1024,
        max_payload_bytes=2 * 1024 * 1024,
        max_rows=10,
    )
    validator.validate_batch(fit_batch([[1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(ServiceError, match="inconsistent list width across batches"):
        validator.validate_batch(fit_batch([[1.0, 2.0]]))


def test_payload_quota_is_enforced_across_batches_before_full_staging():
    first = fit_batch([[1.0, 2.0, 3.0, 4.0]])
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        seq_len=2,
        expected_feature_dim=None,
        max_batch_bytes=1024,
        max_payload_bytes=first.nbytes,
        max_rows=10,
    )
    validator.validate_batch(first)

    with pytest.raises(ServiceError, match="payload data size exceeds limit"):
        validator.validate_batch(first)


def test_predict_schema_rejects_tgt_and_expected_feature_mismatch():
    with pytest.raises(ServiceError, match="columns must be exactly src"):
        InputBatchValidator(
            "predict",
            fit_schema(),
            seq_len=2,
            expected_feature_dim=2,
            max_batch_bytes=1024,
            max_payload_bytes=2048,
            max_rows=10,
        )

    schema = pa.schema([("src", pa.list_(pa.float32()))])
    validator = InputBatchValidator(
        "predict",
        schema,
        seq_len=2,
        expected_feature_dim=2,
        max_batch_bytes=1024,
        max_payload_bytes=2048,
        max_rows=10,
    )
    batch = pa.record_batch(
        [pa.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], type=schema.field(0).type)],
        schema=schema,
    )
    with pytest.raises(ServiceError, match="does not match model feature dimension"):
        validator.validate_batch(batch)


def write_table(path, table):
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=1)


def test_prediction_file_stream_validation_supports_typed_empty(tmp_path):
    path = tmp_path / "empty.arrow"
    table = pa.table({"out": pa.array([], type=pa.list_(pa.float32()))})
    write_table(path, table)

    stats = validate_prediction_file(str(path), "out", expected_rows=0)

    assert stats.rows == 0
    assert stats.batches == 0


def test_prediction_file_rejects_nonfinite_values_and_wrong_rows(tmp_path):
    path = tmp_path / "bad.arrow"
    table = pa.table({
        "out": pa.array(
            [[0.0, 1.0, 2.0, 3.0, 4.0, math.inf]],
            type=pa.list_(pa.float32()),
        )
    })
    write_table(path, table)
    with pytest.raises(ServiceError, match="non-finite"):
        validate_prediction_file(str(path), "out", expected_rows=1)

    write_table(path, pa.table({
        "out": pa.array([[0.0] * 6], type=pa.list_(pa.float32()))
    }))
    with pytest.raises(ServiceError, match="row count 1 does not match"):
        validate_prediction_file(str(path), "out", expected_rows=2)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            pa.array([None], type=pa.list_(pa.float32())),
            "null row",
        ),
        (
            pa.array(
                [[0.0, 1.0, None, 3.0, 4.0, 5.0]],
                type=pa.list_(pa.float32()),
            ),
            "null value",
        ),
        (
            pa.array([[0.0] * 5], type=pa.list_(pa.float32())),
            "list width must be 6, got 5",
        ),
    ],
)
def test_prediction_file_rejects_invalid_list_structure(
    tmp_path,
    values,
    message,
):
    path = tmp_path / "invalid-structure.arrow"
    write_table(path, pa.table({"out": values}))

    with pytest.raises(ServiceError, match=message):
        validate_prediction_file(str(path), "out", expected_rows=1)


def test_schema_fingerprint_ignores_nonsemantic_metadata():
    plain = pa.schema([("src", pa.list_(pa.float32()))])
    annotated = plain.with_metadata({b"producer": b"inventory"})
    assert schema_fingerprint(plain) == schema_fingerprint(annotated)
