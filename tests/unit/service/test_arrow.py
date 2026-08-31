import math

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from app.contracts.flight.v7.arrow import canonical_input_schema
from app.service.adapters.inbound.flight.arrow import (
    InputBatchValidator,
    schema_fingerprint,
    validate_prediction_file,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode

TARGETS = (
    "MeanReturn", "SigmaReturn", "ProbTP", "ProbSL",
    "VolatilityNext", "HittingProbTP",
)


def fit_schema(source_width=4):
    return pa.schema([
        pa.field(
            "src",
            pa.list_(pa.float32(), source_width),
            nullable=False,
        ),
        pa.field(
            "tgt",
            pa.list_(pa.float32(), 6),
            nullable=False,
        ),
    ])


def fit_batch(source_rows, *, source_width=4):
    targets = [[0.0, 0.1, 0.0, 0.0, 0.2, 1.0] for _ in source_rows]
    schema = fit_schema(source_width)
    return pa.record_batch(
        [
            pa.array(source_rows, type=schema.field("src").type),
            pa.array(targets, type=schema.field("tgt").type),
        ],
        schema=schema,
    )


def test_multi_batch_input_is_validated_as_one_logical_payload():
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        targets=TARGETS,
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


def test_record_batch_must_keep_the_doput_fixed_schema():
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        targets=TARGETS,
        seq_len=2,
        expected_feature_dim=None,
        max_batch_bytes=1024 * 1024,
        max_payload_bytes=2 * 1024 * 1024,
        max_rows=10,
    )
    validator.validate_batch(fit_batch([[1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(ServiceError, match="schema differs"):
        validator.validate_batch(
            fit_batch([[1.0, 2.0]], source_width=2)
        )


@pytest.mark.parametrize(
    ("operation", "column"),
    [
        ("fit", "src"),
        ("fit", "tgt"),
        ("predict", "src"),
    ],
)
def test_input_schema_rejects_noncanonical_nested_nullability(
    operation,
    column,
):
    schema = canonical_input_schema(operation, 4)
    fields = list(schema)
    index = schema.get_field_index(column)
    width = schema.field(index).type.list_size
    fields[index] = pa.field(
        column,
        pa.list_(
            pa.field("item", pa.float32(), nullable=False),
            width,
        ),
        nullable=False,
    )

    with pytest.raises(ServiceError) as error:
        InputBatchValidator(
            operation,
            pa.schema(fields),
            targets=TARGETS,
            seq_len=2,
            expected_feature_dim=2,
            max_batch_bytes=1024,
            max_payload_bytes=2048,
            max_rows=10,
        )

    assert error.value.code is ErrorCode.INVALID_ARGUMENT
    assert "canonical physical schema" in error.value.message


def test_payload_quota_is_enforced_across_batches_before_full_staging():
    first = fit_batch([[1.0, 2.0, 3.0, 4.0]])
    validator = InputBatchValidator(
        "fit",
        fit_schema(),
        targets=TARGETS,
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
            targets=TARGETS,
            seq_len=2,
            expected_feature_dim=2,
            max_batch_bytes=1024,
            max_payload_bytes=2048,
            max_rows=10,
        )

    schema = pa.schema([
        pa.field("src", pa.list_(pa.float32(), 6), nullable=False),
    ])
    with pytest.raises(ServiceError, match="does not match model feature dimension"):
        InputBatchValidator(
            "predict",
            schema,
            targets=TARGETS,
            seq_len=2,
            expected_feature_dim=2,
            max_batch_bytes=1024,
            max_payload_bytes=2048,
            max_rows=10,
        )


def write_table(path, table):
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=1)


def test_prediction_file_stream_validation_supports_typed_empty(tmp_path):
    path = tmp_path / "empty.arrow"
    output_type = pa.list_(pa.float32(), 6)
    table = pa.Table.from_arrays(
        [pa.array([], type=output_type)],
        schema=pa.schema([
            pa.field("out", output_type, nullable=False),
        ]),
    )
    write_table(path, table)

    stats = validate_prediction_file(
        str(path), "out", expected_rows=0, targets=TARGETS
    )

    assert stats.rows == 0
    assert stats.batches == 0


def test_prediction_file_rejects_noncanonical_nested_nullability(tmp_path):
    path = tmp_path / "nested-nonnullable.arrow"
    output_type = pa.list_(
        pa.field("item", pa.float32(), nullable=False),
        6,
    )
    table = pa.Table.from_arrays(
        [pa.array([], type=output_type)],
        schema=pa.schema([
            pa.field("out", output_type, nullable=False),
        ]),
    )
    write_table(path, table)

    with pytest.raises(ServiceError, match="FixedSizeList<float32>"):
        validate_prediction_file(
            str(path), "out", expected_rows=0, targets=TARGETS
        )


def test_prediction_file_rejects_nonfinite_values_and_wrong_rows(tmp_path):
    path = tmp_path / "bad.arrow"
    table = pa.table({
        "out": pa.array(
            [[0.0, 1.0, 2.0, 3.0, 4.0, math.inf]],
            type=pa.list_(pa.float32(), 6),
        )
    })
    table = table.cast(pa.schema([
        pa.field("out", pa.list_(pa.float32(), 6), nullable=False),
    ]))
    write_table(path, table)
    with pytest.raises(ServiceError, match="non-finite"):
        validate_prediction_file(
            str(path), "out", expected_rows=1, targets=TARGETS
        )

    write_table(path, pa.table({
        "out": pa.array([[0.0] * 6], type=pa.list_(pa.float32(), 6))
    }).cast(pa.schema([
        pa.field("out", pa.list_(pa.float32(), 6), nullable=False),
    ])))
    with pytest.raises(ServiceError, match="row count 1 does not match"):
        validate_prediction_file(
            str(path), "out", expected_rows=2, targets=TARGETS
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            pa.array([None], type=pa.list_(pa.float32(), 6)),
            "null row",
        ),
        (
            pa.array(
                [[0.0, 1.0, None, 3.0, 4.0, 5.0]],
                type=pa.list_(pa.float32(), 6),
            ),
            "non-finite or null value",
        ),
        (
            pa.array([[0.0] * 5], type=pa.list_(pa.float32(), 5)),
            "FixedSizeList<float32>",
        ),
    ],
)
def test_prediction_file_rejects_invalid_list_structure(
    tmp_path,
    values,
    message,
):
    path = tmp_path / "invalid-structure.arrow"
    table = pa.Table.from_arrays(
        [values],
        schema=pa.schema([
            pa.field("out", values.type, nullable=False),
        ]),
    )
    write_table(path, table)

    with pytest.raises(ServiceError, match=message):
        validate_prediction_file(
            str(path), "out", expected_rows=1, targets=TARGETS
        )


def test_schema_fingerprint_ignores_nonsemantic_metadata():
    plain = canonical_input_schema("predict", 4)
    annotated = plain.with_metadata({b"producer": b"inventory"})
    nested_annotated = pa.schema([
        pa.field(
            "src",
            pa.list_(
                pa.field(
                    "item",
                    pa.float32(),
                    nullable=True,
                    metadata={b"producer": b"inventory"},
                ),
                4,
            ),
            nullable=False,
        ),
    ])
    assert schema_fingerprint(plain) == schema_fingerprint(annotated)
    assert schema_fingerprint(plain) == schema_fingerprint(nested_annotated)
