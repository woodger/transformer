import math
from copy import deepcopy

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from app.contracts.flight.v16.arrow import (
    canonical_input_schema,
    validate_target_values,
)
from app.contracts.flight.v16.target_value_error import TargetValueError
from app.contracts.semantic.v3 import ModelContract
from app.service.adapters.inbound.flight.arrow import (
    InputBatchValidator,
    schema_fingerprint,
    validate_prediction_file,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from tests.fixture_documents import semantic_fixture_document

MODEL_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("single-regression")["modelContract"],
)
TARGET_CONTRACT = MODEL_CONTRACT.target_contract
SOURCE_ENCODING = {
    "featureBlocks": [
        {"windowRows": 2, "nativeRowWidth": 1},
        {"windowRows": 1, "nativeRowWidth": 2},
    ],
}


def input_schema(operation="fit"):
    return canonical_input_schema(
        operation,
        SOURCE_ENCODING,
        seq_len=2,
        feature_dim=4,
        target_contract=TARGET_CONTRACT,
    )


def compact_batch(*, operation="fit", range_ordinal=0, example_offset=0):
    schema = input_schema(operation)
    columns = [
        pa.array([range_ordinal], type=schema.field("rangeOrdinal").type),
        pa.array([example_offset], type=schema.field("exampleOffset").type),
        pa.array([{
            "b0": {
                "nativeRows": [[1.0], [2.0], [3.0]],
                "observationOffsets": [[0, 1]],
            },
            "b1": {
                "nativeRows": [[4.0, 5.0], [6.0, 7.0]],
                "observationOffsets": [[0, 1]],
            },
        }], type=schema.field("features").type),
    ]
    if operation == "fit":
        columns.append(pa.array(
            [[[0.1]]],
            type=schema.field("tgt").type,
        ))
    return pa.record_batch(columns, schema=schema)


def validator(operation="fit", *, schema=None, max_payload_bytes=2 * 1024 * 1024):
    return InputBatchValidator(
        operation,
        input_schema(operation) if schema is None else schema,
        source_encoding=SOURCE_ENCODING,
        target_contract=TARGET_CONTRACT,
        seq_len=2,
        expected_feature_dim=4,
        max_batch_bytes=1024 * 1024,
        max_payload_bytes=max_payload_bytes,
        max_rows=10,
    )


def test_multi_batch_input_is_validated_as_one_logical_payload():
    payload = validator()

    payload.validate_batch(compact_batch())
    payload.validate_batch(compact_batch(example_offset=1))

    stats = payload.stats()
    assert stats.rows == 2
    assert stats.chunks == 2
    assert stats.native_rows == (6, 4)
    assert stats.batches == 2
    assert (stats.first_range_ordinal, stats.first_example_offset) == (0, 0)
    assert (stats.last_range_ordinal, stats.next_example_offset) == (0, 2)


def test_record_batch_must_keep_the_doput_fixed_schema():
    payload = validator()
    payload.validate_batch(compact_batch())

    with pytest.raises(ServiceError, match="schema differs"):
        payload.validate_batch(compact_batch(operation="predict"))


@pytest.mark.parametrize(
    "operation",
    ["fit", "predict"],
)
def test_input_schema_rejects_noncanonical_nullability(operation):
    schema = input_schema(operation)
    fields = list(schema)
    fields[0] = pa.field(
        "rangeOrdinal",
        pa.uint32(),
        nullable=True,
    )

    with pytest.raises(ServiceError) as error:
        validator(operation, schema=pa.schema(fields))

    assert error.value.code is ErrorCode.INVALID_ARGUMENT
    assert "canonical physical schema" in error.value.message


def test_payload_quota_is_enforced_across_batches_before_full_staging():
    first = compact_batch()
    payload = validator(max_payload_bytes=first.nbytes)
    payload.validate_batch(first)

    with pytest.raises(ServiceError, match="payload data size exceeds limit"):
        payload.validate_batch(first)


def test_predict_schema_rejects_fit_schema_and_feature_mismatch():
    with pytest.raises(ServiceError, match="canonical physical schema"):
        validator("predict", schema=input_schema("fit"))

    with pytest.raises(ValueError, match=r"must equal tensorGeometry\.featureDim"):
        InputBatchValidator(
            "predict",
            input_schema("predict"),
            source_encoding=SOURCE_ENCODING,
            target_contract=TARGET_CONTRACT,
            seq_len=2,
            expected_feature_dim=5,
            max_batch_bytes=1024,
            max_payload_bytes=2048,
            max_rows=10,
        )


def test_compact_input_rejects_noncontiguous_and_out_of_bounds_offsets():
    payload = validator()
    payload.validate_batch(compact_batch())
    with pytest.raises(ServiceError, match="exampleOffset is not contiguous"):
        payload.validate_batch(compact_batch(example_offset=2))

    schema = input_schema()
    invalid = compact_batch()
    columns = [invalid.column(index) for index in range(invalid.num_columns)]
    columns[2] = pa.array([{
        "b0": {
            "nativeRows": [[1.0], [2.0]],
            "observationOffsets": [[0, 1]],
        },
        "b1": {
            "nativeRows": [[4.0, 5.0], [6.0, 7.0]],
            "observationOffsets": [[0, 1]],
        },
    }], type=schema.field("features").type)
    with pytest.raises(ServiceError, match="outside range chunk"):
        validator().validate_batch(pa.record_batch(columns, schema=schema))


def write_table(path, table):
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=1)


def test_prediction_file_stream_validation_supports_typed_empty(tmp_path):
    path = tmp_path / "empty.arrow"
    output_type = pa.list_(pa.float32(), 1)
    table = pa.Table.from_arrays(
        [pa.array([], type=output_type)],
        schema=pa.schema([
            pa.field("out", output_type, nullable=False),
        ]),
    )
    write_table(path, table)

    stats = validate_prediction_file(
        str(path), "out", expected_rows=0, target_contract=TARGET_CONTRACT
    )

    assert stats.rows == 0
    assert stats.batches == 0


def test_prediction_file_rejects_noncanonical_nested_nullability(tmp_path):
    path = tmp_path / "nested-nonnullable.arrow"
    output_type = pa.list_(
        pa.field("item", pa.float32(), nullable=False),
        1,
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
            str(path), "out", expected_rows=0, target_contract=TARGET_CONTRACT
        )


def test_prediction_file_rejects_nonfinite_values_and_wrong_rows(tmp_path):
    path = tmp_path / "bad.arrow"
    table = pa.table({
        "out": pa.array(
            [[math.inf]],
            type=pa.list_(pa.float32(), 1),
        )
    })
    table = table.cast(pa.schema([
        pa.field("out", pa.list_(pa.float32(), 1), nullable=False),
    ]))
    write_table(path, table)
    with pytest.raises(ServiceError, match="non-finite"):
        validate_prediction_file(
            str(path), "out", expected_rows=1, target_contract=TARGET_CONTRACT
        )

    write_table(path, pa.table({
        "out": pa.array([[0.0]], type=pa.list_(pa.float32(), 1))
    }).cast(pa.schema([
        pa.field("out", pa.list_(pa.float32(), 1), nullable=False),
    ])))
    with pytest.raises(ServiceError, match="row count 1 does not match"):
        validate_prediction_file(
            str(path), "out", expected_rows=2, target_contract=TARGET_CONTRACT
        )


def test_closed_interval_compares_exact_float32_value_to_binary64_bound():
    target_contract = deepcopy(TARGET_CONTRACT)
    target_contract["slots"][0]["observedConstraint"]["maximum"] = 1.0000001
    values = np.array([[1.0000001192092896]], dtype=np.float32)

    with pytest.raises(TargetValueError):
        validate_target_values(values, target_contract)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            pa.array([None], type=pa.list_(pa.float32(), 1)),
            "null row",
        ),
        (
            pa.array(
                [[None]],
                type=pa.list_(pa.float32(), 1),
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
            str(path),
            "out",
            expected_rows=1,
            target_contract=TARGET_CONTRACT,
        )


def test_schema_fingerprint_ignores_nonsemantic_metadata():
    plain = input_schema("predict")
    annotated = plain.with_metadata({b"producer": b"inventory"})
    fields = list(plain)
    features = plain.field("features")
    fields[2] = pa.field(
        "features",
        pa.struct([
            pa.field(
                field.name,
                field.type,
                nullable=field.nullable,
                metadata={b"producer": b"inventory"},
            )
            for field in features.type
        ]),
        nullable=False,
    )
    nested_annotated = pa.schema(fields)
    assert schema_fingerprint(plain) == schema_fingerprint(annotated)
    assert schema_fingerprint(plain) == schema_fingerprint(nested_annotated)
