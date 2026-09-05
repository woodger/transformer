from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.flight.v11.prediction_stats import PredictionArrowStats
from app.contracts.flight.v11.target_value_error import TargetValueError
from app.contracts.indexed_feature_blocks import feature_block_dimensions


class _NumpyConvertible(Protocol):
    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray: ...


class _FlatPredictionArray(Protocol):
    def is_null(self) -> _NumpyConvertible: ...

    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray: ...


class _PredictionArray(Protocol):
    def is_null(self) -> _NumpyConvertible: ...

    def flatten(self) -> _FlatPredictionArray: ...


class _PredictionBatch(Protocol):
    @property
    def num_rows(self) -> int: ...

    def column(self, index: int) -> _PredictionArray: ...


class _PredictionReader(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    @property
    def num_record_batches(self) -> int: ...

    def get_batch(self, index: int) -> _PredictionBatch: ...


def validate_prediction_file(
    path: str,
    prediction_column: str,
    expected_rows: int,
    target_contract: Mapping[str, object],
) -> PredictionArrowStats:
    width = target_width(target_contract)
    rows = 0
    batches = 0
    with pa.memory_map(path, "r") as source:
        reader = cast(_PredictionReader, ipc.RecordBatchFileReader(source))
        schema = reader.schema
        _validate_prediction_schema(schema, prediction_column, width)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            array = batch.column(0)
            null_row = _first_true(array.is_null().to_numpy(zero_copy_only=False))
            if null_row is not None:
                raise ValueError(
                    f"prediction has null row at index {rows + null_row + 1}"
                )
            flat = array.flatten()
            values = flat.to_numpy(zero_copy_only=False)
            invalid = np.logical_or(
                flat.is_null().to_numpy(zero_copy_only=False),
                ~np.isfinite(values),
            )
            first_invalid = _first_true(invalid)
            if first_invalid is not None:
                raise ValueError(
                    "prediction has non-finite or null value at row "
                    f"{rows + first_invalid // width + 1}"
                )
            validate_target_values(
                values.reshape(-1, width),
                target_contract,
                logical_row_offset=rows,
            )
            rows += batch.num_rows
            batches += 1
    if rows != expected_rows:
        raise ValueError(
            f"prediction row count {rows} does not match input row count "
            f"{expected_rows}"
        )
    return PredictionArrowStats(rows, batches, schema_fingerprint(schema))


def canonical_input_schema(
    operation: str,
    source_encoding: Mapping[str, object],
    seq_len: int,
    feature_dim: int,
    target_contract: Mapping[str, object],
) -> pa.Schema:
    if operation not in ("fit", "predict"):
        raise ValueError("operation must be fit or predict")
    if type(seq_len) is not int or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    blocks = feature_block_dimensions(source_encoding, feature_dim=feature_dim)
    fields = [
        pa.field("rangeOrdinal", pa.uint32(), nullable=False),
        pa.field("exampleOffset", pa.uint64(), nullable=False),
        pa.field(
            "features",
            pa.struct([
                pa.field(
                    f"b{index}",
                    pa.struct([
                        pa.field(
                            "nativeRows",
                            pa.list_(_fixed_size_float32(native_row_width)),
                            nullable=False,
                        ),
                        pa.field(
                            "observationOffsets",
                            pa.list_(_fixed_size_uint32(seq_len)),
                            nullable=False,
                        ),
                    ]),
                    nullable=False,
                )
                for index, (_, _, native_row_width) in enumerate(blocks)
            ]),
            nullable=False,
        ),
    ]
    if operation == "fit":
        fields.append(
            pa.field(
                "tgt",
                pa.list_(_fixed_size_float32(target_width(target_contract))),
                nullable=False,
            )
        )
    return pa.schema(fields)


def canonical_prediction_schema(
    prediction_column: str,
    target_contract: Mapping[str, object],
) -> pa.Schema:
    if type(prediction_column) is not str or not prediction_column:
        raise ValueError("prediction_column must be a non-empty string")
    return pa.schema([
        pa.field(
            prediction_column,
            _fixed_size_float32(target_width(target_contract)),
            nullable=False,
        )
    ])


def schema_fingerprint(schema: pa.Schema) -> str:
    canonical = pa.schema([
        pa.field(
            field.name,
            _type_without_metadata(field.type),
            nullable=field.nullable,
        )
        for field in schema
    ])
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


def target_slots(
    target_contract: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    slots = target_contract.get("slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ValueError("targetContract.slots must be an array")
    result: list[Mapping[str, object]] = []
    for slot in cast(Sequence[object], slots):
        if not isinstance(slot, Mapping):
            raise ValueError("targetContract slot must be an object")
        result.append(cast(Mapping[str, object], slot))
    if not result:
        raise ValueError("targetContract.slots must not be empty")
    return tuple(result)


def target_width(target_contract: Mapping[str, object]) -> int:
    return len(target_slots(target_contract))


def validate_target_values(
    values: np.ndarray,
    target_contract: Mapping[str, object],
    *,
    logical_row_offset: int = 0,
) -> None:
    slots = target_slots(target_contract)
    if values.ndim != 2 or values.shape[1] != len(slots):
        raise ValueError(f"target values must have width {len(slots)}")
    for index, slot in enumerate(slots):
        identity = cast(str, slot["identity"])
        column = values[:, index]
        invalid = ~np.isfinite(column)
        constraint = cast(Mapping[str, object], slot["observedConstraint"])
        if constraint["kind"] == "ClosedInterval":
            minimum = float(cast(int | float, constraint["minimum"]))
            maximum = float(cast(int | float, constraint["maximum"]))
            exact_values = column.astype(np.float64, copy=False)
            invalid = (
                invalid
                | (exact_values < minimum)
                | (exact_values > maximum)
            )
        row = _first_true(invalid)
        if row is not None:
            logical_row = logical_row_offset + row
            raise TargetValueError(
                identity,
                index,
                logical_row,
                f"target {identity} is invalid at logical row {logical_row}",
            )


def _validate_prediction_schema(
    schema: pa.Schema,
    column: str,
    width: int,
) -> None:
    expected = pa.schema([
        pa.field(column, _fixed_size_float32(width), nullable=False)
    ])
    if schema.names != expected.names:
        raise ValueError(f"prediction output must contain exactly column {column!r}")
    field = schema.field(column)
    if field.type != expected.field(column).type:
        raise ValueError(
            f"prediction output must use FixedSizeList<float32>[{width}]"
        )
    if field.nullable:
        raise ValueError("prediction output column must be non-nullable")


def _fixed_size_float32(width: int) -> pa.FixedSizeListType:
    return pa.list_(pa.field("item", pa.float32(), nullable=True), width)


def _fixed_size_uint32(width: int) -> pa.FixedSizeListType:
    return pa.list_(pa.field("item", pa.uint32(), nullable=True), width)


def _type_without_metadata(value_type: pa.DataType) -> pa.DataType:
    if pa.types.is_struct(value_type):
        return pa.struct([
            pa.field(
                field.name,
                _type_without_metadata(field.type),
                nullable=field.nullable,
            )
            for field in value_type
        ])
    if pa.types.is_list(value_type):
        value_field = value_type.value_field
        return pa.list_(pa.field(
            value_field.name,
            _type_without_metadata(value_field.type),
            nullable=value_field.nullable,
        ))
    if pa.types.is_fixed_size_list(value_type):
        value_field = value_type.value_field
        return pa.list_(
            pa.field(
                value_field.name,
                _type_without_metadata(value_field.type),
                nullable=value_field.nullable,
            ),
            value_type.list_size,
        )
    return value_type


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


__all__ = [
    "PredictionArrowStats",
    "TargetValueError",
    "canonical_input_schema",
    "canonical_prediction_schema",
    "schema_fingerprint",
    "target_slots",
    "target_width",
    "validate_prediction_file",
    "validate_target_values",
]
