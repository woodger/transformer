from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app import config as defaults
from app.contracts.flight.v11.arrow import (
    canonical_input_schema,
    canonical_prediction_schema,
    target_width,
    validate_target_values,
)
from app.contracts.indexed_feature_blocks import feature_block_dimensions
from app.contracts.json_types import JsonObject
from app.worker.checkpoints.atomic import atomic_output_path
from app.worker.data.tensors import TrainingBatch

if TYPE_CHECKING:
    import torch

FRAME_HEADER_BYTES = 8
FLOAT32_MAX = float(np.finfo(np.float32).max)


class _NumpyConvertible(Protocol):
    def to_numpy(self, *, zero_copy_only: bool) -> np.ndarray: ...


class _FlatArray(_NumpyConvertible, Protocol):
    def is_null(self) -> _NumpyConvertible: ...


class _ListArray(Protocol):
    def __len__(self) -> int: ...

    @property
    def offsets(self) -> _NumpyConvertible: ...

    def is_null(self) -> _NumpyConvertible: ...

    def flatten(self) -> _FlatArray: ...


class _ChunkedArray(Protocol):
    @property
    def chunks(self) -> Sequence[_ListArray]: ...


class _ArrowTable(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    @property
    def num_rows(self) -> int: ...

    def column(self, name: str | int) -> _ChunkedArray: ...


class _ArrowReader(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

    def read_all(self) -> pa.Table: ...

    @property
    def num_record_batches(self) -> int: ...

    def get_batch(self, index: int) -> pa.RecordBatch: ...


class _ArrowWriter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def write_table(self, table: pa.Table) -> None: ...


class _ListType(Protocol):
    @property
    def value_type(self) -> pa.DataType: ...

    @property
    def list_size(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _ValidatedArrowColumns:
    features: np.ndarray
    targets: np.ndarray | None


def table_to_tensors(
    table: pa.Table,
    target_contract: JsonObject,
) -> TrainingBatch:
    columns = _validated_arrow_columns(
        table,
        require_target=True,
        target_contract=target_contract,
    )
    features = _list_values_to_tensor(columns.features)
    if columns.targets is None:
        raise AssertionError("fit Arrow validation did not return targets")
    target_tensor = _list_values_to_tensor(columns.targets)

    return TrainingBatch(features=features, targets=target_tensor)


def table_to_source_tensor(table: pa.Table) -> torch.Tensor:
    columns = _validated_arrow_columns(
        table,
        require_target=False,
        target_contract=None,
    )
    return _list_values_to_tensor(columns.features)


def validate_arrow_table(
    table: pa.Table,
    require_target: bool = False,
    target_contract: JsonObject | None = None,
) -> None:
    _validated_arrow_columns(
        table,
        require_target=require_target,
        target_contract=target_contract,
    )


def _validated_arrow_columns(
    table: pa.Table,
    require_target: bool,
    target_contract: JsonObject | None,
) -> _ValidatedArrowColumns:
    typed_table = cast(_ArrowTable, table)
    source_values = _validate_list_column(
        typed_table,
        "src",
        allow_nan=True,
    )
    target_values = None
    if require_target:
        if target_contract is None:
            raise ValueError("target contract is required for fit Arrow")
        target_values = _validate_list_column(
            typed_table,
            "tgt",
            allow_nan=False,
            expected_width=target_width(target_contract),
        )
        validate_target_values(target_values, target_contract)

    return _ValidatedArrowColumns(
        features=source_values,
        targets=target_values,
    )


def read_arrow(
    path: str | os.PathLike[str],
    target_contract: JsonObject,
) -> TrainingBatch:
    with open(path, "rb") as f:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(f))
        table = reader.read_all()

    return table_to_tensors(table, target_contract)


def read_source_arrow(path: str | os.PathLike[str]) -> torch.Tensor:
    with open(path, "rb") as f:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(f))
        table = reader.read_all()

    return table_to_source_tensor(table)


def iter_committed_fit_arrow(
    path: str,
    *,
    expected_rows: int,
    expected_chunks: int,
    expected_native_rows: Sequence[int],
    source_encoding: Mapping[str, object],
    seq_len: int,
    feature_dim: int,
    target_contract: JsonObject,
) -> Iterator[TrainingBatch]:
    """Decode one immutable compact fit artifact in bounded row slices.

    The service validates values before durable commit and the worker verifies
    the receipt digest before the first read. Replay rechecks its physical
    schema and receipt counters without retaining the full dense dataset.
    """
    for features, target_values in _iter_committed_indexed_arrow(
        path,
        expected_rows=expected_rows,
        expected_chunks=expected_chunks,
        expected_native_rows=expected_native_rows,
        source_encoding=source_encoding,
        seq_len=seq_len,
        feature_dim=feature_dim,
        require_target=True,
        target_contract=target_contract,
    ):
        if target_values is None:
            raise AssertionError("fit compact input has no target values")
        yield TrainingBatch(
            features=_list_values_to_tensor(features),
            targets=_list_values_to_tensor(target_values),
        )


def iter_committed_source_arrow(
    path: str,
    *,
    expected_rows: int,
    expected_chunks: int,
    expected_native_rows: Sequence[int],
    source_encoding: Mapping[str, object],
    seq_len: int,
    feature_dim: int,
    target_contract: JsonObject,
) -> Iterator[torch.Tensor]:
    """Decode one immutable compact prediction artifact in bounded slices."""
    for features, _ in _iter_committed_indexed_arrow(
        path,
        expected_rows=expected_rows,
        expected_chunks=expected_chunks,
        expected_native_rows=expected_native_rows,
        source_encoding=source_encoding,
        seq_len=seq_len,
        feature_dim=feature_dim,
        require_target=False,
        target_contract=target_contract,
    ):
        yield _list_values_to_tensor(features)


def _iter_committed_indexed_arrow(
    path: str,
    *,
    expected_rows: int,
    expected_chunks: int,
    expected_native_rows: Sequence[int],
    source_encoding: Mapping[str, object],
    seq_len: int,
    feature_dim: int,
    require_target: bool,
    target_contract: JsonObject,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    if expected_rows < 0:
        raise ValueError("committed Arrow row count must be non-negative")
    if expected_chunks < 0:
        raise ValueError("committed Arrow chunk count must be non-negative")
    blocks = feature_block_dimensions(
        source_encoding,
        feature_dim=feature_dim,
    )
    if len(expected_native_rows) != len(blocks):
        raise ValueError(
            "committed Arrow native row counters differ from sourceEncoding"
        )

    expected_schema = canonical_input_schema(
        "fit" if require_target else "predict",
        source_encoding,
        seq_len,
        feature_dim,
        target_contract,
    )
    observed_rows = 0
    observed_chunks = 0
    observed_native_rows = [0] * len(blocks)
    with pa.memory_map(path, "r") as source:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(source))
        if not reader.schema.equals(expected_schema, check_metadata=False):
            raise ValueError(
                "Committed Arrow physical schema differs from the worker contract"
            )
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            features: pa.StructArray = batch.column(
                batch.schema.get_field_index("features")
            )
            target_column = (
                batch.column(batch.schema.get_field_index("tgt"))
                if require_target
                else None
            )
            for chunk_index in range(batch.num_rows):
                decoded_blocks: list[tuple[np.ndarray, np.ndarray, int]] = []
                logical_rows: int | None = None
                for block_index, (_, window_rows, native_width) in enumerate(
                    blocks
                ):
                    block: pa.StructArray = features.field(block_index)
                    native_scalar: pa.ListScalar = block.field(
                        "nativeRows"
                    )[chunk_index]
                    offset_scalar: pa.ListScalar = block.field(
                        "observationOffsets"
                    )[chunk_index]
                    native_values: pa.FixedSizeListArray = native_scalar.values
                    offset_values: pa.FixedSizeListArray = offset_scalar.values
                    native = native_values.flatten().to_numpy(
                        zero_copy_only=False
                    ).reshape(-1, native_width)
                    offsets = offset_values.flatten().to_numpy(
                        zero_copy_only=False
                    ).reshape(-1, seq_len)
                    if logical_rows is None:
                        logical_rows = offsets.shape[0]
                    elif logical_rows != offsets.shape[0]:
                        raise ValueError(
                            "Committed Arrow feature block row counts differ"
                        )
                    observed_native_rows[block_index] += native.shape[0]
                    decoded_blocks.append((native, offsets, window_rows))

                if logical_rows is None:
                    raise ValueError("Committed Arrow has no feature blocks")
                target_values = _committed_target_values(
                    target_column,
                    chunk_index,
                    logical_rows,
                    target_width(target_contract),
                )
                if target_values is not None:
                    validate_target_values(
                        target_values,
                        target_contract,
                        logical_row_offset=observed_rows,
                    )
                row_bytes = 4 * (
                    seq_len * feature_dim
                    + (
                        target_width(target_contract)
                        if require_target
                        else 0
                    )
                )
                rows_per_slice = max(1, (8 * 1024 * 1024) // row_bytes)
                for start in range(0, logical_rows, rows_per_slice):
                    stop = min(start + rows_per_slice, logical_rows)
                    block_values = [
                        _decode_feature_block(
                            native,
                            offsets[start:stop],
                            window_rows,
                        )
                        for native, offsets, window_rows in decoded_blocks
                    ]
                    dense = np.ascontiguousarray(
                        np.concatenate(block_values, axis=2),
                        dtype=np.float32,
                    )
                    targets_slice = (
                        None
                        if target_values is None
                        else np.array(
                            target_values[start:stop],
                            dtype=np.float32,
                            copy=True,
                            order="C",
                        )
                    )
                    yield dense, targets_slice
                observed_rows += logical_rows
                observed_chunks += 1

    if observed_rows != expected_rows:
        raise ValueError(
            f"Committed Arrow logical row count {observed_rows} does not match "
            f"receipt row count {expected_rows}"
        )
    if observed_chunks != expected_chunks:
        raise ValueError(
            f"Committed Arrow chunk count {observed_chunks} does not match "
            f"receipt chunk count {expected_chunks}"
        )
    if tuple(observed_native_rows) != tuple(expected_native_rows):
        raise ValueError(
            "Committed Arrow native row counts do not match the receipt"
        )


def _decode_feature_block(
    native: np.ndarray,
    offsets: np.ndarray,
    window_rows: int,
) -> np.ndarray:
    indices = (
        offsets.astype(np.int64, copy=False)[:, :, np.newaxis]
        + np.arange(window_rows, dtype=np.int64)[np.newaxis, np.newaxis, :]
    )
    values = native[indices]
    return values.reshape(values.shape[0], values.shape[1], -1)


def _committed_target_values(
    column: pa.Array | None,
    chunk_index: int,
    logical_rows: int,
    target_width: int,
) -> np.ndarray | None:
    if column is None:
        return None
    scalar: pa.ListScalar = column[chunk_index]
    rows: pa.FixedSizeListArray = scalar.values
    values = rows.flatten().to_numpy(zero_copy_only=False).reshape(
        -1,
        target_width,
    )
    if values.shape[0] != logical_rows:
        raise ValueError(
            "Committed Arrow target row count differs from feature rows"
        )
    return values


def _validate_list_column(
    table: _ArrowTable,
    name: str,
    *,
    allow_nan: bool,
    expected_width: int | None = None,
) -> np.ndarray:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Arrow table must contain '{name}' column")

    column_type = table.schema.field(column_index).type
    if not (
        pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
        or pa.types.is_fixed_size_list(column_type)
    ):
        raise ValueError(
            f"Arrow column '{name}' must be a list<float32> or list<float64> column"
        )

    list_type = cast(_ListType, column_type)
    if list_type.value_type not in (pa.float32(), pa.float64()):
        raise ValueError(
            f"Arrow column '{name}' must be a list<float32> or list<float64> column"
        )

    width = list_type.list_size if pa.types.is_fixed_size_list(column_type) else None
    chunks: list[tuple[int, int, np.ndarray]] = []
    row_offset = 0

    for chunk in table.column(column_index).chunks:
        if len(chunk) == 0:
            continue

        null_row = _first_true(chunk.is_null().to_numpy(zero_copy_only=False))
        if null_row is not None:
            raise ValueError(
                f"Arrow column '{name}' has null row at index "
                f"{row_offset + null_row + 1}"
            )

        if not pa.types.is_fixed_size_list(column_type):
            offsets = chunk.offsets.to_numpy(zero_copy_only=False)
            lengths = np.diff(offsets)
            if width is None:
                width = int(lengths[0])
            inconsistent = _first_true(lengths != width)
            if inconsistent is not None:
                raise ValueError(
                    f"Arrow column '{name}' has inconsistent list length at row "
                    f"{row_offset + inconsistent + 1}"
                )

        if width is None:
            raise ValueError(f"Arrow column '{name}' list width is unavailable")

        flat = chunk.flatten()
        null_value = _first_true(flat.is_null().to_numpy(zero_copy_only=False))
        if null_value is not None:
            row_index, value_index = divmod(null_value, width)
            raise ValueError(
                f"Arrow column '{name}' has null element at row "
                f"{row_offset + row_index + 1}, position {value_index + 1}"
            )

        flat_values = flat.to_numpy(zero_copy_only=False)
        invalid = (
            np.isinf(flat_values)
            if allow_nan
            else ~np.isfinite(flat_values)
        )
        invalid_value = _first_true(invalid)
        if invalid_value is not None:
            row_index, value_index = divmod(invalid_value, width)
            raise ValueError(
                f"Arrow column '{name}' has non-finite value at row "
                f"{row_offset + row_index + 1}, position {value_index + 1}"
            )

        outside_float32 = _first_true(np.abs(flat_values) > FLOAT32_MAX)
        if outside_float32 is not None:
            row_index, value_index = divmod(outside_float32, width)
            raise ValueError(
                f"Arrow column '{name}' has value outside float32 range at "
                f"row {row_offset + row_index + 1}, position {value_index + 1}"
            )

        chunks.append((row_offset, len(chunk), flat_values))
        row_offset += len(chunk)

    if expected_width is not None and width is not None and width != expected_width:
        raise ValueError(
            f"Arrow column '{name}' must have list length {expected_width}, got {width}"
        )
    if name == "src" and table.num_rows > 0 and width == 0:
        raise ValueError("Arrow column 'src' must have a positive list length")

    if table.num_rows == 0:
        empty_width = width if width is not None else expected_width
        return np.empty((0, empty_width or 0), dtype=np.float32)

    if width is None:
        raise ValueError(f"Arrow column '{name}' list width is unavailable")
    values = np.empty((table.num_rows, width), dtype=np.float32)
    for offset, rows, flat_values in chunks:
        values[offset:offset + rows] = flat_values.reshape(rows, width)
    return values


def _list_values_to_tensor(values: np.ndarray) -> torch.Tensor:
    import torch

    # PyTorch leaves the ndarray parameter unknown in its public type surface.
    return torch.from_numpy(values)  # pyright: ignore[reportUnknownMemberType]


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Unexpected end of framed Arrow stream")

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def iter_framed_arrow(
    stream: BinaryIO,
    max_frame_bytes: int = defaults.DEFAULT_MAX_FRAME_BYTES,
) -> Iterator[pa.Table]:
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be greater than zero")

    while True:
        header = stream.read(FRAME_HEADER_BYTES)

        if header == b"":
            return

        if len(header) != FRAME_HEADER_BYTES:
            raise EOFError("Incomplete framed Arrow header")

        payload_size = int.from_bytes(header, byteorder="big", signed=False)
        if payload_size == 0:
            return
        if payload_size > max_frame_bytes:
            raise ValueError(
                f"Framed Arrow payload size {payload_size} exceeds maximum "
                f"{max_frame_bytes} bytes"
            )

        payload = _read_exact(stream, payload_size)
        reader = cast(
            _ArrowReader,
            ipc.RecordBatchFileReader(pa.BufferReader(payload)),
        )

        yield reader.read_all()


def write_arrow(
    path: str,
    predictions: torch.Tensor,
    col_name: str,
    target_contract: JsonObject,
    expected_rows: int | None = None,
) -> None:
    table = predictions_to_table(
        predictions,
        col_name,
        expected_rows=expected_rows,
        target_contract=target_contract,
    )

    with atomic_output_path(path) as temporary_path:
        with pa.OSFile(temporary_path, "wb") as sink:
            with cast(
                _ArrowWriter,
                ipc.new_file(  # pyright: ignore[reportUnknownMemberType]
                    sink,
                    cast(_ArrowTable, table).schema,
                ),
            ) as writer:
                writer.write_table(table)


def predictions_to_table(
    predictions: torch.Tensor,
    col_name: str,
    target_contract: JsonObject,
    expected_rows: int | None = None,
) -> pa.Table:
    import torch

    width = target_width(target_contract)
    if predictions.ndim != 2 or predictions.shape[1] != width:
        raise ValueError(
            f"Predictions must have shape [rows, {width}], "
            f"got {list(predictions.shape)}"
        )
    if expected_rows is not None and predictions.shape[0] != expected_rows:
        raise ValueError(
            f"Predictions row count {predictions.shape[0]} does not match input row "
            f"count {expected_rows}"
        )
    arr = predictions.detach().cpu().to(dtype=torch.float32).numpy()
    if not np.isfinite(arr).all():
        raise ValueError("Predictions must contain only finite values")
    validate_target_values(arr, target_contract)
    arr = np.ascontiguousarray(arr)
    values = pa.array(arr.reshape(-1), type=pa.float32())
    column = pa.FixedSizeListArray.from_arrays(values, width)
    schema = canonical_prediction_schema(col_name, target_contract)
    return pa.Table.from_arrays([column], schema=schema)


def empty_predictions_table(
    col_name: str,
    target_contract: JsonObject,
) -> pa.Table:
    schema = canonical_prediction_schema(col_name, target_contract)
    return pa.Table.from_arrays(
        [pa.array([], type=schema.field(0).type)],
        schema=schema,
    )


def write_framed_arrow(stream: BinaryIO, table: pa.Table) -> None:
    sink = pa.BufferOutputStream()
    with cast(
        _ArrowWriter,
        ipc.new_file(  # pyright: ignore[reportUnknownMemberType]
            sink,
            cast(_ArrowTable, table).schema,
        ),
    ) as writer:
        writer.write_table(table)

    payload = sink.getvalue()
    stream.write(
        len(payload).to_bytes(
            FRAME_HEADER_BYTES,
            byteorder="big",
            signed=False,
        )
    )
    stream.write(memoryview(payload))
    stream.flush()
