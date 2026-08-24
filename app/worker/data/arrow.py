from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app import config as defaults
from app.contracts.flight.v5.arrow import (
    TARGET_WIDTH,
    canonical_input_schema,
    canonical_prediction_schema,
    validate_target_space_values,
)
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


def table_to_tensors(table: pa.Table) -> TrainingBatch:
    columns = _validated_arrow_columns(
        table,
        require_target=True,
    )
    features = _list_values_to_tensor(columns.features)
    if columns.targets is None:
        raise AssertionError("fit Arrow validation did not return targets")
    targets = _list_values_to_tensor(columns.targets)

    return TrainingBatch(features=features, targets=targets)


def table_to_source_tensor(table: pa.Table) -> torch.Tensor:
    columns = _validated_arrow_columns(table, require_target=False)
    return _list_values_to_tensor(columns.features)


def validate_arrow_table(
    table: pa.Table,
    require_target: bool = False,
) -> None:
    _validated_arrow_columns(table, require_target=require_target)


def _validated_arrow_columns(
    table: pa.Table,
    require_target: bool,
) -> _ValidatedArrowColumns:
    typed_table = cast(_ArrowTable, table)
    source_values = _validate_list_column(
        typed_table,
        "src",
        allow_nan=True,
    )
    target_values = None
    if require_target:
        target_values = _validate_list_column(
            typed_table,
            "tgt",
            allow_nan=False,
            expected_width=TARGET_WIDTH,
        )
        _validate_target_values(target_values)

    return _ValidatedArrowColumns(
        features=source_values,
        targets=target_values,
    )


def read_arrow(
    path: str | os.PathLike[str],
) -> TrainingBatch:
    with open(path, "rb") as f:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(f))
        table = reader.read_all()

    return table_to_tensors(table)


def read_source_arrow(path: str | os.PathLike[str]) -> torch.Tensor:
    with open(path, "rb") as f:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(f))
        table = reader.read_all()

    return table_to_source_tensor(table)


def read_committed_fit_arrow(
    path: str,
    *,
    expected_rows: int,
    source_width: int,
) -> TrainingBatch:
    """Replay one service-validated immutable fit artifact.

    The service validates values before durable commit and the worker verifies
    the receipt digest before the first read. Replay therefore rechecks only
    the physical Arrow contract and receipt dimensions instead of rescanning
    every value on every epoch.
    """

    table = _read_committed_table(
        path,
        expected_rows=expected_rows,
        source_width=source_width,
        require_target=True,
    )
    return TrainingBatch(
        features=_committed_column_to_tensor(table, "src", source_width),
        targets=_committed_column_to_tensor(table, "tgt", TARGET_WIDTH),
    )


def read_committed_source_arrow(
    path: str,
    *,
    expected_rows: int,
    source_width: int,
) -> torch.Tensor:
    """Read one service-validated immutable prediction input artifact."""

    table = _read_committed_table(
        path,
        expected_rows=expected_rows,
        source_width=source_width,
        require_target=False,
    )
    return _committed_column_to_tensor(table, "src", source_width)


def _read_committed_table(
    path: str,
    *,
    expected_rows: int,
    source_width: int,
    require_target: bool,
) -> _ArrowTable:
    if expected_rows < 0:
        raise ValueError("committed Arrow row count must be non-negative")
    if source_width <= 0:
        raise ValueError("committed Arrow source width must be positive")

    expected_schema = canonical_input_schema(
        "fit" if require_target else "predict",
        source_width,
    )

    with open(path, "rb") as source:
        reader = cast(_ArrowReader, ipc.RecordBatchFileReader(source))
        if not reader.schema.equals(expected_schema, check_metadata=False):
            raise ValueError(
                "Committed Arrow physical schema differs from the worker contract"
            )
        table = cast(_ArrowTable, reader.read_all())

    if table.num_rows != expected_rows:
        raise ValueError(
            f"Committed Arrow row count {table.num_rows} does not match receipt "
            f"row count {expected_rows}"
        )
    return table


def _committed_column_to_tensor(
    table: _ArrowTable,
    name: str,
    width: int,
) -> torch.Tensor:
    values = np.empty((table.num_rows, width), dtype=np.float32)
    row_offset = 0
    for chunk in table.column(name).chunks:
        rows = len(chunk)
        if rows == 0:
            continue
        flat_values = chunk.flatten().to_numpy(zero_copy_only=False)
        values[row_offset:row_offset + rows] = flat_values.reshape(rows, width)
        row_offset += rows
    return _list_values_to_tensor(values)


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


def _validate_target_values(values: np.ndarray) -> None:
    validate_target_space_values(values)


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
    expected_rows: int | None = None,
) -> None:
    table = predictions_to_table(
        predictions,
        col_name,
        expected_rows=expected_rows,
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
    expected_rows: int | None = None,
) -> pa.Table:
    import torch

    if predictions.ndim != 2 or predictions.shape[1] != TARGET_WIDTH:
        raise ValueError(
            f"Predictions must have shape [rows, {TARGET_WIDTH}], "
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
    validate_target_space_values(arr)
    arr = np.ascontiguousarray(arr)
    values = pa.array(arr.reshape(-1), type=pa.float32())
    column = pa.FixedSizeListArray.from_arrays(values, TARGET_WIDTH)
    schema = canonical_prediction_schema(col_name)
    return pa.Table.from_arrays([column], schema=schema)


def empty_predictions_table(col_name: str) -> pa.Table:
    schema = canonical_prediction_schema(col_name)
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
