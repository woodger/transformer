from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.flight.v18.arrow import (
    canonical_input_schema,
    canonical_prediction_schema,
    target_width,
    validate_target_values,
)
from app.contracts.flight.v18.source_encoding import feature_block_dimensions
from app.contracts.json_types import JsonObject
from app.worker.checkpoints.atomic import atomic_output_path
from app.worker.data.tensors import TrainingBatch

if TYPE_CHECKING:
    import torch


class _ArrowTable(Protocol):
    @property
    def schema(self) -> pa.Schema: ...


class _ArrowReader(Protocol):
    @property
    def schema(self) -> pa.Schema: ...

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
    binary_target_indices: Sequence[int] = (),
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
        binary_target_indices=binary_target_indices,
    ):
        if target_values is None:
            raise AssertionError("fit compact input has no target values")
        yield TrainingBatch(
            features=_list_values_to_tensor(features),
            targets=_list_values_to_tensor(target_values),
        )


def iter_committed_fit_features(
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
    """Декодировать признаки fit-артефакта без материализации целей."""
    for features, _ in _iter_committed_indexed_arrow(
        path,
        expected_rows=expected_rows,
        expected_chunks=expected_chunks,
        expected_native_rows=expected_native_rows,
        source_encoding=source_encoding,
        seq_len=seq_len,
        feature_dim=feature_dim,
        require_target=True,
        target_contract=target_contract,
        materialize_targets=False,
    ):
        yield _list_values_to_tensor(features)


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
    binary_target_indices: Sequence[int] = (),
    materialize_targets: bool = True,
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
                if require_target and materialize_targets
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
                target_values = (
                    None
                    if not materialize_targets
                    else _committed_target_values(
                        target_column,
                        chunk_index,
                        logical_rows,
                        target_width(target_contract),
                    )
                )
                if target_values is not None:
                    validate_target_values(
                        target_values,
                        target_contract,
                        logical_row_offset=observed_rows,
                        binary_target_indices=binary_target_indices,
                    )
                row_bytes = 4 * (
                    seq_len * feature_dim
                    + (
                        target_width(target_contract)
                        if require_target and materialize_targets
                        else 0
                    )
                )
                # Ограничиваем временную плотную реконструкцию, пока компактный IPC-артефакт
                # остаётся отображённым в память.
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


def _list_values_to_tensor(values: np.ndarray) -> torch.Tensor:
    import torch

    # Библиотека PyTorch оставляет параметр ndarray неизвестным в публичной типовой поверхности.
    return torch.from_numpy(values)  # pyright: ignore[reportUnknownMemberType]


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
