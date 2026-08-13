from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.worker.data.tensors import TrainingBatch

_MAX_SHUFFLE_WINDOW_BATCHES = 32
_MAX_SHUFFLE_WINDOW_BYTES = 64 * 1024 * 1024

TrainingBatches = Iterable[TrainingBatch]


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingBatchLoader:
    batch: TrainingBatch
    batch_size: int

    def __iter__(self) -> Iterator[TrainingBatch]:
        dataset = TensorDataset(self.batch.features, self.batch.targets)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        for batch_features, batch_targets in loader:
            yield TrainingBatch(
                features=batch_features,
                targets=batch_targets,
            )


@dataclass(frozen=True)
class _PrefetchEnd:
    pass


_PREFETCH_END = _PrefetchEnd()


@dataclass(frozen=True)
class _PrefetchError:
    error: BaseException


class BatchPrefetcher:
    """Prepare at most one closed-input batch ahead of the trainer."""

    def __init__(self, batches: TrainingBatches) -> None:
        self._batches = iter(batches)
        self._queue: queue.Queue[
            TrainingBatch | _PrefetchError | _PrefetchEnd
        ] = queue.Queue(maxsize=1)
        self._slot = threading.Semaphore(1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._produce,
            name="transformer-batch-prefetch",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> BatchPrefetcher:
        return self

    def __next__(self) -> TrainingBatch:
        message = self._queue.get()
        self._slot.release()
        if isinstance(message, _PrefetchEnd):
            self._thread.join()
            raise StopIteration
        if isinstance(message, _PrefetchError):
            self._thread.join()
            raise message.error
        return message

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.1)

    def _produce(self) -> None:
        try:
            while self._reserve_slot():
                try:
                    message = next(self._batches)
                except StopIteration:
                    self._queue.put(_PREFETCH_END)
                    return
                except BaseException as exc:
                    self._queue.put(_PrefetchError(exc))
                    return
                if self._stop.is_set():
                    self._slot.release()
                    return
                self._queue.put(message)
        finally:
            if isinstance(self._batches, Closable):
                self._batches.close()

    def _reserve_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slot.acquire(timeout=0.05):
                return True
        return False


@dataclass(frozen=True, slots=True)
class PayloadBatcher:
    batch_size: int

    def batches(
        self,
        payloads: TrainingBatches,
        generator: torch.Generator,
    ) -> Iterator[TrainingBatch]:
        window_rows: int | None = None
        features_buffer: torch.Tensor | None = None
        targets_buffer: torch.Tensor | None = None
        buffered_rows = 0

        for batch in payloads:
            payload_rows = batch.features.size(0)
            if batch.targets.size(0) != payload_rows:
                raise ValueError("features and targets row counts must match")
            if payload_rows == 0:
                continue

            if features_buffer is None:
                row_bytes = (
                    batch.features[0].numel()
                    * batch.features.element_size()
                    + batch.targets[0].numel()
                    * batch.targets.element_size()
                )
                batch_bytes = self.batch_size * row_bytes
                window_batches = min(
                    _MAX_SHUFFLE_WINDOW_BATCHES,
                    max(1, _MAX_SHUFFLE_WINDOW_BYTES // batch_bytes),
                )
                window_rows = self.batch_size * window_batches
                features_buffer = torch.empty(
                    (window_rows, *batch.features.shape[1:]),
                    dtype=batch.features.dtype,
                    device=batch.features.device,
                )
                targets_buffer = torch.empty(
                    (window_rows, *batch.targets.shape[1:]),
                    dtype=batch.targets.dtype,
                    device=batch.targets.device,
                )

            if window_rows is None or targets_buffer is None:
                raise AssertionError("shuffle buffers were not initialized")

            offset = 0
            while offset < payload_rows:
                copied_rows = min(
                    window_rows - buffered_rows,
                    payload_rows - offset,
                )
                buffer_end = buffered_rows + copied_rows
                payload_end = offset + copied_rows
                features_buffer[buffered_rows:buffer_end].copy_(
                    batch.features[offset:payload_end]
                )
                targets_buffer[buffered_rows:buffer_end].copy_(
                    batch.targets[offset:payload_end]
                )
                buffered_rows = buffer_end
                offset = payload_end

                if buffered_rows == window_rows:
                    yield from self._shuffled_batches(
                        features_buffer,
                        targets_buffer,
                        buffered_rows,
                        generator,
                    )
                    buffered_rows = 0

            del batch

        if buffered_rows:
            if features_buffer is None or targets_buffer is None:
                raise AssertionError("shuffle buffers were not initialized")
            yield from self._shuffled_batches(
                features_buffer,
                targets_buffer,
                buffered_rows,
                generator,
            )

    def prefetched(
        self,
        payloads: TrainingBatches,
        generator: torch.Generator,
    ) -> BatchPrefetcher:
        return BatchPrefetcher(self.batches(payloads, generator))

    def data_loader(self, batch: TrainingBatch) -> TrainingBatches:
        return TrainingBatchLoader(batch=batch, batch_size=self.batch_size)

    def _shuffled_batches(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        rows: int,
        generator: torch.Generator,
    ) -> Iterator[TrainingBatch]:
        order = torch.randperm(
            rows,
            generator=generator,
            device=features.device,
        )
        for offset in range(0, rows, self.batch_size):
            indices = order[offset:offset + self.batch_size]
            yield TrainingBatch(
                features=features.index_select(0, indices),
                targets=targets.index_select(0, indices),
            )


__all__ = [
    "BatchPrefetcher",
    "Closable",
    "PayloadBatcher",
    "TrainingBatchLoader",
    "TrainingBatches",
]
