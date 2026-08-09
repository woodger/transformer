from __future__ import annotations

from app.service.bootstrap.worker_pool import WorkerPool as ServiceWorkerPool


def _read_legacy_checkpoint_metadata(path: str, device: str) -> dict:
    from app.worker.runtime.checkpoints.checkpoint import load_checkpoint_metadata

    return load_checkpoint_metadata(path, device)


class WorkerPool(ServiceWorkerPool):
    """Compatibility facade for legacy injected subprocess tests."""

    def __init__(self, *args, checkpoint_metadata_reader=None, **kwargs):
        super().__init__(
            *args,
            checkpoint_metadata_reader=(
                checkpoint_metadata_reader
                or _read_legacy_checkpoint_metadata
            ),
            **kwargs,
        )


__all__ = ["WorkerPool"]
