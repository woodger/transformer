import os
import uuid
from pathlib import Path

import pytest

from app.service.adapters.outbound.artifact_storage.recovery_store import (
    RecoveryStore,
)
from app.service.adapters.outbound.artifact_storage.spool import (
    RuntimeDirectoryLocked,
)


def test_recovery_store_uses_managed_job_paths_and_durable_replace(
    tmp_path,
):
    store = RecoveryStore(tmp_path / "recovery").initialize()
    job_id = str(uuid.uuid4())
    destination = store.input_path(job_id, 0)
    target, temporary = store.create_temporary(destination)
    target.write(b"arrow")
    target.close()

    store.durable_replace(temporary, destination)

    assert Path(destination).read_bytes() == b"arrow"
    assert store.relative_path(destination) == (
        f"jobs/{job_id}/inputs/0.arrow"
    )
    assert store.absolute_path(
        f"jobs/{job_id}/inputs/0.arrow"
    ) == destination
    with pytest.raises(ValueError, match="traversal"):
        store.absolute_path("../models/checkpoint.pth")


def test_reconcile_preserves_registered_files_and_removes_orphans(
    tmp_path,
):
    store = RecoveryStore(tmp_path / "recovery").initialize()
    active_job = str(uuid.uuid4())
    orphan_job = str(uuid.uuid4())
    active_input = store.input_path(active_job, 0)
    active_checkpoint = store.checkpoint_path(active_job, 2)
    orphan_checkpoint = store.checkpoint_path(active_job, 1)
    orphan_job_input = store.input_path(orphan_job, 0)
    temporary = active_input + ".write.tmp"
    for path, data in (
        (active_input, b"input"),
        (active_checkpoint, b"checkpoint"),
        (orphan_checkpoint, b"old"),
        (orphan_job_input, b"orphan"),
        (temporary, b"temporary"),
    ):
        store.ensure_parent(path)
        Path(path).write_bytes(data)

    removed = store.reconcile(
        {
            store.relative_path(active_input),
            store.relative_path(active_checkpoint),
        },
        known_job_ids={active_job},
    )

    assert Path(active_input).read_bytes() == b"input"
    assert Path(active_checkpoint).read_bytes() == b"checkpoint"
    assert not os.path.exists(orphan_checkpoint)
    assert not os.path.exists(store.job_directory(orphan_job))
    assert not os.path.exists(temporary)
    assert set(removed) == {
        f"jobs/{active_job}/checkpoints/1.pth",
        f"jobs/{active_job}/inputs/0.arrow.write.tmp",
        f"jobs/{orphan_job}",
    }


def test_recovery_store_has_exclusive_service_lock(tmp_path):
    first = RecoveryStore(tmp_path / "recovery").initialize()
    second = RecoveryStore(tmp_path / "recovery").initialize()
    first.acquire_lock()
    try:
        with pytest.raises(RuntimeDirectoryLocked):
            second.acquire_lock()
    finally:
        first.release_lock()

    second.acquire_lock()
    second.release_lock()
