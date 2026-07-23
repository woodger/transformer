import os
import uuid
from types import SimpleNamespace

import pytest

import app.flight.spool as spool_module
from app.flight.constants import ErrorCode
from app.flight.errors import ServiceError
from app.flight.spool import RuntimeDirectoryLocked, Spool


@pytest.fixture
def spool(tmp_path):
    return Spool(tmp_path / "runtime", tmp_path / "models").initialize()


def test_runtime_directory_lock_excludes_second_service(spool):
    second = Spool(spool.runtime_dir).initialize()

    spool.acquire_lock()
    try:
        with pytest.raises(RuntimeDirectoryLocked, match="already owned"):
            second.acquire_lock()
    finally:
        spool.release_lock()

    second.acquire_lock()
    second.release_lock()


def test_staged_file_atomically_replaces_target_and_removes_temporary(spool):
    job_id = str(uuid.uuid4())
    destination = spool.input_path(job_id, 0)
    spool.atomic_write_bytes(destination, b"old")

    with spool.staged_file(destination) as (target, temporary_path):
        target.write(b"new payload")
        assert os.path.exists(temporary_path)
        assert open(destination, "rb").read() == b"old"

    assert open(destination, "rb").read() == b"new payload"
    assert not os.path.exists(temporary_path)


def test_rename_failure_leaves_old_target_and_cleans_temporary(spool, monkeypatch):
    job_id = str(uuid.uuid4())
    destination = spool.input_path(job_id, 0)
    spool.atomic_write_bytes(destination, b"old")
    real_replace = os.replace

    def fail_replace(source, target):
        if target == destination:
            raise OSError("injected rename failure")
        return real_replace(source, target)

    monkeypatch.setattr(spool_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected rename failure"):
        with spool.staged_file(destination) as (target, temporary_path):
            target.write(b"not committed")

    assert open(destination, "rb").read() == b"old"
    assert not os.path.exists(temporary_path)


def test_file_fsync_failure_never_replaces_target(spool, monkeypatch):
    job_id = str(uuid.uuid4())
    destination = spool.input_path(job_id, 0)
    spool.atomic_write_bytes(destination, b"old")
    real_fsync = os.fsync

    def fail_file_fsync(descriptor):
        path = os.readlink(f"/proc/self/fd/{descriptor}")
        if path.endswith(".tmp"):
            raise OSError("injected fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(spool_module.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        with spool.staged_file(destination) as (target, temporary_path):
            target.write(b"not durable")

    assert open(destination, "rb").read() == b"old"
    assert not os.path.exists(temporary_path)


def test_reconcile_removes_orphans_and_preserves_ledger_references(spool):
    job_id = str(uuid.uuid4())
    referenced_input = spool.input_path(job_id, 0)
    orphan_input = spool.input_path(job_id, 1)
    orphan_output = spool.attempt_output_path(job_id, 1, 0)
    spool.atomic_write_bytes(referenced_input, b"kept")
    spool.atomic_write_bytes(orphan_input, b"orphan")
    spool.atomic_write_bytes(orphan_output, b"orphan output")

    kept_model = "mdl_kept"
    orphan_model = "mdl_orphan"
    kept_checkpoint = spool.model_checkpoint_path(kept_model)
    kept_metadata = spool.model_metadata_path(kept_model)
    spool.atomic_write_bytes(kept_checkpoint, b"checkpoint")
    spool.atomic_write_json(kept_metadata, {"format": "v2"})
    spool.atomic_write_bytes(spool.model_checkpoint_path(orphan_model), b"orphan")

    dangling_file, dangling_path = spool.create_temporary(spool.input_path(job_id, 2))
    dangling_file.write(b"partial")
    dangling_file.close()

    result = spool.reconcile({spool.relative_path(referenced_input)})
    removed_models = spool.reconcile_model_directories({kept_model})

    assert os.path.exists(referenced_input)
    assert os.path.exists(kept_checkpoint)
    assert os.path.exists(kept_metadata)
    assert not os.path.exists(orphan_input)
    assert not os.path.exists(orphan_output)
    assert not os.path.exists(spool.model_directory(orphan_model))
    assert not os.path.exists(dangling_path)
    assert result["removed"]
    assert removed_models == (orphan_model,)


def test_startup_reconcile_removes_job_directory_absent_from_ledger(spool):
    known_job_id = str(uuid.uuid4())
    deleted_job_id = str(uuid.uuid4())
    known_log = spool.attempt_stdout_path(known_job_id, 1)
    orphan_log = spool.attempt_stdout_path(deleted_job_id, 1)
    orphan_checkpoint = spool.attempt_checkpoint_path(deleted_job_id, 1)
    spool.atomic_write_bytes(known_log, b"keep")
    spool.atomic_write_bytes(orphan_log, b"orphan")
    spool.atomic_write_bytes(orphan_checkpoint, b"orphan checkpoint")

    result = spool.reconcile(set(), known_job_ids={known_job_id})

    assert os.path.isfile(known_log)
    assert not os.path.exists(spool.job_directory(deleted_job_id))
    assert spool.relative_path(spool.job_directory(deleted_job_id)) in result["removed"]


def test_startup_reconcile_removes_unpublished_attempt_results_for_known_job(spool):
    job_id = str(uuid.uuid4())
    output = spool.attempt_output_path(job_id, 1, 0)
    checkpoint = spool.attempt_checkpoint_path(job_id, 1)
    stdout = spool.attempt_stdout_path(job_id, 1)
    stderr = spool.attempt_stderr_path(job_id, 1)
    metrics = spool.attempt_metrics_path(job_id, 1)
    spool.atomic_write_bytes(output, b"unpublished output")
    spool.atomic_write_bytes(checkpoint, b"unpublished checkpoint")
    spool.atomic_write_bytes(stdout, b"diagnostic stdout")
    spool.atomic_write_bytes(stderr, b"diagnostic stderr")
    spool.atomic_write_bytes(metrics, b'{"epoch":1}\n')

    result = spool.reconcile(set(), known_job_ids={job_id})

    assert not os.path.exists(output)
    assert not os.path.exists(checkpoint)
    assert os.path.isfile(stdout)
    assert os.path.isfile(stderr)
    assert os.path.isfile(metrics)
    assert spool.relative_path(output) in result["removed"]
    assert spool.relative_path(checkpoint) in result["removed"]


def test_preledger_cleanup_removes_only_temporary_artifacts(spool):
    job_id = str(uuid.uuid4())
    committed = spool.input_path(job_id, 0)
    spool.atomic_write_bytes(committed, b"committed")
    temporary, temporary_path = spool.create_temporary(spool.input_path(job_id, 1))
    temporary.write(b"crash-left staging bytes")
    temporary.close()

    removed = spool.cleanup_temporary_files()

    assert os.path.isfile(committed)
    assert not os.path.exists(temporary_path)
    assert spool.relative_path(temporary_path) in removed


def test_paths_cannot_escape_runtime_directory_or_follow_external_symlink(spool, tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        spool.absolute_path("../outside.arrow")
    with pytest.raises(ValueError, match="relative"):
        spool.absolute_path(str(tmp_path / "outside.arrow"))

    outside = tmp_path / "outside"
    outside.mkdir()
    link = os.path.join(spool.runtime_dir, "linked")
    os.symlink(outside, link, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        spool.absolute_path("linked/file.arrow")


def test_disk_watermark_is_checked_before_staging(spool, monkeypatch):
    monkeypatch.setattr(
        spool,
        "disk_usage",
        lambda: SimpleNamespace(total=100, used=91, free=9),
    )

    with pytest.raises(ServiceError) as error:
        spool.ensure_free_space(5, required_bytes=5)
    assert error.value.code == ErrorCode.DISK_FULL
