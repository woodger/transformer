from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from app.config import PROJECT_ROOT

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RuntimeDirectoryLocked(RuntimeError):
    pass


class RuntimeDirectoryLock:
    """Exclusive process-lifetime ownership of the Flight runtime directory."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.fspath(path))
        self._file: BinaryIO | None = None

    def acquire(self) -> RuntimeDirectoryLock:
        if self._file is not None:
            return self
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        lock_file = open(self.path, "a+b", buffering=0)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeDirectoryLocked(
                f"Flight runtime directory is already owned: {os.path.dirname(self.path)}"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n".encode("ascii"))
        os.fsync(lock_file.fileno())
        self._file = lock_file
        return self

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> RuntimeDirectoryLock:
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class Spool:
    """Server-controlled runtime paths and model publication primitives.

    Unfinished artifacts live below ``runtime_dir``. Successfully validated
    model generations are copied and atomically published below ``models_dir``.
    """

    def __init__(self, runtime_dir: str, models_dir: str | None = None):
        self.runtime_dir = os.path.abspath(os.fspath(runtime_dir))
        self.models_dir = os.path.abspath(
            os.fspath(models_dir or os.path.join(PROJECT_ROOT, "models"))
        )
        if os.path.commonpath((self.runtime_dir, self.models_dir)) in (
            self.runtime_dir,
            self.models_dir,
        ):
            raise ValueError("runtime and model directories must not overlap")
        self.spool_dir = os.path.join(self.runtime_dir, "spool")
        self.jobs_dir = os.path.join(self.spool_dir, "jobs")
        self.epoch_path = os.path.join(self.runtime_dir, "storage-epoch")
        self.lock = RuntimeDirectoryLock(os.path.join(self.runtime_dir, "service.lock"))

    def initialize(self) -> Spool:
        created = []
        for directory in (self.runtime_dir, self.spool_dir, self.jobs_dir, self.models_dir):
            if not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)
                created.append(directory)
        for directory in reversed(created):
            parent = os.path.dirname(directory)
            if parent and os.path.isdir(parent):
                fsync_directory(parent)
        return self

    def acquire_lock(self) -> RuntimeDirectoryLock:
        return self.lock.acquire()

    def release_lock(self) -> None:
        self.lock.release()

    def storage_epoch(self) -> str:
        """Return the runtime filesystem generation, creating it when absent."""
        try:
            with open(self.epoch_path, encoding="ascii") as source:
                value = source.read().strip()
            return _uuid_component(value, "storage epoch")
        except FileNotFoundError:
            value = str(uuid.uuid4())
            self.atomic_write_bytes(self.epoch_path, f"{value}\n".encode("ascii"))
            return value

    def input_directory(self, job_id: str) -> str:
        return os.path.join(self.job_directory(job_id), "inputs")

    def input_path(self, job_id: str, ordinal: int) -> str:
        _nonnegative(ordinal, "ordinal")
        return os.path.join(self.input_directory(job_id), f"{ordinal}.arrow")

    def job_directory(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, _uuid_component(job_id, "job_id"))

    def attempt_directory(self, job_id: str, attempt: int) -> str:
        _positive(attempt, "attempt")
        return os.path.join(self.job_directory(job_id), "attempts", str(attempt))

    def attempt_metrics_path(self, job_id: str, attempt: int) -> str:
        return os.path.join(self.attempt_directory(job_id, attempt), "metrics.jsonl")

    def attempt_stdout_path(self, job_id: str, attempt: int) -> str:
        return os.path.join(self.attempt_directory(job_id, attempt), "stdout.log")

    def attempt_stderr_path(self, job_id: str, attempt: int) -> str:
        return os.path.join(self.attempt_directory(job_id, attempt), "stderr.log")

    def attempt_recovery_events_path(
        self,
        job_id: str,
        attempt: int,
    ) -> str:
        return os.path.join(
            self.attempt_directory(job_id, attempt),
            "recovery-events.jsonl",
        )

    def attempt_output_path(self, job_id: str, attempt: int, ordinal: int) -> str:
        _nonnegative(ordinal, "ordinal")
        return os.path.join(
            self.attempt_directory(job_id, attempt),
            "outputs",
            f"{ordinal}.arrow",
        )

    def attempt_checkpoint_path(self, job_id: str, attempt: int) -> str:
        return os.path.join(self.attempt_directory(job_id, attempt), "checkpoint.pth")

    def model_directory(self, model_ref: str) -> str:
        return os.path.join(self.models_dir, _safe_component(model_ref, "model_ref"))

    def model_checkpoint_path(self, model_ref: str) -> str:
        return os.path.join(self.model_directory(model_ref), "checkpoint.pth")

    def model_metadata_path(self, model_ref: str) -> str:
        return os.path.join(self.model_directory(model_ref), "metadata.json")

    def relative_path(self, absolute_path: str) -> str:
        resolved = self._inside_runtime(absolute_path)
        return os.path.relpath(resolved, self.runtime_dir).replace(os.sep, "/")

    def absolute_path(self, relative_path: str) -> str:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("artifact path must be a non-empty relative path")
        if os.path.isabs(relative_path):
            raise ValueError("artifact path must be relative to the runtime directory")
        normalized = relative_path.replace("\\", "/")
        if normalized in ("", "."):
            raise ValueError("artifact path must name a file inside the runtime directory")
        if ".." in Path(normalized).parts:
            raise ValueError("artifact path must not contain path traversal")
        return self._inside_runtime(
            os.path.join(self.runtime_dir, *normalized.split("/"))
        )

    def model_relative_path(self, absolute_path: str) -> str:
        resolved = self._inside_models(absolute_path)
        return os.path.relpath(resolved, self.models_dir).replace(os.sep, "/")

    def model_absolute_path(self, relative_path: str) -> str:
        if not isinstance(relative_path, str) or not relative_path or os.path.isabs(relative_path):
            raise ValueError("model path must be a non-empty relative path")
        normalized = relative_path.replace("\\", "/")
        if normalized in ("", ".") or ".." in Path(normalized).parts:
            raise ValueError("model path must not contain path traversal")
        return self._inside_models(
            os.path.join(self.models_dir, *normalized.split("/"))
        )

    def ensure_parent(self, path: str) -> None:
        path, root = self._inside_managed(path)
        parent = os.path.dirname(path)
        _make_directories_durable(parent, stop_at=root)

    def create_temporary(self, destination: str) -> tuple[BinaryIO, str]:
        destination, _ = self._inside_managed(destination)
        self.ensure_parent(destination)
        descriptor, path = tempfile.mkstemp(
            dir=os.path.dirname(destination),
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
        )
        return os.fdopen(descriptor, "w+b"), path

    def durable_replace(self, temporary_path: str, destination: str) -> str:
        temporary_path, temporary_root = self._inside_managed(temporary_path)
        destination, destination_root = self._inside_managed(destination)
        if temporary_root != destination_root:
            raise ValueError("temporary artifact and destination use different filesystems")
        if os.path.dirname(temporary_path) != os.path.dirname(destination):
            raise ValueError("temporary artifact must be beside its destination")
        _fsync_file(temporary_path)
        os.replace(temporary_path, destination)
        fsync_directory(os.path.dirname(destination))
        return destination

    @contextmanager
    def staged_file(self, destination: str) -> Iterator[tuple[BinaryIO, str]]:
        file, temporary_path = self.create_temporary(destination)
        try:
            yield file, temporary_path
            file.flush()
            os.fsync(file.fileno())
            file.close()
            os.replace(temporary_path, destination)
            fsync_directory(os.path.dirname(destination))
        except BaseException:
            if not file.closed:
                file.close()
            _unlink_if_exists(temporary_path)
            raise

    def atomic_write_bytes(self, destination: str, data: bytes) -> str:
        with self.staged_file(destination) as (target, _):
            target.write(data)
        return destination

    def atomic_write_json(self, destination: str, document: dict) -> str:
        data = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.atomic_write_bytes(destination, data)

    def remove(self, path: str) -> bool:
        path, _ = self._inside_managed(path)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            return False
        fsync_directory(os.path.dirname(path))
        return True

    def disk_usage(self) -> shutil._ntuple_diskusage:
        return shutil.disk_usage(self.runtime_dir)

    def cleanup_temporary_files(self) -> tuple[str, ...]:
        """Remove definitively orphaned sibling temp artifacts at startup.

        The caller must hold the runtime-directory process lock. This pass
        runs before PostgreSQL recovery so crash-left temp files release tmpfs
        space first.
        """
        removed: list[str] = []
        for root, directories, files in os.walk(self.runtime_dir, topdown=False):
            for name in files:
                if not name.endswith(".tmp"):
                    continue
                candidate = os.path.join(root, name)
                if self.remove(candidate):
                    removed.append(self.relative_path(candidate))
            for name in directories:
                if not name.endswith(".tmp"):
                    continue
                candidate = os.path.join(root, name)
                if self.remove(candidate):
                    removed.append(self.relative_path(candidate))
        return tuple(sorted(set(removed)))

    def reconcile(
        self,
        referenced_paths: Sequence[str] | set[str],
        *,
        temporary_paths: Sequence[str] = (),
        known_job_ids: Sequence[str] | set[str] | None = None,
    ) -> dict:
        """Remove startup leftovers without touching ledger-referenced data."""
        referenced = {self.absolute_path(path) for path in referenced_paths}
        removed: list[str] = []

        if known_job_ids is not None and os.path.isdir(self.jobs_dir):
            known = {
                _uuid_component(job_id, "job_id")
                for job_id in known_job_ids
            }
            for name in os.listdir(self.jobs_dir):
                candidate = os.path.join(self.jobs_dir, name)
                if not os.path.isdir(candidate) or os.path.islink(candidate):
                    continue
                try:
                    job_id = _uuid_component(name, "job_id")
                except ValueError:
                    continue
                if job_id in known:
                    continue
                if self.remove(candidate):
                    removed.append(self.relative_path(candidate))

        for relative in temporary_paths:
            try:
                candidate = self.absolute_path(relative)
            except ValueError:
                continue
            if self.remove(candidate):
                removed.append(self.relative_path(candidate))

        for root, directories, files in os.walk(self.runtime_dir, topdown=False):
            for name in files:
                candidate = os.path.join(root, name)
                if name.endswith(".tmp"):
                    if self.remove(candidate):
                        removed.append(self.relative_path(candidate))
                    continue
                if name == "checkpoint.pth" and _is_attempt_checkpoint(
                    candidate,
                    self.jobs_dir,
                ):
                    # Attempt checkpoints are never published in place.  A
                    # successful fit owns a copy under models/<modelRef>/;
                    # any checkpoint left below an attempt after restart is
                    # therefore unpublished and must not survive recovery.
                    if candidate not in referenced and self.remove(candidate):
                        removed.append(self.relative_path(candidate))
                    continue
                if not name.endswith(".arrow"):
                    continue
                if candidate in referenced:
                    continue
                if _is_artifact_area(candidate, self.jobs_dir):
                    if self.remove(candidate):
                        removed.append(self.relative_path(candidate))
            for name in directories:
                candidate = os.path.join(root, name)
                if name.endswith(".tmp") and self.remove(candidate):
                    removed.append(self.relative_path(candidate))

        return {"removed": sorted(set(removed))}

    def reconcile_model_directories(self, model_refs: Sequence[str] | set[str]) -> tuple[str, ...]:
        known = {_safe_component(value, "model_ref") for value in model_refs}
        removed = []
        for name in os.listdir(self.models_dir):
            if not name.startswith("mdl_") or name in known:
                continue
            candidate = os.path.join(self.models_dir, name)
            if os.path.isdir(candidate) and not os.path.islink(candidate) and self.remove(candidate):
                removed.append(name)
        return tuple(sorted(removed))

    def _inside_runtime(self, path: str) -> str:
        return self._inside_root(path, self.runtime_dir, "runtime")

    def _inside_models(self, path: str) -> str:
        return self._inside_root(path, self.models_dir, "models")

    def _inside_managed(self, path: str) -> tuple[str, str]:
        candidate = os.path.abspath(os.fspath(path))
        for root in (self.runtime_dir, self.models_dir):
            if os.path.commonpath((root, candidate)) == root:
                label = "runtime" if root == self.runtime_dir else "models"
                return self._inside_root(candidate, root, label), root
        raise ValueError("artifact path escapes managed storage")

    @staticmethod
    def _inside_root(path: str, root: str, label: str) -> str:
        candidate = os.path.abspath(os.fspath(path))
        if os.path.commonpath((root, candidate)) != root:
            raise ValueError(f"artifact path escapes the {label} directory")
        real_root = os.path.realpath(root)
        real_candidate = os.path.realpath(candidate)
        if os.path.commonpath((real_root, real_candidate)) != real_root:
            raise ValueError("artifact path escapes through a symlink")
        return candidate


def fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_directories_durable(path: str, *, stop_at: str) -> None:
    missing = []
    current = path
    while current != stop_at and not os.path.exists(current):
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            raise ValueError("directory escapes managed storage")
        current = parent
    os.makedirs(path, exist_ok=True)
    for directory in reversed(missing):
        fsync_directory(os.path.dirname(directory))


def _unlink_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _is_artifact_area(path: str, jobs_dir: str) -> bool:
    relative = os.path.relpath(path, jobs_dir).replace(os.sep, "/")
    parts = relative.split("/")
    return "inputs" in parts or "outputs" in parts


def _is_attempt_checkpoint(path: str, jobs_dir: str) -> bool:
    relative = os.path.relpath(path, jobs_dir).replace(os.sep, "/")
    parts = relative.split("/")
    return (
        len(parts) == 4
        and parts[1] == "attempts"
        and parts[2].isdigit()
        and int(parts[2]) > 0
        and parts[3] == "checkpoint.pth"
    )


def _uuid_component(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} contains unsafe characters")
    if value in (".", ".."):
        raise ValueError(f"{label} contains unsafe characters")
    return value


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
