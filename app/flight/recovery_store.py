from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from app.flight.spool import (
    RuntimeDirectoryLock,
    fsync_directory,
)

_CHECKPOINT_NAME = re.compile(r"^[1-9][0-9]*\.pth$")


class RecoveryStore:
    """Persistent, server-owned fit inputs and training checkpoints."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(os.fspath(root_dir))
        self.jobs_dir = os.path.join(self.root_dir, "jobs")
        self.lock = RuntimeDirectoryLock(
            os.path.join(self.root_dir, "service.lock")
        )

    def initialize(self) -> RecoveryStore:
        created = []
        for directory in (self.root_dir, self.jobs_dir):
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

    def job_directory(self, job_id: str) -> str:
        return os.path.join(
            self.jobs_dir,
            _uuid_component(job_id, "job_id"),
        )

    def input_directory(self, job_id: str) -> str:
        return os.path.join(self.job_directory(job_id), "inputs")

    def input_path(self, job_id: str, ordinal: int) -> str:
        _nonnegative(ordinal, "ordinal")
        return os.path.join(
            self.input_directory(job_id),
            f"{ordinal}.arrow",
        )

    def checkpoint_directory(self, job_id: str) -> str:
        return os.path.join(
            self.job_directory(job_id),
            "checkpoints",
        )

    def checkpoint_path(self, job_id: str, generation: int) -> str:
        _positive(generation, "generation")
        return os.path.join(
            self.checkpoint_directory(job_id),
            f"{generation}.pth",
        )

    def relative_path(self, absolute_path: str) -> str:
        resolved = self._inside_root(absolute_path)
        return os.path.relpath(
            resolved,
            self.root_dir,
        ).replace(os.sep, "/")

    def absolute_path(self, relative_path: str) -> str:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or os.path.isabs(relative_path)
        ):
            raise ValueError(
                "recovery path must be a non-empty relative path"
            )
        normalized = relative_path.replace("\\", "/")
        if normalized in ("", ".") or ".." in Path(normalized).parts:
            raise ValueError("recovery path must not contain path traversal")
        return self._inside_root(
            os.path.join(self.root_dir, *normalized.split("/"))
        )

    def ensure_parent(self, path: str) -> None:
        path = self._inside_root(path)
        parent = os.path.dirname(path)
        missing = []
        current = parent
        while (
            current != self.root_dir
            and not os.path.exists(current)
        ):
            missing.append(current)
            current = os.path.dirname(current)
        os.makedirs(parent, exist_ok=True)
        for directory in reversed(missing):
            fsync_directory(os.path.dirname(directory))

    def create_temporary(
        self,
        destination: str,
    ) -> tuple[BinaryIO, str]:
        destination = self._inside_root(destination)
        self.ensure_parent(destination)
        descriptor, path = tempfile.mkstemp(
            dir=os.path.dirname(destination),
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
        )
        return os.fdopen(descriptor, "w+b"), path

    def durable_replace(
        self,
        temporary_path: str,
        destination: str,
    ) -> str:
        temporary_path = self._inside_root(temporary_path)
        destination = self._inside_root(destination)
        if os.path.dirname(temporary_path) != os.path.dirname(destination):
            raise ValueError(
                "temporary recovery artifact must be beside its destination"
            )
        _fsync_file(temporary_path)
        os.replace(temporary_path, destination)
        fsync_directory(os.path.dirname(destination))
        return destination

    @contextmanager
    def staged_file(
        self,
        destination: str,
    ) -> Iterator[tuple[BinaryIO, str]]:
        file, temporary = self.create_temporary(destination)
        try:
            yield file, temporary
            file.flush()
            os.fsync(file.fileno())
            file.close()
            os.replace(temporary, destination)
            fsync_directory(os.path.dirname(destination))
        except BaseException:
            if not file.closed:
                file.close()
            _unlink_if_exists(temporary)
            raise

    def remove(self, path: str) -> bool:
        path = self._inside_root(path)
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
        return shutil.disk_usage(self.root_dir)

    def cleanup_temporary_files(self) -> tuple[str, ...]:
        removed = []
        for root, _, files in os.walk(self.root_dir):
            for name in files:
                if not name.endswith(".tmp"):
                    continue
                path = os.path.join(root, name)
                if self.remove(path):
                    removed.append(self.relative_path(path))
        return tuple(sorted(set(removed)))

    def reconcile(
        self,
        referenced_paths: Sequence[str] | set[str],
        *,
        known_job_ids: Sequence[str] | set[str],
        temporary_paths: Sequence[str] = (),
    ) -> tuple[str, ...]:
        referenced = {
            self.absolute_path(path)
            for path in referenced_paths
        }
        known = {
            _uuid_component(job_id, "job_id")
            for job_id in known_job_ids
        }
        removed = []
        for relative_path in temporary_paths:
            try:
                candidate = self.absolute_path(relative_path)
            except ValueError:
                continue
            if self.remove(candidate):
                removed.append(self.relative_path(candidate))
        for name in os.listdir(self.jobs_dir):
            candidate = os.path.join(self.jobs_dir, name)
            if not os.path.isdir(candidate) or os.path.islink(candidate):
                continue
            try:
                job_id = _uuid_component(name, "job_id")
            except ValueError:
                continue
            if job_id not in known:
                if self.remove(candidate):
                    removed.append(self.relative_path(candidate))
                continue
            for root, _, files in os.walk(candidate):
                for filename in files:
                    path = os.path.join(root, filename)
                    if filename.endswith(".tmp"):
                        if self.remove(path):
                            removed.append(self.relative_path(path))
                        continue
                    if (
                        filename.endswith(".arrow")
                        or _CHECKPOINT_NAME.fullmatch(filename)
                    ) and path not in referenced:
                        if self.remove(path):
                            removed.append(self.relative_path(path))
        return tuple(sorted(set(removed)))

    def _inside_root(self, path: str) -> str:
        candidate = os.path.abspath(os.fspath(path))
        if os.path.commonpath((self.root_dir, candidate)) != self.root_dir:
            raise ValueError("artifact path escapes the recovery directory")
        real_root = os.path.realpath(self.root_dir)
        real_candidate = os.path.realpath(candidate)
        if (
            os.path.commonpath((real_root, real_candidate))
            != real_root
        ):
            raise ValueError(
                "artifact path escapes through a symlink"
            )
        return candidate


def _uuid_component(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


__all__ = ["RecoveryStore"]
