from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.project import PROJECT_ROOT
from app.version import __version__

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    application_version: str
    git_commit: str


def load_build_identity(
    *,
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> BuildIdentity:
    environment = os.environ if environ is None else environ
    configured = environment.get("TRANSFORMER_GIT_COMMIT")
    commit = configured.lower() if configured else _read_git_commit(project_root)
    if commit is None or _COMMIT.fullmatch(commit) is None:
        raise ValueError(
            "TRANSFORMER_GIT_COMMIT must contain the deployed 40-character "
            "Git commit when repository metadata is unavailable"
        )
    return BuildIdentity(__version__, commit)


def _read_git_commit(project_root: Path) -> str | None:
    git_path = project_root / ".git"
    if git_path.is_file():
        marker = git_path.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            return None
        git_path = (project_root / marker.removeprefix("gitdir: ")).resolve()
    try:
        head = (git_path / "HEAD").read_text(encoding="ascii").strip()
    except OSError:
        return None
    if _COMMIT.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        return None
    reference = head.removeprefix("ref: ")
    try:
        value = (git_path / reference).read_text(encoding="ascii").strip()
    except OSError:
        value = _packed_reference(git_path, reference)
    return value if _COMMIT.fullmatch(value or "") else None


def _packed_reference(git_path: Path, reference: str) -> str | None:
    try:
        lines = (git_path / "packed-refs").read_text(
            encoding="ascii"
        ).splitlines()
    except OSError:
        return None
    suffix = f" {reference}"
    for line in lines:
        if line.startswith(("#", "^")) or not line.endswith(suffix):
            continue
        return line.split(" ", 1)[0]
    return None


__all__ = ["BuildIdentity", "load_build_identity"]
