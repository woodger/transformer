from __future__ import annotations

import pytest

from app.service.bootstrap.build_identity import load_build_identity
from app.version import __version__


def test_build_identity_uses_explicit_deployed_commit(tmp_path):
    identity = load_build_identity(
        environ={"TRANSFORMER_GIT_COMMIT": "A" * 40},
        project_root=tmp_path,
    )

    assert identity.application_version == __version__
    assert identity.git_commit == "a" * 40


def test_build_identity_reads_symbolic_git_head(tmp_path):
    git_path = tmp_path / ".git"
    reference = git_path / "refs" / "heads" / "develop"
    reference.parent.mkdir(parents=True)
    (git_path / "HEAD").write_text(
        "ref: refs/heads/develop\n",
        encoding="ascii",
    )
    reference.write_text("b" * 40 + "\n", encoding="ascii")

    identity = load_build_identity(environ={}, project_root=tmp_path)

    assert identity.git_commit == "b" * 40


def test_build_identity_requires_commit_without_git_metadata(tmp_path):
    with pytest.raises(ValueError, match="TRANSFORMER_GIT_COMMIT"):
        load_build_identity(environ={}, project_root=tmp_path)
