import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager


def resolve_artifact_path(
    path: str,
    base_dir: str,
    *,
    label: str,
) -> str:
    value = os.path.expanduser(path)
    if not value:
        raise ValueError(f"{label} must not be empty")
    if os.path.isabs(value):
        return os.path.abspath(value)

    normalized = os.path.normpath(value)
    base = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base, normalized))
    lexical_escape = os.path.commonpath((base, candidate)) != base
    symlink_escape = (
        os.path.commonpath((os.path.realpath(base), os.path.realpath(candidate)))
        != os.path.realpath(base)
    )
    if lexical_escape or symlink_escape:
        raise ValueError(
            f"relative {label} must stay inside {os.path.basename(base_dir)}/; "
            "use an absolute path explicitly"
        )
    return candidate


@contextmanager
def atomic_output_path(
    path: str,
) -> Generator[str, None, None]:
    """Yield a sibling temporary path and atomically replace the target on success."""
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or os.curdir
    os.makedirs(parent, exist_ok=True)

    file_descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(target)}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)

    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
