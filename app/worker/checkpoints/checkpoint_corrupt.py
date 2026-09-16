class CheckpointCorrupt(ValueError):
    """Checkpoint violates the active provider checkpoint contract."""


__all__ = ["CheckpointCorrupt"]
