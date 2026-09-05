class CheckpointCorrupt(ValueError):
    """Checkpoint claims v6 but its contents violate the frozen contract."""


__all__ = ["CheckpointCorrupt"]
