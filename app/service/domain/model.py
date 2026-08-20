from enum import StrEnum


class ModelLifecycleState(StrEnum):
    AVAILABLE = "AVAILABLE"
    DELETING = "DELETING"


class ModelDeletionBlocked(RuntimeError):
    """The requested deletion would break an active durable responsibility."""


__all__ = ["ModelDeletionBlocked", "ModelLifecycleState"]
