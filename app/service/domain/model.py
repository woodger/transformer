from enum import StrEnum


class ModelLifecycleState(StrEnum):
    AVAILABLE = "AVAILABLE"
    DELETING = "DELETING"
    DELETED = "DELETED"


class ModelDeletionBlocked(RuntimeError):
    """Запрошенное удаление нарушило бы активную durable-ответственность."""


__all__ = ["ModelDeletionBlocked", "ModelLifecycleState"]
