from dataclasses import dataclass


@dataclass(eq=False)
class TargetValueError(ValueError):
    target_identity: str
    target_index: int
    logical_row: int
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)


__all__ = ["TargetValueError"]

