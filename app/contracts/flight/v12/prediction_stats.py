from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionArrowStats:
    rows: int
    batches: int
    schema_fingerprint: str


__all__ = ["PredictionArrowStats"]

