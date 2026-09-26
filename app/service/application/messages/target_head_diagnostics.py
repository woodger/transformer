from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GetTargetHeadDiagnosticsReportQuery:
    owner_subject: str
    request_id: str
    model_ref: str
    page_size: int
    cursor: str | None


__all__ = ["GetTargetHeadDiagnosticsReportQuery"]
