"""Compatibility exports for service-domain lifecycle policies."""

from app.service.domain.policies import (
    AttemptOutcomeDecision,
    CancelDecision,
    DeviceDecision,
    RecoveryDecision,
    decide_attempt_outcome,
    decide_cancel,
    decide_interrupted_attempt,
    is_terminal,
    resolve_device,
    validate_transition,
)

__all__ = [
    "AttemptOutcomeDecision",
    "CancelDecision",
    "DeviceDecision",
    "RecoveryDecision",
    "decide_attempt_outcome",
    "decide_cancel",
    "decide_interrupted_attempt",
    "is_terminal",
    "resolve_device",
    "validate_transition",
]
