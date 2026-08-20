from __future__ import annotations

import re
from datetime import UTC, datetime

from sqlalchemy import func, select

from app.service.adapters.outbound.postgres.ledger.support import advisory_lock
from app.service.adapters.outbound.postgres.models import (
    Job,
    ModelAlias,
    PublishedModel,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.job import TERMINAL_EXECUTION_STATES
from app.service.domain.model import (
    ModelDeletionBlocked,
    ModelLifecycleState,
)
from app.service.domain.records import ModelLifecycleRecord

_MODEL_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL_EXECUTION_STATES = tuple(
    state.value for state in TERMINAL_EXECUTION_STATES
)


class PublishedModelStore:
    """Own administrative lifecycle changes for published model generations."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def list_models(self) -> list[ModelLifecycleRecord]:
        with self.database.session() as session:
            rows = session.scalars(
                select(PublishedModel)
                .order_by(
                    PublishedModel.owner_subject,
                    PublishedModel.label,
                    PublishedModel.generation,
                )
            ).all()
            return [_record(model) for model in rows]

    def request_deletion(self, model_ref: str) -> ModelLifecycleRecord:
        model_ref = _model_ref(model_ref)
        with self.database.transaction() as session:
            # Publication uses the same transaction-scoped generation lock.
            # The first read discovers its owner/label key; the second read
            # observes the winning lifecycle state under a row lock.
            candidate = session.get(PublishedModel, model_ref)
            if candidate is None:
                raise LookupError(f"model generation not found: {model_ref}")

            advisory_lock(
                session,
                "model-generation",
                candidate.owner_subject,
                candidate.label,
            )
            model = session.get(PublishedModel, model_ref, with_for_update=True)
            if model is None:
                raise LookupError(f"model generation not found: {model_ref}")

            state = ModelLifecycleState(model.lifecycle_state)
            if state != ModelLifecycleState.AVAILABLE:
                return _record(model)

            active_jobs = session.scalar(
                select(func.count(Job.job_id)).where(
                    Job.operation == "predict",
                    Job.resolved_model_ref == model_ref,
                    Job.execution_state.not_in(_TERMINAL_EXECUTION_STATES),
                )
            )
            if active_jobs:
                raise ModelDeletionBlocked(
                    f"model generation has {active_jobs} active predict job(s): "
                    f"{model_ref}"
                )

            now = datetime.now(UTC)
            aliases = session.scalars(
                select(ModelAlias)
                .where(ModelAlias.model_ref == model_ref)
                .with_for_update()
            ).all()
            for alias in aliases:
                session.delete(alias)

            model.lifecycle_state = ModelLifecycleState.DELETING.value
            model.deletion_requested_at = now
            session.flush()
            return _record(model)

    def pending_deletions(self, *, limit: int = 100) -> tuple[str, ...]:
        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("model deletion limit must be positive")
        with self.database.session() as session:
            rows = session.scalars(
                select(PublishedModel.model_ref)
                .where(
                    PublishedModel.lifecycle_state
                    == ModelLifecycleState.DELETING.value
                )
                .order_by(
                    PublishedModel.deletion_requested_at,
                    PublishedModel.model_ref,
                )
                .limit(limit)
            ).all()
            return tuple(rows)

    def complete_deletion(self, model_ref: str) -> bool:
        model_ref = _model_ref(model_ref)
        with self.database.transaction() as session:
            model = session.get(PublishedModel, model_ref, with_for_update=True)
            if model is None:
                return False
            state = ModelLifecycleState(model.lifecycle_state)
            if state != ModelLifecycleState.DELETING:
                raise RuntimeError(
                    f"model generation is not pending deletion: {model_ref}"
                )

            # The request transaction already removes the current alias. The
            # foreign key cascade is the final guard against a stale alias.
            session.delete(model)
            session.flush()
            return True

    def retained_model_refs(self) -> set[str]:
        with self.database.session() as session:
            rows = session.scalars(
                select(PublishedModel.model_ref).where(
                    PublishedModel.lifecycle_state.in_((
                        ModelLifecycleState.AVAILABLE.value,
                        ModelLifecycleState.DELETING.value,
                    ))
                )
            ).all()
            return set(rows)


def _model_ref(value: object) -> str:
    if not isinstance(value, str) or not _MODEL_REF.fullmatch(value):
        raise ValueError(
            "model reference must be 1-128 safe identifier characters"
        )
    return value


def _record(
    model: PublishedModel,
) -> ModelLifecycleRecord:
    return ModelLifecycleRecord(
        model_ref=model.model_ref,
        owner_subject=model.owner_subject,
        label=model.label,
        generation=model.generation,
        state=ModelLifecycleState(model.lifecycle_state),
        created_at=model.created_at.timestamp(),
        deletion_requested_at=(
            None
            if model.deletion_requested_at is None
            else model.deletion_requested_at.timestamp()
        ),
    )


__all__ = ["PublishedModelStore"]
