from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v1 import validate_catalog_document
from app.project import PROJECT_ROOT
from app.service.adapters.inbound.flight.constants import (
    MODEL_CATALOG_DETAIL_ACTION,
    MODEL_CATALOG_LIST_ACTION,
)
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.adapters.inbound.flight.model_catalog import (
    CatalogModelMetadataVerifier,
    model_detail,
)
from app.service.adapters.inbound.flight.presentation import (
    present_catalog_models_page,
)
from app.service.adapters.inbound.flight.validation import validate_action_request
from app.service.adapters.outbound.artifacts.job_artifacts import (
    CatalogModelArtifactVerifier,
)
from app.service.application.messages.model_catalog import (
    CatalogModelRecord,
    GetCatalogModelQuery,
    ListCatalogModelsQuery,
    StoredCatalogPage,
)
from app.service.application.ports.model_catalog import (
    CatalogArtifactVerificationError,
    CatalogModelNotFound,
)
from app.service.application.queries.model_catalog import (
    GetCatalogModel,
    ListCatalogModels,
)
from app.service.application.services.model_catalog_cursor import (
    ExpiredCatalogCursor,
    InvalidCatalogCursor,
)
from app.service.domain.errors import ServiceError
from app.service.domain.records import PublishedModelRecord

FIXTURES = (
    PROJECT_ROOT / "app" / "contracts" / "model_catalog" / "v1" / "fixtures"
)
NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class _Store:
    def __init__(self, models: tuple[PublishedModelRecord, ...]) -> None:
        self.models = models
        self.continuation: tuple[int, datetime, str, int] | None = None

    def cursor_signing_key(self) -> bytes:
        return b"catalog-test-key".ljust(32, b"0")

    def first_page(self, owner_subject: str, *, limit: int) -> StoredCatalogPage:
        assert owner_subject == "consumer"
        entries = tuple(
            CatalogModelRecord(model, index, NOW - timedelta(seconds=index))
            for index, model in enumerate(self.models, start=1)
        )
        return StoredCatalogPage(10, entries[:limit], len(entries) > limit)

    def continuation_page(
        self,
        owner_subject: str,
        *,
        high_water_ordinal: int,
        after_created_at: datetime,
        after_model_ref: str,
        limit: int,
    ) -> StoredCatalogPage:
        assert owner_subject == "consumer"
        self.continuation = (
            high_water_ordinal,
            after_created_at,
            after_model_ref,
            limit,
        )
        remaining = self.models[1:]
        entries = tuple(
            CatalogModelRecord(model, index + 2, NOW - timedelta(seconds=index + 2))
            for index, model in enumerate(remaining)
        )
        return StoredCatalogPage(high_water_ordinal, entries[:limit], False)

    def get_model(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> CatalogModelRecord | None:
        assert owner_subject == "consumer"
        model = next(
            (item for item in self.models if item.model_ref == model_ref),
            None,
        )
        return (
            None
            if model is None
            else CatalogModelRecord(model, 1, NOW)
        )


class _Verifier:
    def __init__(self) -> None:
        self.models: list[str] = []

    def verify(self, model: PublishedModelRecord) -> None:
        self.models.append(model.model_ref)


class _ArtifactStore:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def model_absolute_path(self, relative_path: object) -> str:
        del relative_path
        return str(self.checkpoint)

    def model_checkpoint_path(self, model_ref: str) -> str:
        del model_ref
        return str(self.checkpoint)


def test_list_cursor_is_owner_page_size_high_water_and_expiry_bound() -> None:
    first_model = _model_record("detail.result.random.json")
    second_model = _model_record("detail.result.published-model.json")
    store = _Store((first_model, second_model))
    query = ListCatalogModels(
        store,
        cursor_ttl_seconds=900,
        clock=lambda: NOW,
    )

    first = query.execute(ListCatalogModelsQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        page_size=1,
        cursor=None,
    ))
    assert len(first.models) == 1
    assert first.cursor_expires_at == NOW + timedelta(seconds=900)
    assert first.next_cursor is not None

    presented = present_catalog_models_page(first)
    assert presented["cursorExpiresAt"] == "2026-09-05T12:15:00.000000Z"
    validate_catalog_document(presented, "list-result")

    second = query.execute(ListCatalogModelsQuery(
        owner_subject="consumer",
        request_id=REQUEST_ID,
        page_size=1,
        cursor=first.next_cursor,
    ))
    assert [item.model_ref for item in second.models] == [
        second_model.model_ref
    ]
    assert second.next_cursor is None
    assert second.cursor_expires_at is None
    assert store.continuation is not None
    assert store.continuation[0] == 10
    assert store.continuation[2] == first_model.model_ref

    with pytest.raises(InvalidCatalogCursor):
        query.execute(ListCatalogModelsQuery(
            owner_subject="another-consumer",
            request_id=REQUEST_ID,
            page_size=1,
            cursor=first.next_cursor,
        ))
    with pytest.raises(InvalidCatalogCursor):
        query.execute(ListCatalogModelsQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            page_size=2,
            cursor=first.next_cursor,
        ))
    expired_query = ListCatalogModels(
        store,
        cursor_ttl_seconds=900,
        clock=lambda: NOW + timedelta(seconds=900),
    )
    with pytest.raises(ExpiredCatalogCursor):
        expired_query.execute(ListCatalogModelsQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            page_size=1,
            cursor=first.next_cursor,
        ))


def test_detail_is_full_valid_projection_and_verifies_artifact() -> None:
    model = _model_record("detail.result.random.json")
    store = _Store((model,))
    verifier = _Verifier()
    result = GetCatalogModel(
        store,
        metadata_verifier=CatalogModelMetadataVerifier(),
        artifact_verifier=verifier,
    ).execute(
        GetCatalogModelQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
        )
    )
    detail = model_detail(result.model)
    document = {
        "contract": "transformer-model-catalog",
        "revision": 1,
        "requestId": result.request_id,
        "model": detail,
    }

    validate_catalog_document(document, "detail-result")
    summary = cast(JsonObject, detail["summary"])
    assert summary["producingRunId"] == model.producing_job_id
    assert verifier.models == [model.model_ref]


def test_flight_catalog_actions_use_the_accepted_documents() -> None:
    model = _model_record("detail.result.random.json")
    store = _Store((model,))
    list_query = ListCatalogModels(store, cursor_ttl_seconds=900, clock=lambda: NOW)
    detail_query = GetCatalogModel(
        store,
        metadata_verifier=CatalogModelMetadataVerifier(),
        artifact_verifier=_Verifier(),
    )
    unused = cast(Any, None)
    coordinator = JobCoordinator(
        create_job=unused,
        acquire_job=unused,
        close_input=unused,
        cancel_job=unused,
        get_status=unused,
        list_inputs=unused,
        list_outputs=unused,
        list_catalog_models=list_query,
        get_catalog_model=detail_query,
        service_status=unused,
        availability=unused,
    )
    list_document = {
        "contract": "transformer-model-catalog",
        "revision": 1,
        "requestId": REQUEST_ID,
        "pageSize": 1,
        "cursor": None,
    }
    list_request = validate_action_request(
        MODEL_CATALOG_LIST_ACTION,
        list_document,
    )
    list_result = json.loads(coordinator.dispatch(
        MODEL_CATALOG_LIST_ACTION,
        "consumer",
        list_request,
        list_document,
    ))
    validate_catalog_document(list_result, "list-result")
    assert list_result["models"][0]["modelRef"] == model.model_ref

    detail_document = {
        "contract": "transformer-model-catalog",
        "revision": 1,
        "requestId": REQUEST_ID,
        "modelRef": model.model_ref,
    }
    detail_request = validate_action_request(
        MODEL_CATALOG_DETAIL_ACTION,
        detail_document,
    )
    detail_result = json.loads(coordinator.dispatch(
        MODEL_CATALOG_DETAIL_ACTION,
        "consumer",
        detail_request,
        detail_document,
    ))
    validate_catalog_document(detail_result, "detail-result")
    assert detail_result["model"]["summary"] == list_result["models"][0]


def test_catalog_revision_uses_json_integer_equivalence() -> None:
    document = {
        "contract": "transformer-model-catalog",
        "revision": 2.0,
        "requestId": REQUEST_ID,
        "pageSize": 1,
        "cursor": None,
    }

    with pytest.raises(ServiceError) as error:
        validate_action_request(MODEL_CATALOG_LIST_ACTION, document)

    assert error.value.detail is not None
    assert error.value.detail["reason"] == "CATALOG_QUERY_REVISION_UNAVAILABLE"
    assert error.value.detail["requestedRevision"] == 2


def test_detail_hides_unknown_and_foreign_models_behind_same_outcome() -> None:
    store = _Store(())
    with pytest.raises(CatalogModelNotFound):
        GetCatalogModel(
            store,
            metadata_verifier=CatalogModelMetadataVerifier(),
            artifact_verifier=_Verifier(),
        ).execute(
            GetCatalogModelQuery(
                owner_subject="consumer",
                request_id=REQUEST_ID,
                model_ref="mdl_99999999999999999999999999999999",
            )
        )


def test_detail_maps_concurrent_deletion_to_model_not_found() -> None:
    model = _model_record("detail.result.random.json")

    class _DeletingStore(_Store):
        def __init__(self) -> None:
            super().__init__((model,))
            self.reads = 0

        def get_model(
            self,
            owner_subject: str,
            model_ref: str,
        ) -> CatalogModelRecord | None:
            self.reads += 1
            if self.reads > 1:
                return None
            return super().get_model(owner_subject, model_ref)

    class _UnavailableArtifact:
        def verify(self, candidate: PublishedModelRecord) -> None:
            raise CatalogArtifactVerificationError(
                "MODEL_CHECKPOINT_UNAVAILABLE",
                modelRef=candidate.model_ref,
            )

    with pytest.raises(CatalogModelNotFound):
        GetCatalogModel(
            _DeletingStore(),
            metadata_verifier=CatalogModelMetadataVerifier(),
            artifact_verifier=_UnavailableArtifact(),
        ).execute(GetCatalogModelQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
        ))


def test_invalid_registry_metadata_is_not_returned_as_partial_detail() -> None:
    model = _model_record("detail.result.random.json")
    model.metadata["jobId"] = "not-a-uuid"
    with pytest.raises(ServiceError) as error:
        model_detail(model)
    assert error.value.detail is not None
    assert error.value.detail["reason"] == "STORED_MODEL_METADATA_INVALID"


def test_detail_rejects_metadata_before_reading_a_corrupt_artifact() -> None:
    model = _model_record("detail.result.random.json")
    model.metadata["jobId"] = "not-a-uuid"

    class _CorruptArtifact:
        called = False

        def verify(self, candidate: PublishedModelRecord) -> None:
            self.called = True
            raise CatalogArtifactVerificationError(
                "MODEL_CHECKPOINT_DIGEST_MISMATCH",
                modelRef=candidate.model_ref,
                expectedSha256="a" * 64,
                actualSha256="b" * 64,
            )

    artifact = _CorruptArtifact()
    query = GetCatalogModel(
        _Store((model,)),
        metadata_verifier=CatalogModelMetadataVerifier(),
        artifact_verifier=artifact,
    )

    with pytest.raises(ServiceError) as error:
        query.execute(GetCatalogModelQuery(
            owner_subject="consumer",
            request_id=REQUEST_ID,
            model_ref=model.model_ref,
        ))

    assert error.value.detail is not None
    assert error.value.detail["reason"] == "STORED_MODEL_METADATA_INVALID"
    assert artifact.called is False


def test_catalog_artifact_verifier_reports_size_and_digest_separately(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    model = _model_record("detail.result.random.json")
    model = replace(
        model,
        checkpoint_path="models/example/checkpoint.pth",
        byte_count=len(b"checkpoint") + 1,
    )
    store = _ArtifactStore(checkpoint)
    with pytest.raises(CatalogArtifactVerificationError) as size_error:
        CatalogModelArtifactVerifier(
            store,
            max_verification_bytes=1024**3,
        ).verify(model)
    assert size_error.value.reason == "MODEL_CHECKPOINT_SIZE_MISMATCH"


def _model_record(fixture_name: str) -> PublishedModelRecord:
    result = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    detail = result["model"]
    summary = detail["summary"]
    metadata = {
        "format": "transformer-checkpoint-v6",
        "serviceVersion": "0.2.0",
        "generation": detail["progress"]["completedEpochs"],
        "jobId": summary["producingRunId"],
        "dataContract": detail["dataContract"],
        "modelContract": detail["modelContract"],
        "semanticDigests": summary["semanticDigests"],
        "trainingConfig": detail["trainingConfig"],
        "diagnostics": detail["diagnostics"],
        "selection": detail["selection"],
        "initialization": detail["initialization"],
        "jobConfigSha256": detail["jobConfigSha256"],
        "manifestSha256": "f" * 64,
        "progress": detail["progress"],
        "modelRef": summary["modelRef"],
        "label": summary["label"],
        "checkpoint": summary["checkpoint"],
    }
    created_at = datetime.fromisoformat(summary["createdAt"]).timestamp()
    return PublishedModelRecord(
        model_ref=summary["modelRef"],
        owner_subject="consumer",
        label=summary["label"],
        generation=summary["generation"],
        checkpoint_path=f"models/{summary['modelRef']}/checkpoint.pth",
        byte_count=summary["checkpoint"]["bytes"],
        sha256=summary["checkpoint"]["sha256"],
        metadata=metadata,
        data_contract=detail["dataContract"],
        model_contract=detail["modelContract"],
        semantic_digests=summary["semanticDigests"],
        producing_job_id=summary["producingRunId"],
        created_at=created_at,
    )
