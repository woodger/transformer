from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

import rfc8785

from app.contracts.semantic.v4 import ModelContract
from app.contracts.target_head_diagnostics.v2.constants import (
    ARTIFACT_FORMAT,
    MAX_COMMITTED_ARTIFACT_ROWS,
)
from app.contracts.worker.v17 import validate_document
from app.contracts.worker.v17.diagnostics import (
    TARGET_HEAD_FULL_COMMITTED_ARTIFACT,
)
from app.service.application.messages.target_head_diagnostics import (
    GetTargetHeadDiagnosticsReportQuery,
)
from app.service.application.ports.model_catalog import (
    CatalogMetadataVerifier,
    CatalogModelNotFound,
    ModelCatalogStore,
)
from app.service.application.services.target_head_diagnostics_cursor import (
    TargetHeadDiagnosticsCursorError,
    decode_target_head_diagnostics_cursor,
    encode_target_head_diagnostics_cursor,
)
from app.service.application.services.training_telemetry_snapshot import (
    TrainingTelemetrySnapshotStore,
)
from app.service.domain.json_types import JsonObject, JsonValue
from app.service.domain.records import PublishedModelRecord


class GetTargetHeadDiagnosticsReport:
    def __init__(
        self,
        store: ModelCatalogStore,
        snapshots: TrainingTelemetrySnapshotStore,
        *,
        metadata_verifier: CatalogMetadataVerifier,
        cursor_ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._snapshots = snapshots
        self._metadata_verifier = metadata_verifier
        self._cursor_ttl_seconds = cursor_ttl_seconds
        self._clock = clock
        self._cursor_key: bytes | None = None
        self._cursor_lock = threading.Lock()

    def execute(
        self,
        query: GetTargetHeadDiagnosticsReportQuery,
    ) -> JsonObject:
        now = self._clock().astimezone(UTC)
        model = self._visible_model(query.owner_subject, query.model_ref)
        metadata = model.metadata
        if not _target_head_configured(metadata):
            self._ensure_visible(query.owner_subject, model.model_ref)
            return {
                "requestId": query.request_id,
                "state": "notConfigured",
                "modelRef": model.model_ref,
                "reason": "TARGET_HEAD_DIAGNOSTICS_NOT_CONFIGURED",
            }
        if _target_head_format_unsupported(metadata):
            self._ensure_visible(query.owner_subject, model.model_ref)
            return {
                "requestId": query.request_id,
                "state": "unavailable",
                "modelRef": model.model_ref,
                "reason": "FORMAT_UNSUPPORTED",
            }

        producing_run_id = _string(metadata, "jobId")
        cursor = None
        if query.cursor is not None:
            cursor = decode_target_head_diagnostics_cursor(
                self._signing_key(),
                query.owner_subject,
                query.cursor,
                model_ref=model.model_ref,
                producing_run_id=producing_run_id,
                page_size=query.page_size,
                now=now,
                boot_identity=self._snapshots.boot_identity,
            )
        report = self._retained_report(
            query.owner_subject,
            model.model_ref,
            producing_run_id,
            cursor,
            now,
        )
        if report is None:
            report = _report_from_metadata(model)
        if report is None:
            self._ensure_visible(query.owner_subject, model.model_ref)
            return {
                "requestId": query.request_id,
                "state": "unavailable",
                "modelRef": model.model_ref,
                "reason": "NO_COMPLETE_REPORT",
            }

        after_epoch = 0 if cursor is None else _positive_integer(
            cursor["afterEpoch"],
            "cursor after epoch",
        )
        epochs = _array(report, "epochs")
        if after_epoch >= len(epochs):
            raise TargetHeadDiagnosticsCursorError("invalid")
        page = epochs[after_epoch : after_epoch + query.page_size]
        terminal = after_epoch + len(page) == len(epochs)
        expires_at = (
            now + timedelta(seconds=self._cursor_ttl_seconds)
            if cursor is None
            else cast(datetime, cursor["expiresAt"])
        )
        next_cursor = None
        cursor_expires_at = None
        if not terminal:
            retained = self._snapshots.admit(
                _snapshot_key(
                    query.owner_subject,
                    model.model_ref,
                    producing_run_id,
                    _string(report, "reportIdentity"),
                ),
                report,
                _snapshot_projection(report),
                operation="targetHeadDiagnostics",
                expires_at=expires_at,
                now=now,
            )
            if not isinstance(retained, dict):
                raise RuntimeError("retained target head report is invalid")
            report = cast(JsonObject, retained)
            next_cursor = encode_target_head_diagnostics_cursor(
                self._signing_key(),
                query.owner_subject,
                model_ref=model.model_ref,
                producing_run_id=producing_run_id,
                report_identity=_string(report, "reportIdentity"),
                page_size=query.page_size,
                after_epoch=_positive_integer(
                    page[-1]["epoch"],
                    "epoch",
                ),
                expires_at=expires_at,
                boot_identity=self._snapshots.boot_identity,
            )
            cursor_expires_at = _timestamp(expires_at)

        self._ensure_visible(query.owner_subject, model.model_ref)
        return {
            "requestId": query.request_id,
            "state": "available",
            "modelRef": model.model_ref,
            "producingRunId": producing_run_id,
            "modelDefinitionSha256": _string(report, "modelDefinitionSha256"),
            "sampleIdentity": _string(report, "sampleIdentity"),
            "sampleRowCount": _positive_integer(
                report["sampleRowCount"],
                "sample row count",
            ),
            "layout": [dict(item) for item in _array(report, "layout")],
            "coverage": dict(_object(report, "coverage")),
            "epochPage": {
                "items": [dict(item) for item in page],
                "nextCursor": next_cursor,
                "cursorExpiresAt": cursor_expires_at,
            },
        }

    def _visible_model(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> PublishedModelRecord:
        entry = self._store.get_model(owner_subject, model_ref)
        if entry is None:
            raise CatalogModelNotFound(model_ref)
        self._metadata_verifier.verify(entry.model)
        return entry.model

    def _ensure_visible(
        self,
        owner_subject: str,
        model_ref: str,
    ) -> None:
        if self._store.get_model(owner_subject, model_ref) is None:
            raise CatalogModelNotFound(model_ref)

    def _retained_report(
        self,
        owner_subject: str,
        model_ref: str,
        producing_run_id: str,
        cursor: Mapping[str, object] | None,
        now: datetime,
    ) -> JsonObject | None:
        if cursor is None:
            return None
        report = self._snapshots.get(
            _snapshot_key(
                owner_subject,
                model_ref,
                producing_run_id,
                _string(cursor, "reportIdentity"),
            ),
            now=now,
        )
        if report is None:
            raise TargetHeadDiagnosticsCursorError("invalid")
        if not isinstance(report, dict):
            raise RuntimeError("retained target head report is invalid")
        return cast(JsonObject, report)

    def _signing_key(self) -> bytes:
        with self._cursor_lock:
            if self._cursor_key is None:
                key = self._store.cursor_signing_key()
                if len(key) < 32:
                    raise ValueError(
                        "target head cursor signing key must contain 32 bytes"
                    )
                self._cursor_key = key
            return self._cursor_key


def _report_from_metadata(model: PublishedModelRecord) -> JsonObject | None:
    artifact = model.metadata.get("targetHeadDiagnostics")
    if artifact is None:
        return None
    if not isinstance(artifact, dict):
        raise ValueError("target head diagnostics artifact is invalid")
    document = cast(JsonObject, artifact)
    validate_document(document, "target-head-diagnostics-artifact")
    _validate_artifact(model, document)
    return {
        "reportIdentity": hashlib.sha256(rfc8785.dumps(document)).hexdigest(),
        "modelDefinitionSha256": document["modelDefinitionSha256"],
        "sampleIdentity": document["sampleIdentity"],
        "sampleRowCount": document["sampleRowCount"],
        "layout": [dict(item) for item in _array(document, "layout")],
        "coverage": dict(_object(document, "coverage")),
        "epochs": [dict(item) for item in _array(document, "epochs")],
    }


def _validate_artifact(
    model: PublishedModelRecord,
    artifact: JsonObject,
) -> None:
    metadata = model.metadata
    if (
        artifact.get("jobId") != metadata.get("jobId")
        or artifact.get("modelDefinitionSha256")
        != model.semantic_digests.get("modelDefinitionSha256")
        or artifact.get("jobConfigSha256") != metadata.get("jobConfigSha256")
        or artifact.get("manifestSha256") != metadata.get("manifestSha256")
        or _positive_integer(artifact.get("sampleRowCount"), "sample row count")
        > MAX_COMMITTED_ARTIFACT_ROWS
    ):
        raise ValueError("target head diagnostics artifact differs from model")
    contract = ModelContract.from_document(model.model_contract)
    tuning = contract.model_tuning
    expected_encoder_layers = _positive_integer(
        tuning.get("encoderLayerCount"),
        "encoder layer count",
    )
    expected_layout = [
        {
            "targetIdentity": identity,
            "targetIndex": index,
            "directComponentIdentity": str(component["identity"]),
        }
        for index, (identity, component) in enumerate(
            zip(
                contract.target_identities,
                contract.direct_components,
                strict=True,
            )
        )
    ]
    if artifact.get("layout") != expected_layout:
        raise ValueError("target head diagnostics layout differs from model")
    progress = _object(metadata, "progress")
    completed_epochs = _positive_integer(
        progress.get("completedEpochs"),
        "completed epochs",
    )
    if artifact.get("coverage") != {"completedEpochs": completed_epochs}:
        raise ValueError("target head diagnostics coverage differs from model")

    epochs = _array(artifact, "epochs")
    if len(epochs) != completed_epochs:
        raise ValueError("target head diagnostics epochs are incomplete")
    previous_step = -1
    sample_rows = _positive_integer(
        artifact.get("sampleRowCount"),
        "sample row count",
    )
    for epoch_index, epoch in enumerate(epochs, start=1):
        if _positive_integer(epoch.get("epoch"), "epoch") != epoch_index:
            raise ValueError("target head diagnostics epochs are not consecutive")
        global_step = _nonnegative_integer(epoch.get("globalStep"), "global step")
        if global_step <= previous_step:
            raise ValueError("target head diagnostics global step is not increasing")
        previous_step = global_step
        _validate_representation_flow(
            _object(epoch, "representationFlow"),
            expected_encoder_layers,
        )
        target_heads = _array(epoch, "targetHeads")
        if len(target_heads) != len(expected_layout):
            raise ValueError("target head diagnostics target layout differs")
        for target_head in target_heads:
            _validate_target_head(target_head, sample_rows)


def _validate_target_head(
    target_head: Mapping[str, JsonValue],
    sample_rows: int,
) -> None:
    _validate_distribution(_object(target_head, "rawLogit"), sample_rows)
    _validate_distribution(_object(target_head, "publicPrediction"), sample_rows)
    count = _nonnegative_integer(
        target_head.get("gradientBatchCount"),
        "gradient batch count",
    )
    mean = target_head.get("gradientL2Mean")
    maximum = target_head.get("gradientL2Maximum")
    if count == 0:
        if mean is not None or maximum is not None:
            raise ValueError("empty target head gradient has values")
    else:
        mean_value = _nonnegative_finite(mean, "gradient mean")
        maximum_value = _nonnegative_finite(maximum, "gradient maximum")
        if mean_value > maximum_value:
            raise ValueError("target head gradient mean exceeds maximum")
    _nonnegative_finite(
        target_head.get("weightL2AfterEpoch"),
        "target head weight norm",
    )
    _finite(target_head.get("biasAfterEpoch"), "target head bias")


def _validate_representation_flow(
    flow: Mapping[str, JsonValue],
    expected_encoder_layers: int,
) -> None:
    _validate_row_representation(
        _object(flow, "encoderInput"),
        "encoder input norm",
    )
    layers = _array(flow, "encoderLayers")
    if len(layers) != expected_encoder_layers:
        raise ValueError("target head diagnostics encoder layer count differs")
    for index, layer in enumerate(layers):
        layer_index = _nonnegative_integer(
            layer.get("layerIndex"),
            "encoder layer index",
        )
        if layer_index != index:
            raise ValueError("target head diagnostics encoder layer order differs")
        _nonnegative_finite(
            layer.get("rowCenteredL2Mean"),
            "encoder layer norm",
        )
    _validate_row_representation(
        _object(flow, "targetHeadInput"),
        "target head input norm",
    )


def _validate_row_representation(
    representation: Mapping[str, JsonValue],
    name: str,
) -> None:
    _nonnegative_finite(
        representation.get("rowCenteredL2Mean"),
        name,
    )


def _validate_distribution(
    distribution: Mapping[str, JsonValue],
    sample_rows: int,
) -> None:
    if _positive_integer(distribution.get("rowCount"), "distribution row count") != sample_rows:
        raise ValueError("target head distribution row count differs")
    minimum = _finite(distribution.get("minimum"), "distribution minimum")
    maximum = _finite(distribution.get("maximum"), "distribution maximum")
    mean = _finite(distribution.get("mean"), "distribution mean")
    _nonnegative_finite(
        distribution.get("standardDeviation"),
        "distribution standard deviation",
    )
    if minimum > mean or mean > maximum:
        raise ValueError("target head distribution bounds are invalid")


def _target_head_configured(metadata: Mapping[str, JsonValue]) -> bool:
    diagnostics = metadata.get("diagnostics")
    return (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("targetHead")
        == TARGET_HEAD_FULL_COMMITTED_ARTIFACT
    )


def _target_head_format_unsupported(metadata: Mapping[str, JsonValue]) -> bool:
    artifact = metadata.get("targetHeadDiagnostics")
    return (
        isinstance(artifact, Mapping)
        and isinstance(artifact.get("format"), str)
        and artifact["format"] != ARTIFACT_FORMAT
    )


def _snapshot_key(
    owner_subject: str,
    model_ref: str,
    producing_run_id: str,
    report_identity: str,
) -> tuple[str, ...]:
    return (
        "targetHeadDiagnostics",
        owner_subject,
        model_ref,
        producing_run_id,
        report_identity,
    )


def _snapshot_projection(report: Mapping[str, JsonValue]) -> JsonValue:
    return {
        "snapshotType": "targetHeadDiagnostics",
        "reportIdentity": report["reportIdentity"],
        "modelDefinitionSha256": report["modelDefinitionSha256"],
        "sampleIdentity": report["sampleIdentity"],
        "sampleRowCount": report["sampleRowCount"],
        "layout": report["layout"],
        "coverage": report["coverage"],
        "epochs": report["epochs"],
    }


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def _object(
    document: Mapping[str, JsonValue],
    name: str,
) -> JsonObject:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"target head diagnostics {name} is invalid")
    return cast(JsonObject, value)


def _array(
    document: Mapping[str, JsonValue],
    name: str,
) -> list[JsonObject]:
    value = document.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"target head diagnostics {name} is invalid")
    return cast(list[JsonObject], value)


def _string(document: Mapping[str, object], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"target head diagnostics {name} is invalid")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"target head diagnostics {name} is invalid")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"target head diagnostics {name} is invalid")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"target head diagnostics {name} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"target head diagnostics {name} is non-finite")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"target head diagnostics {name} is negative")
    return result


__all__ = ["GetTargetHeadDiagnosticsReport"]
