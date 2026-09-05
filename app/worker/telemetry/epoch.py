from __future__ import annotations

import math

from app.contracts.json_types import JsonObject, JsonValue
from app.worker.telemetry.epoch_observations import EpochTelemetry
from app.worker.telemetry.observed_epoch import ObservedTrainingEpoch
from app.worker.telemetry.target_errors import TargetErrorObservation


def epoch_telemetry_document(
    result: ObservedTrainingEpoch,
    **extra: JsonValue,
) -> JsonObject | None:
    telemetry = result.telemetry
    if telemetry is None:
        return None
    telemetry.finalize_gradient_statistics()
    return {
        **extra,
        "rows": result.rows,
        "batches": result.batches,
        "loss": result.loss,
        "directLosses": [
            {
                "componentIdentity": component_identity,
                "operator": operator,
                "targetIdentity": result.targets[index],
                "targetIndex": index,
                "value": result.direct_loss_values[component_identity],
            }
            for index, (component_identity, operator) in enumerate(
                result.direct_components
            )
        ],
        "auxiliaryLosses": [
            {
                "componentIdentity": component_identity,
                "operator": operator,
                "value": result.auxiliary_loss_values[component_identity],
            }
            for component_identity, operator in result.auxiliary_components
        ],
        "targetMetrics": [
            {
                "targetIdentity": target,
                "targetIndex": index,
                "mae": telemetry.target_mae[target],
                "rmse": telemetry.target_rmse[target],
            }
            for index, target in enumerate(result.targets)
        ],
        "gradientInteractions": telemetry.gradient_interactions_document(),
        "selectionScore": result.selection_score,
        "trainingBatchesCompleted": telemetry.training_batches_completed,
        "optimizerUpdatesApplied": telemetry.optimizer_updates_applied,
        "optimizerUpdatesSkipped": telemetry.optimizer_updates_skipped,
        "ampOverflowBatches": telemetry.amp_overflow_batches,
        "finiteGradientBatches": telemetry.finite_gradient_batches,
        "nonFiniteGradientBatches": telemetry.non_finite_gradient_batches,
        "preClipGradientNormMean": telemetry.pre_clip_gradient_norm_mean,
        "preClipGradientNormMax": telemetry.pre_clip_gradient_norm_max,
        "preClipGradientNormP95": telemetry.pre_clip_gradient_norm_p95,
        "nanRatio": telemetry.nan_ratio,
        "maskedTokenRatio": telemetry.masked_token_ratio,
        "completeTokenRatio": telemetry.complete_token_ratio,
        "partialTokenRatio": telemetry.partial_token_ratio,
        "emptyTokenRatio": telemetry.empty_token_ratio,
        "inputPipelineMs": telemetry.input_pipeline_ms,
        "missingStatsMs": telemetry.missing_stats_ms,
        "hostToDeviceMs": telemetry.host_to_device_ms,
        "trainStepMs": telemetry.train_step_ms,
        "elapsedMs": telemetry.elapsed_ms,
        "step": result.step,
        "lr": result.lr,
    }


def format_epoch_log_line(
    result: ObservedTrainingEpoch,
    **extra: object,
) -> str:
    telemetry = result.telemetry
    gradient_mean = None
    if telemetry is not None:
        telemetry.finalize_gradient_statistics()
        gradient_mean = telemetry.pre_clip_gradient_norm_mean
    fields: dict[str, object] = {
        **extra,
        "loss": f"{result.loss:.6f}",
        **{
            f"loss_{identity}": f"{result.direct_loss_values[identity]:.6f}"
            for identity, _operator in result.direct_components
        },
        **{
            f"loss_{identity}": f"{result.auxiliary_loss_values[identity]:.6f}"
            for identity, _operator in result.auxiliary_components
        },
        "grad_mean": "n/a" if gradient_mean is None else f"{gradient_mean:.3f}",
        "rows": result.rows,
        "batches": result.batches,
        "nan": "n/a" if telemetry is None else f"{telemetry.nan_ratio:.4f}",
        "masked_tokens": (
            "n/a" if telemetry is None else f"{telemetry.masked_token_ratio:.4f}"
        ),
        "step": result.step,
        "lr": f"{result.lr:.6g}",
        "ms": "n/a" if telemetry is None else f"{telemetry.elapsed_ms:.0f}",
    }
    return " ".join(f"{key}={value}" for key, value in fields.items())


def format_epoch_console_line(
    result: ObservedTrainingEpoch,
    **extra: object,
) -> str:
    telemetry = result.telemetry
    if telemetry is not None:
        telemetry.finalize_gradient_statistics()
    fields: list[str] = []
    for key in ("frame", "epoch"):
        if key in extra and extra[key] is not None:
            fields.append(f"{key}={extra[key]}")

    selection_score = extra.get("selection_score")
    selection_text = (
        f"{selection_score:.6g}"
        if isinstance(selection_score, (int, float))
        and math.isfinite(selection_score)
        else "n/a"
    )
    gradient_mean = (
        None if telemetry is None else telemetry.pre_clip_gradient_norm_mean
    )
    elapsed_ms = None if telemetry is None else telemetry.elapsed_ms
    fields.extend([
        f"selection={selection_text}",
        f"loss={result.loss:.6f}",
        *(
            f"{target}_mae="
            + (
                "n/a"
                if telemetry is None
                else f"{telemetry.target_mae[target]:.6g}"
            )
            for target in result.targets
        ),
        (
            "grad_mean=n/a"
            if gradient_mean is None
            else f"grad_mean={gradient_mean:.3f}"
        ),
        f"rows={result.rows}",
        f"batches={result.batches}",
        "time=n/a" if elapsed_ms is None else f"time={elapsed_ms / 1000.0:.1f}s",
    ])
    return " ".join(fields)


__all__ = [
    "EpochTelemetry",
    "ObservedTrainingEpoch",
    "TargetErrorObservation",
    "epoch_telemetry_document",
    "format_epoch_console_line",
    "format_epoch_log_line",
]
