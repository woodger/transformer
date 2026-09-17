from __future__ import annotations

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


__all__ = [
    "EpochTelemetry",
    "ObservedTrainingEpoch",
    "TargetErrorObservation",
    "epoch_telemetry_document",
]
