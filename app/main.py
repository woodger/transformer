from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.args import parse_args

if TYPE_CHECKING:
    import torch

    from app.contracts.semantic.v2 import ModelContract
    from app.contracts.worker.v13.config import ModelConfig
    from app.local.fit import FitArguments, ModelBuilder, TrainerBuilder
    from app.local.fit_stream import FitStreamArguments
    from app.local.plot_metrics import PlotMetricsArguments
    from app.local.predict import PredictArguments
    from app.local.predict_stream import PredictStreamArguments
    from app.worker.training.trainer import Trainer


class CliArguments(Protocol):
    action: str
    flight_action: str
    tokens_action: str
    migrations_action: str
    models_action: str
    subject: str
    token_id: str
    model_ref: str
    deleted: bool
    seed: int
    deterministic: bool
    device: str
    data: str | None
    metrics_name: str | None
    model_name: str
    model_contract: str
    pred_col: str
    preds_path: str
    plots_dir: str
    host: str | None
    port: int | None
    tls_cert_file: str | None
    tls_key_file: str | None
    tls_ca_file: str | None
    tls_require_client_cert: bool | None


def get_device(value: str | None) -> torch.device:
    from app.worker.runtime.device import get_device as implementation

    if value == "gpu":
        try:
            return implementation("cuda")
        except RuntimeError as exc:
            if str(exc) == "CUDA was requested but is not available":
                raise RuntimeError(
                    "GPU was requested but is not available"
                ) from None
            raise
    if value in (None, "cpu", "auto"):
        return implementation(value)
    raise ValueError(f"Unsupported device: {value}")


def device_name(device: torch.device) -> str:
    return "gpu" if device.type == "cuda" else str(device)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    from app.worker.runtime.reproducibility import (
        configure_reproducibility as implementation,
    )

    return implementation(seed, deterministic)


def build_model(
    args_or_config: object,
    features_cpu: torch.Tensor,
    targets_cpu: torch.Tensor | None,
    device: torch.device,
    model_contract: ModelContract,
) -> torch.nn.Module:
    from app.worker.training.factory import build_model as implementation

    return implementation(
        args_or_config,
        features_cpu,
        targets_cpu,
        device,
        model_contract,
    )


def build_trainer(
    args_or_config: object,
    model: torch.nn.Module,
    device: torch.device,
    model_config: ModelConfig | None = None,
    data_contract: Mapping[str, object] | None = None,
    *,
    metrics_path: str | None = None,
    model_contract: ModelContract,
    initialization: Mapping[str, object] | None = None,
) -> Trainer:
    from app.worker.training.factory import build_trainer as implementation

    return implementation(
        args_or_config,
        model,
        device,
        model_config,
        data_contract,
        metrics_path=metrics_path,
        model_contract=model_contract,
        initialization=initialization,
    )


def fit_stream(args: FitStreamArguments, device: torch.device) -> None:
    from app.local.fit_stream import run

    return run(args, device, build_model, build_trainer)


def predict_stream(
    args: PredictStreamArguments,
    device: torch.device,
) -> None:
    from app.local.predict_stream import run

    return run(args, device, build_model, build_trainer)


def run_fit(
    args: FitArguments,
    device: torch.device,
    model_factory: ModelBuilder,
    trainer_factory: TrainerBuilder,
) -> None:
    from app.local.fit import run

    return run(args, device, model_factory, trainer_factory)


def run_predict(
    args: PredictArguments,
    device: torch.device,
    model_factory: ModelBuilder,
    trainer_factory: TrainerBuilder,
) -> None:
    from app.local.predict import run

    return run(args, device, model_factory, trainer_factory)


def run_plot_metrics(args: PlotMetricsArguments) -> None:
    from app.local.plot_metrics import run

    return run(args)


def run_gmark(args: object) -> int:
    from app.local.gmark import run

    return run(args)


def reset_metrics_log(path: str | None) -> None:
    from app.worker.telemetry import reset_metrics_log as implementation

    return implementation(path)


def resolve_metrics_path(name: str | None) -> str | None:
    from app.worker.telemetry.paths import resolve_metrics_path as implementation

    return implementation(name)


def main() -> None:
    args = cast(CliArguments, parse_args())

    if args.action == "flight" and args.flight_action == "serve":
        from app.service.bootstrap.application import run_from_args

        run_from_args(args)
        return

    if args.action == "auth":
        from app.admin.bootstrap.auth_tokens import run

        run(args)
        return

    if args.action == "db":
        from app.admin.bootstrap.db_migrations import run

        run(args)
        return

    if args.action == "models":
        from app.admin.bootstrap.models import run

        run(args)
        return

    if args.action == "plot-metrics":
        run_plot_metrics(args)
        return

    if args.action == "gmark":
        raise SystemExit(run_gmark(args))

    if args.action in ("fit", "fit-stream"):
        configure_reproducibility(args.seed, args.deterministic)

    device = get_device(args.device)

    if args.action == "predict-stream":
        print(f"Using device: {device_name(device)}", file=sys.stderr)
        if args.data is not None:
            raise ValueError("predict-stream reads stdin; data path is not supported")
        predict_stream(args, device)
        return

    print(f"Using device: {device_name(device)}")

    if args.action == "fit-stream":
        if args.data is not None:
            raise ValueError("fit-stream reads stdin; data path is not supported")
        reset_metrics_log(resolve_metrics_path(args.metrics_name))
        fit_stream(args, device)
        return

    if args.data is None:
        raise ValueError("data path is required for fit and predict")

    if args.action == "fit":
        reset_metrics_log(resolve_metrics_path(args.metrics_name))
        run_fit(args, device, build_model, build_trainer)
        return

    run_predict(args, device, build_model, build_trainer)


if __name__ == "__main__":
    main()
