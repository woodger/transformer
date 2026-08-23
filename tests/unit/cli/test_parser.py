from types import SimpleNamespace

import pytest

from app.cli.parser import build_parser
from app.contracts.worker.v7.config import (
    DEFAULT_DETERMINISTIC as DETERMINISTIC,
    DEFAULT_SEED as SEED,
    DEFAULT_WEIGHT_DECAY as WEIGHT_DECAY,
)
from app.service.bootstrap.config import HOST_DEFAULT, PORT_DEFAULT
from app.worker.data.arrow import DEFAULT_MAX_FRAME_BYTES
from app.worker.runtime.device import get_device
from app.worker.training.run_config import (
    ModelConfig,
    TrainConfig,
    model_config_from_args,
    train_config_from_args,
)


def parse(*args):
    return build_parser().parse_args(list(args))


def test_fit_namespace_has_only_fit_options():
    args = parse("fit", "train.arrow", "--seq-len", "20")

    assert args.action == "fit"
    assert args.data == "train.arrow"
    assert args.seq_len == 20
    assert args.device == "cpu"
    assert args.metrics_name is None
    assert args.seed == SEED
    assert args.deterministic is DETERMINISTIC
    assert args.weight_decay == WEIGHT_DECAY
    assert args.select_best_checkpoint is None
    assert args.direct_loss_weights is None
    assert not hasattr(args, "preds_path")
    assert not hasattr(args, "pred_col")
    assert not hasattr(args, "plots_dir")
    assert not hasattr(args, "max_frame_bytes")


def test_predict_namespace_keeps_checkpoint_validation_overrides():
    args = parse(
        "predict",
        "input.arrow",
        "--seq-len",
        "20",
        "--hidden",
        "128",
        "--layers",
        "3",
        "--dropout",
        "0.2",
        "--nhead",
        "4",
        "--mode",
        "strict",
    )

    assert args.action == "predict"
    assert args.data == "input.arrow"
    assert args.preds_path == "/tmp/preds.arrow"
    assert args.pred_col == "out"
    assert args.seq_len == 20
    assert args.hidden == 128
    assert args.layers == 3
    assert args.dropout == 0.2
    assert args.nhead == 4
    assert args.context_mode == "strict"
    assert args.use_amp is False
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "metrics_name")
    assert not hasattr(args, "max_frame_bytes")


def test_checkpoint_and_output_aliases_preserve_namespace_contract():
    fit_args = parse(
        "fit",
        "train.arrow",
        "--seq-len",
        "20",
        "--checkpoint-out",
        "trained.pth",
        "--metrics-out",
        "train.jsonl",
    )
    predict_args = parse(
        "predict",
        "input.arrow",
        "--checkpoint",
        "trained.pth",
        "--output",
        "predictions.arrow",
        "--use-amp",
    )

    assert fit_args.model_name == "trained.pth"
    assert fit_args.metrics_name == "train.jsonl"
    assert predict_args.model_name == "trained.pth"
    assert predict_args.preds_path == "predictions.arrow"
    assert predict_args.use_amp is True


def test_stream_namespaces_have_no_positional_input():
    fit_args = parse("fit-stream", "--seq-len", "12")
    predict_args = parse("predict-stream")

    assert fit_args.action == "fit-stream"
    assert fit_args.data is None
    assert fit_args.seq_len == 12
    assert fit_args.max_frame_bytes == DEFAULT_MAX_FRAME_BYTES
    assert fit_args.weight_decay == WEIGHT_DECAY
    assert predict_args.action == "predict-stream"
    assert predict_args.data is None
    assert predict_args.seq_len is None
    assert predict_args.max_frame_bytes == DEFAULT_MAX_FRAME_BYTES
    assert not hasattr(predict_args, "weight_decay")

    with pytest.raises(SystemExit):
        parse("fit-stream", "train.arrow", "--seq-len", "12")
    with pytest.raises(SystemExit):
        parse("predict-stream", "input.arrow")


def test_stream_frame_limit_is_configurable_and_positive():
    fit_args = parse(
        "fit-stream",
        "--seq-len",
        "12",
        "--max-frame-bytes",
        "1024",
    )
    predict_args = parse("predict-stream", "--max-frame-bytes", "2048")

    assert fit_args.max_frame_bytes == 1024
    assert predict_args.max_frame_bytes == 2048

    with pytest.raises(SystemExit):
        parse("fit-stream", "--seq-len", "12", "--max-frame-bytes", "0")
    with pytest.raises(SystemExit):
        parse("predict-stream", "--max-frame-bytes", "-1")


@pytest.mark.parametrize(
    "argv",
    (
        ("fit", "train.arrow", "--seq-len", "12", "--max-frame-bytes", "10"),
        ("predict", "input.arrow", "--max-frame-bytes", "10"),
        ("plot-metrics", "train.jsonl", "--max-frame-bytes", "10"),
    ),
)
def test_non_stream_commands_reject_frame_limit(argv):
    with pytest.raises(SystemExit):
        parse(*argv)


def test_plot_metrics_namespace_uses_required_metrics_file():
    args = parse("plot-metrics", "train.jsonl")

    assert args.action == "plot-metrics"
    assert args.data == "train.jsonl"
    assert args.metrics_name is None
    assert args.plots_dir == "metrics_plots"
    assert not hasattr(args, "device")
    assert not hasattr(args, "model_name")


def test_gmark_namespace_has_training_stress_options_only():
    args = parse(
        "gmark",
        "--duration",
        "15",
        "--device",
        "1",
        "--seq-len",
        "12",
        "--feature-dim",
        "128",
        "--batch-size",
        "16",
        "--hidden",
        "64",
        "--layers",
        "2",
        "--nhead",
        "4",
        "--use-amp",
        "--memory-fraction",
        "0.5",
        "--max-temperature",
        "75",
        "--status-interval",
        "1",
        "--warmup-steps",
        "2",
        "--seed",
        "7",
    )

    assert args.action == "gmark"
    assert args.duration == 15
    assert args.device == 1
    assert args.seq_len == 12
    assert args.feature_dim == 128
    assert args.batch_size == 16
    assert args.hidden == 64
    assert args.layers == 2
    assert args.nhead == 4
    assert args.use_amp is True
    assert args.memory_fraction == 0.5
    assert args.max_temperature == 75
    assert args.status_interval == 1
    assert args.warmup_steps == 2
    assert args.seed == 7
    assert args.data is None
    assert args.metrics_name is None
    assert not hasattr(args, "model_name")
    assert not hasattr(args, "dtype")


def test_gmark_training_profile_has_production_defaults():
    args = parse("gmark")

    assert args.seq_len == 10
    assert args.feature_dim == 891
    assert args.batch_size == 256
    assert args.hidden == 256
    assert args.layers == 5
    assert args.nhead == 8
    assert args.use_amp is False
    assert args.memory_fraction == 0.0
    assert args.seed == 42


@pytest.mark.parametrize(
    "option",
    (
        ("--duration", "0"),
        ("--device", "-1"),
        ("--seq-len", "0"),
        ("--feature-dim", "0"),
        ("--batch-size", "0"),
        ("--hidden", "0"),
        ("--layers", "0"),
        ("--nhead", "0"),
        ("--memory-fraction", "-0.1"),
        ("--memory-fraction", "0.91"),
        ("--memory-fraction", "nan"),
        ("--max-temperature", "-1"),
        ("--status-interval", "0"),
        ("--warmup-steps", "0"),
        ("--seed", str(2**32)),
    ),
)
def test_gmark_numeric_options_are_validated_by_argparse(option):
    with pytest.raises(SystemExit):
        parse("gmark", *option)


def test_access_and_database_namespaces_are_nested():
    issue = parse("auth", "tokens", "issue")
    listed = parse("auth", "tokens", "list")
    token_id = "12345678-1234-4234-8234-123456789abc"
    revoked = parse("auth", "tokens", "revoke", token_id)
    status = parse("db", "migrations", "status")
    models = parse("models", "list")
    deleted_models = parse("models", "list", "--deleted")
    deleted = parse(
        "models",
        "delete",
        "mdl_0123456789abcdef0123456789abcdef",
    )

    assert (issue.action, issue.auth_action, issue.tokens_action) == (
        "auth",
        "tokens",
        "issue",
    )
    assert not hasattr(issue, "subject")
    assert listed.tokens_action == "list"
    assert revoked.token_id == token_id
    assert (status.action, status.db_action, status.migrations_action) == (
        "db",
        "migrations",
        "status",
    )
    assert (models.action, models.models_action) == ("models", "list")
    assert models.deleted is False
    assert deleted_models.deleted is True
    assert deleted.models_action == "delete"
    assert deleted.model_ref == "mdl_0123456789abcdef0123456789abcdef"


def test_auth_tokens_issue_rejects_removed_subject_option():
    with pytest.raises(SystemExit):
        parse("auth", "tokens", "issue", "--subject", "inventory")


def test_obsolete_auth_clients_namespace_is_not_available():
    with pytest.raises(SystemExit):
        parse("auth", "clients", "list")


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("fit",),
        ("fit", "train.arrow"),
        ("predict",),
        ("fit-stream",),
        ("flight",),
        ("auth",),
        ("auth", "tokens"),
        ("auth", "tokens", "revoke"),
        ("db",),
        ("db", "migrations"),
        ("models",),
        ("models", "delete"),
        ("plot-metrics",),
    ),
)
def test_required_subcommands_and_arguments(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(list(argv))


def test_command_help_contains_only_applicable_options(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["predict", "--help"])
    assert exc.value.code == 0

    predict_help = capsys.readouterr().out
    normalized_predict_help = " ".join(predict_help.split())
    assert "Examples:" in predict_help
    assert (
        "transformer predict ./data/test.arrow --checkpoint=model.pth "
        "--output=/tmp/preds.arrow"
    ) in predict_help
    assert "--preds-path" in predict_help
    assert "--seq-len" in predict_help
    assert "--epochs" not in predict_help
    assert "--metrics-name" not in predict_help
    assert "--plots-dir" not in predict_help
    assert "read from the checkpoint" in normalized_predict_help
    assert "If specified, it must match" in normalized_predict_help

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["fit-stream", "--help"])
    assert exc.value.code == 0

    fit_stream_help = capsys.readouterr().out
    assert "Examples:" not in fit_stream_help
    assert "--epochs" in fit_stream_help
    assert "--weight-decay" in fit_stream_help
    assert "--seed" in fit_stream_help
    assert "--deterministic" in fit_stream_help
    assert "--preds-path" not in fit_stream_help
    assert "--pred-col" not in fit_stream_help
    assert "--plots-dir" not in fit_stream_help

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["plot-metrics", "--help"])
    assert exc.value.code == 0

    plot_help = capsys.readouterr().out
    assert "Examples:" not in plot_help
    assert "METRICS_FILE" in plot_help
    assert "--plots-dir" in plot_help
    assert "--device" not in plot_help
    assert "--model-name" not in plot_help
    assert "--seq-len" not in plot_help

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["gmark", "--help"])
    assert exc.value.code == 0

    gmark_help = capsys.readouterr().out
    assert "Examples:" in gmark_help
    assert "transformer gmark --duration=300 --use-amp" in gmark_help
    assert "--seq-len" in gmark_help
    assert "--feature-dim" in gmark_help
    assert "--batch-size" in gmark_help
    assert "--hidden" in gmark_help
    assert "--layers" in gmark_help
    assert "--nhead" in gmark_help
    assert "--use-amp" in gmark_help
    assert "--memory-fraction" in gmark_help
    assert "--max-temperature" in gmark_help
    assert "--checkpoint" not in gmark_help
    assert "--dtype" not in gmark_help
    assert "--matrix-size" not in gmark_help
    assert "--host" not in gmark_help

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["flight", "--help"])
    assert exc.value.code == 0

    flight_help = capsys.readouterr().out
    assert "serve" in flight_help
    assert "-h, --help" in flight_help
    assert "--host" not in flight_help
    assert "Examples:" not in flight_help

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["models", "--help"])
    assert exc.value.code == 0

    models_help = capsys.readouterr().out
    assert "list" in models_help
    assert "delete" in models_help
    assert "exact model generation" in models_help

def test_flight_serve_help_documents_configuration_contract(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["flight", "serve", "--help"])
    assert exc.value.code == 0

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())

    assert "usage: transformer flight serve [options]" in output
    assert "Run the durable Arrow Flight service for fit and predict jobs." in output

    assert "\noptions:\n" in output
    for removed_heading in (
        "Configuration:",
        "Network:",
        "Transport policy:",
        "TLS:",
        "mTLS:",
        "Authentication:",
    ):
        assert removed_heading not in output

    option_labels = (
        "--host HOST",
        "--port PORT",
        "--allow-plaintext",
        "--tls-cert-file FILE",
        "--tls-key-file FILE",
        "--tls-ca-file FILE",
        "--tls-require-client-cert",
    )
    positions = [output.index(label) for label in option_labels]
    assert positions == sorted(positions)

    assert f"Listen host. (default: {HOST_DEFAULT})" in output
    assert f"Listen port. (default: {PORT_DEFAULT})" in output
    assert "built-in default" not in output
    expected_multiline_entries = (
        "Requires --tls-key-file.",
        "Requires --tls-cert-file.",
        "Requires server TLS.",
        "Requires --tls-ca-file and server TLS.",
    )
    stripped_lines = {line.strip() for line in output.splitlines()}
    assert set(expected_multiline_entries) <= stripped_lines
    assert "Allow serving without TLS." in normalized_output
    assert "--bearer-tokens-file" not in output
    assert "--profile" not in output
    assert "--config" not in output
    assert "--state-dir" not in output
    assert "Examples:" not in output


def test_defaults_are_shown_in_command_help(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fit", "--help"])

    output = capsys.readouterr().out
    normalized_output = " ".join(output.split())
    assert "Examples:" in output
    assert "transformer fit ./data/train.arrow --seq-len=20" in output
    assert "omit to disable metrics logging" in normalized_output
    assert "(default: cpu)" in output
    assert f"(default: {SEED})" in output
    assert "(default: 0.0005)" in output


@pytest.mark.parametrize(
    "argv",
    (
        ("--help",),
        ("fit", "--help"),
        ("predict", "--help"),
        ("fit-stream", "--help"),
        ("predict-stream", "--help"),
        ("gmark", "--help"),
        ("flight", "--help"),
        ("flight", "serve", "--help"),
        ("auth", "--help"),
        ("auth", "tokens", "--help"),
        ("auth", "tokens", "issue", "--help"),
        ("auth", "tokens", "list", "--help"),
        ("auth", "tokens", "revoke", "--help"),
        ("models", "--help"),
        ("db", "migrations", "status", "--help"),
        ("db", "migrations", "apply", "--help"),
        ("db", "migrations", "rollback", "--help"),
        ("models", "list", "--help"),
        ("models", "delete", "--help"),
        ("plot-metrics", "--help"),
    ),
)
def test_help_does_not_render_internal_none_defaults(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(list(argv))
    assert exc.value.code == 0

    assert "(default: None)" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    (
        ("fit", "--help"),
        ("predict", "--help"),
        ("fit-stream", "--help"),
        ("predict-stream", "--help"),
        ("gmark", "--help"),
        ("flight", "serve", "--help"),
        ("auth", "tokens", "issue", "--help"),
        ("auth", "tokens", "list", "--help"),
        ("auth", "tokens", "revoke", "--help"),
        ("db", "migrations", "status", "--help"),
        ("db", "migrations", "apply", "--help"),
        ("db", "migrations", "rollback", "--help"),
        ("models", "list", "--help"),
        ("models", "delete", "--help"),
        ("plot-metrics", "--help"),
    ),
)
def test_leaf_help_does_not_repeat_global_help_option(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(list(argv))
    assert exc.value.code == 0

    output = capsys.readouterr().out
    assert "-h, --help" not in output
    assert "show this help message and exit" not in output


@pytest.mark.parametrize(
    "options",
    (
        ("--seq-len", "0"),
        ("--hidden", "0"),
        ("--layers", "-1"),
        ("--dropout", "-0.1"),
        ("--dropout", "1"),
        ("--dropout", "nan"),
        ("--nhead", "0"),
        ("--lr", "0"),
        ("--lr", "inf"),
        ("--weight-decay", "-1"),
        ("--weight-decay", "inf"),
        ("--weight-decay", "nan"),
        ("--batch-size", "0"),
        ("--epochs", "0"),
        ("--stage-size", "0"),
        ("--selection-patience", "-1"),
        ("--seed", "-1"),
    ),
)
def test_training_numeric_options_are_validated_by_argparse(options):
    with pytest.raises(SystemExit):
        parse("fit", "train.arrow", "--seq-len", "10", *options)


def test_zero_is_valid_for_dropout_selection_seed_and_weight_decay():
    args = parse(
        "fit",
        "train.arrow",
        "--seq-len",
        "10",
        "--dropout",
        "0",
        "--select-best-checkpoint",
        "--selection-patience",
        "0",
        "--selection-min-delta",
        "0",
        "--seed",
        "0",
        "--weight-decay",
        "0",
    )

    assert args.dropout == 0
    assert args.selection_patience == 0
    assert args.selection_min_delta == 0
    assert args.seed == 0
    assert args.weight_decay == 0


def test_model_config_rejects_hidden_not_divisible_by_nhead():
    with pytest.raises(ValueError, match=r"hidden .* must be divisible by nhead"):
        ModelConfig(seq_len=10, hidden=30, nhead=8)


def test_parser_rejects_hidden_not_divisible_by_nhead():
    with pytest.raises(SystemExit):
        parse(
            "fit",
            "train.arrow",
            "--seq-len",
            "10",
            "--hidden",
            "30",
            "--nhead",
            "8",
        )


def test_training_seed_options_are_plumbed_into_train_config():
    args = parse(
        "fit",
        "train.arrow",
        "--seq-len",
        "10",
        "--seed",
        "7",
        "--deterministic",
    )

    config = train_config_from_args(args)

    assert config.seed == 7
    assert config.deterministic is True
    assert config.to_dict()["seed"] == 7
    assert config.to_dict()["deterministic"] is True


def test_direct_loss_weights_are_plumbed_into_train_config():
    args = parse(
        "fit",
        "train.arrow",
        "--seq-len",
        "10",
        "--direct-loss-weights",
        "1,2,3,4,5,6",
    )

    config = train_config_from_args(args)

    assert config.direct_loss_weights == (1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize("action", ("fit", "fit-stream"))
def test_training_weight_decay_is_plumbed_into_train_config(action):
    positional = ("train.arrow",) if action == "fit" else ()
    args = parse(
        action,
        *positional,
        "--seq-len",
        "10",
        "--weight-decay",
        "0.0025",
    )

    config = train_config_from_args(args)

    assert config.weight_decay == 0.0025


@pytest.mark.parametrize("action", ("fit", "fit-stream"))
def test_checkpoint_selection_policy_is_plumbed_into_train_config(action):
    positional = ("train.arrow",) if action == "fit" else ()
    args = parse(
        action,
        *positional,
        "--seq-len",
        "10",
        "--select-best-checkpoint",
        "--selection-min-delta",
        "0.05",
        "--selection-patience",
        "3",
    )

    selection = train_config_from_args(args).selection
    assert selection.min_delta == 0.05
    assert selection.patience == 3


@pytest.mark.parametrize(
    "options",
    (
        ("--selection-min-delta", "-0.1"),
        ("--selection-min-delta", "nan"),
        ("--seed", str(2**32)),
    ),
)
def test_bounded_training_options_are_validated_by_argparse(options):
    with pytest.raises(SystemExit):
        parse("fit", "train.arrow", "--seq-len", "10", *options)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"seq_len": 0}, "seq_len"),
        ({"seq_len": 10, "layers": 0}, "layers"),
        ({"seq_len": 10, "dropout": 1.0}, "dropout"),
        ({"seq_len": 10, "context_mode": "unknown"}, "context_mode"),
        ({"seq_len": 10, "out_dim": 5}, "out_dim"),
        ({"seq_len": 10, "feature_dim": 0}, "feature_dim"),
    ),
)
def test_model_config_validates_programmatic_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"lr": 0}, "lr"),
        ({"batch_size": 0}, "batch_size"),
        ({"epochs": 0}, "epochs"),
        ({"loss_stage": 3}, "loss_stage"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"direct_loss_weights": (1, 1, 1, 1, 1, 0)}, "direct_loss_weights"),
        ({"selection": {"minDelta": -1, "patience": 1}}, "min_delta"),
        ({"seed": 2**32}, "seed"),
    ),
)
def test_train_config_validates_programmatic_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**kwargs)


def test_device_auto_selects_cuda_only_when_available(monkeypatch):
    monkeypatch.setattr(
        "app.worker.runtime.device.torch.cuda.is_available",
        lambda: True,
    )
    assert get_device("auto").type == "cuda"

    monkeypatch.setattr(
        "app.worker.runtime.device.torch.cuda.is_available",
        lambda: False,
    )
    assert get_device("auto").type == "cpu"


def test_explicit_cuda_errors_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.worker.runtime.device.torch.cuda.is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match=r"CUDA .* not available"):
        get_device("cuda")


CHECKPOINT_MODEL_CONFIG = {
    "seq_len": 20,
    "hidden": 256,
    "layers": 5,
    "dropout": 0.1,
    "nhead": 8,
    "context_mode": "relaxed",
    "out_dim": 6,
    "feature_dim": 17,
}


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--seq-len", "30", "--seq-len=30 conflicts with checkpoint value 20"),
        ("--hidden", "128", "--hidden=128 conflicts with checkpoint value 256"),
        ("--layers", "4", "--layers=4 conflicts with checkpoint value 5"),
        ("--dropout", "0.2", "--dropout=0.2 conflicts with checkpoint value 0.1"),
        ("--nhead", "4", "--nhead=4 conflicts with checkpoint value 8"),
        ("--mode", "strict", "--mode=strict conflicts with checkpoint value relaxed"),
    ),
)
def test_prediction_model_overrides_must_match_checkpoint(option, value, message):
    args = parse("predict", "input.arrow", option, value)

    with pytest.raises(ValueError, match=message):
        model_config_from_args(args, checkpoint_config=CHECKPOINT_MODEL_CONFIG)


def test_matching_prediction_overrides_are_allowed_and_feature_dim_is_checkpoint_only():
    args = parse(
        "predict",
        "input.arrow",
        "--seq-len",
        "20",
        "--hidden",
        "256",
        "--layers",
        "5",
        "--dropout",
        "0.1",
        "--nhead",
        "8",
        "--mode",
        "relaxed",
    )
    args.feature_dim = 999

    config = model_config_from_args(args, checkpoint_config=CHECKPOINT_MODEL_CONFIG)

    assert config.feature_dim == 17
    assert config.seq_len == 20


def test_training_model_config_without_checkpoint_uses_defaults():
    args = SimpleNamespace(
        seq_len=12,
        hidden=None,
        layers=None,
        dropout=None,
        nhead=None,
        context_mode=None,
        out_dim=None,
    )

    config = model_config_from_args(args)

    assert config.seq_len == 12
    assert config.hidden == 256
    assert config.nhead == 8
    assert config.feature_dim is None
