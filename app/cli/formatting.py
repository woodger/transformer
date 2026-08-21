import argparse

from app.version import __version__

COMMAND_GROUPS = (
    (
        "Flight",
        (("flight serve", "Run the durable Arrow Flight job service."),),
    ),
    (
        "Models",
        (
            ("models list", "List published model generations"),
            ("models delete <model-ref>", "Delete one model generation"),
        ),
    ),
    (
        "Training and inference",
        (
            ("fit", "Train from an Arrow file."),
            ("predict", "Predict from an Arrow file."),
            ("fit-stream", "Train from framed stdin."),
            ("predict-stream", "Predict from framed stdin."),
        ),
    ),
    (
        "Diagnostics",
        (("gmark", "Stress one CUDA GPU with synthetic training."),),
    ),
    (
        "Metrics",
        (("plot-metrics", "Render SVG charts from metrics JSONL."),),
    ),
    (
        "Database",
        (
            ("db migrations status", "Read-only schema migration state"),
            ("db migrations apply", "Apply pending schema migrations"),
            ("db migrations rollback", "Revert the latest schema migration"),
        ),
    ),
)
COMMAND_HELP = {
    name: description
    for _, commands in COMMAND_GROUPS
    for name, description in commands
}
COMMAND_EXAMPLES = {
    "fit": """Examples:
  transformer fit ./data/train.arrow --seq-len=20
""",
    "predict": """Examples:
  transformer predict ./data/test.arrow --checkpoint=model.pth --output=/tmp/preds.arrow
""",
    "gmark": """Examples:
  transformer gmark --duration=300 --use-amp
""",
}


def format_root_help() -> str:
    command_groups = "\n\n".join(
        f"{group}:\n"
        + "\n".join(
            f"  {name:<32}{description}" for name, description in commands
        )
        for group, commands in COMMAND_GROUPS
    )
    return (
        f"transformer {__version__}\n\n"
        "Usage:\n"
        "  transformer <command> [args] [options]\n"
        "  transformer <command> --help\n"
        "  transformer --help\n"
        "  transformer --version\n\n"
        "Global options:\n"
        "  --help, -h       Show help and exit\n"
        "  --version, -v    Show package version and exit\n\n"
        "Commands:\n\n"
        f"{command_groups}\n\n"
        "Command details:\n"
        "  transformer <command> --help\n"
    )


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action) or ""


class FlightServiceHelpFormatter(argparse.RawTextHelpFormatter):
    pass


__all__ = [
    "COMMAND_EXAMPLES",
    "COMMAND_HELP",
    "FlightServiceHelpFormatter",
    "HelpFormatter",
    "format_root_help",
]
