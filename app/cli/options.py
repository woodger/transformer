import argparse
import math
from typing import Any, Protocol


class ArgumentTarget(Protocol):
    def add_argument(
        self,
        *name_or_flags: str,
        **options: Any,
    ) -> argparse.Action: ...


def add_hidden_help_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help=argparse.SUPPRESS,
    )


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def seed(value: str) -> int:
    parsed = nonnegative_int(value)
    if parsed > 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a non-negative number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def dropout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be in the range [0, 1)"
        ) from exc
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in the range [0, 1)")
    return parsed


def six_positive_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected six comma-separated positive numbers"
        ) from exc
    if len(parsed) != 6 or any(
        not math.isfinite(item) or item <= 0
        for item in parsed
    ):
        raise argparse.ArgumentTypeError(
            "expected six comma-separated positive numbers"
        )
    return parsed


def gmark_memory_fraction(value: str) -> float:
    parsed = nonnegative_float(value)
    if parsed > 0.9:
        raise argparse.ArgumentTypeError("must be in the range [0, 0.9]")
    return parsed


__all__ = [
    "ArgumentTarget",
    "add_hidden_help_argument",
    "dropout",
    "gmark_memory_fraction",
    "nonnegative_float",
    "nonnegative_int",
    "positive_float",
    "positive_int",
    "seed",
    "six_positive_floats",
]
