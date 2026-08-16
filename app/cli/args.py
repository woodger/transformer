import argparse
from collections.abc import Sequence

from app.cli.parser import build_parser


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(args)
