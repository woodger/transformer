import argparse
from collections.abc import Iterable
from typing import TypeVar, overload

from app.cli.formatting import HelpFormatter, format_root_help
from app.cli.parsers.admin import add_admin_parsers
from app.cli.parsers.service import add_service_parsers
from app.version import __version__

_NamespaceT = TypeVar("_NamespaceT")


class ArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(
        self,
        args: Iterable[str] | None,
        namespace: _NamespaceT,
    ) -> _NamespaceT: ...

    @overload
    def parse_args(
        self,
        *,
        namespace: _NamespaceT,
    ) -> _NamespaceT: ...

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: _NamespaceT | None = None,
    ) -> argparse.Namespace | _NamespaceT:
        parsed = super().parse_args(args, namespace)
        if parsed is None:
            raise AssertionError("argument parser returned no namespace")
        hidden = getattr(parsed, "hidden", None)
        nhead = getattr(parsed, "nhead", None)
        if hidden is not None and nhead is not None and hidden % nhead != 0:
            self.error(
                f"--hidden={hidden} must be divisible by --nhead={nhead}"
            )
        return parsed


class RootArgumentParser(ArgumentParser):
    def format_help(self) -> str:
        return format_root_help()


def build_parser() -> argparse.ArgumentParser:
    parser = RootArgumentParser(
        prog="transformer",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        parser_class=ArgumentParser,
        dest="action",
        required=True,
        title="commands",
        metavar="COMMAND",
    )
    add_service_parsers(subparsers)
    add_admin_parsers(subparsers)
    return parser


__all__ = ["ArgumentParser", "RootArgumentParser", "build_parser"]
