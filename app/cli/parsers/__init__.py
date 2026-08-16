import argparse
from typing import Any, Protocol


class SubparserTarget(Protocol):
    def add_parser(
        self,
        name: str,
        **options: Any,
    ) -> argparse.ArgumentParser: ...


__all__ = ["SubparserTarget"]
