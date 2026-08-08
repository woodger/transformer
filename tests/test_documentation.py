import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "readme.md"


def test_readme_is_the_navigation_entrypoint():
    document = README.read_text(encoding="utf-8")

    assert re.findall(r"^## (.+)$", document, flags=re.MULTILINE) == [
        "Что есть в проекте",
        "Режимы работы",
        "Быстрый старт",
        "CLI команды",
        "Документация",
        "Структура проекта",
        "Развертывание",
    ]
    for path in (
        "docs/getting-started.md",
        "docs/cli/index.md",
        "docs/local-arrow-protocol.md",
        "docs/training-runtime.md",
        "docs/flight-operations.md",
        "docs/deployment/systemd.md",
    ):
        assert (PROJECT_ROOT / path).is_file()
