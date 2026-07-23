import ast
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"

CORE_PACKAGES = (
    "app.data",
    "app.metrics",
    "app.model",
    "app.runtime",
    "app.storage",
    "app.training",
)
OUTER_PACKAGES = (
    "app.cli",
    "app.commands",
    "app.database",
    "app.flight",
)
LEDGER_SLICES = (
    "app.flight.ledger_inputs",
    "app.flight.ledger_execution",
    "app.flight.ledger_artifacts",
    "app.flight.ledger_maintenance",
)


def _module_name(path: Path) -> str:
    parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_within(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _application_modules() -> dict[str, Path]:
    return {
        _module_name(path): path
        for path in APP_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _application_imports(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            relative_name = "." * node.level + (node.module or "")
            imported = importlib.util.resolve_name(relative_name, package)
        else:
            imported = node.module
        if imported is None:
            continue

        imports.add(imported)
        imports.update(
            f"{imported}.{alias.name}"
            for alias in node.names
            if alias.name != "*"
        )

    return {name for name in imports if _is_within(name, "app")}


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visited = set()
    active = []
    active_set = set()

    def visit(module: str) -> list[str] | None:
        visited.add(module)
        active.append(module)
        active_set.add(module)
        for dependency in sorted(graph[module]):
            if dependency in active_set:
                start = active.index(dependency)
                return [*active[start:], dependency]
            if dependency not in visited:
                cycle = visit(dependency)
                if cycle is not None:
                    return cycle
        active.pop()
        active_set.remove(module)
        return None

    for module in sorted(graph):
        if module not in visited:
            cycle = visit(module)
            if cycle is not None:
                return cycle
    return None


def test_documented_package_dependencies_point_inward():
    modules = _application_modules()
    violations = []

    for source, path in modules.items():
        imports = _application_imports(path, source)
        if any(_is_within(source, package) for package in CORE_PACKAGES):
            forbidden = OUTER_PACKAGES
        elif _is_within(source, "app.database"):
            forbidden = ("app.flight",)
        else:
            continue

        for dependency in imports:
            if any(_is_within(dependency, package) for package in forbidden):
                violations.append(f"{source} -> {dependency}")

    assert violations == [], (
        "documented package dependency boundary was crossed:\n"
        + "\n".join(sorted(violations))
    )


def test_application_internal_import_graph_is_acyclic():
    modules = _application_modules()
    graph = {
        source: {
            dependency
            for dependency in _application_imports(path, source)
            if dependency in modules
        }
        for source, path in modules.items()
    }

    cycle = _find_cycle(graph)

    assert cycle is None, f"application import cycle: {' -> '.join(cycle or [])}"


def test_flight_control_plane_slices_follow_one_way_dependencies():
    modules = _application_modules()
    imports = {
        module: _application_imports(path, module)
        for module, path in modules.items()
    }

    assert set(LEDGER_SLICES) <= imports["app.flight.ledger"]
    assert "app.flight.job_actions" in imports["app.flight.coordinator"]

    forbidden_control_plane = (
        "app.flight.application",
        "app.flight.coordinator",
        "app.flight.job_actions",
        "app.flight.server",
        "app.flight.worker",
    )
    violations = []
    for source in LEDGER_SLICES:
        for dependency in imports[source]:
            if any(
                _is_within(dependency, package)
                for package in forbidden_control_plane
            ):
                violations.append(f"{source} -> {dependency}")
            if dependency in LEDGER_SLICES and dependency != source:
                violations.append(f"{source} -> {dependency}")

    for dependency in imports["app.flight.job_actions"]:
        if (
            _is_within(dependency, "app.database")
            or dependency
            in {
                "app.flight.application",
                "app.flight.coordinator",
                "app.flight.server",
                "app.flight.worker",
            }
        ):
            violations.append(
                f"app.flight.job_actions -> {dependency}"
            )

    assert violations == [], (
        "Flight control-plane slice dependency points outward:\n"
        + "\n".join(sorted(violations))
    )
