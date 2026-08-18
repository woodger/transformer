import ast
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

from app.project import PROJECT_ROOT

APP_ROOT = PROJECT_ROOT / "app"


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


def _imports(path: Path, module: str) -> set[str]:
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
            relative = "." * node.level + (node.module or "")
            imported = importlib.util.resolve_name(relative, package)
        else:
            imported = node.module
        if imported is not None:
            imports.add(imported)
    return imports


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visited = set()
    active: list[str] = []
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


def test_service_domain_and_application_dependencies_point_inward():
    modules = _application_modules()
    violations = []
    allowed_domain = (
        "app.service.domain",
        "app.contracts.worker",
    )
    allowed_application = (
        "app.service.application",
        "app.service.domain",
        "app.contracts.metrics",
        "app.contracts.worker",
    )
    forbidden_libraries = ("pyarrow", "sqlalchemy", "torch")
    for source, path in modules.items():
        if _is_within(source, "app.service.domain"):
            allowed = allowed_domain
        elif _is_within(source, "app.service.application"):
            allowed = allowed_application
        else:
            continue
        for dependency in _imports(path, source):
            if dependency.startswith("app.") and not any(
                _is_within(dependency, package) for package in allowed
            ):
                violations.append(f"{source} -> {dependency}")
            if any(_is_within(dependency, package) for package in forbidden_libraries):
                violations.append(f"{source} -> {dependency}")
    assert violations == [], "outward service dependency:\n" + "\n".join(
        sorted(violations)
    )


def test_pure_control_plane_has_no_runtime_side_effect_dependencies():
    modules = _application_modules()
    packages = (
        "app.service.domain",
        "app.service.application.commands",
        "app.service.application.queries",
        "app.service.application.ports",
    )
    forbidden = ("os", "pathlib", "subprocess", "pyarrow", "sqlalchemy", "torch")
    violations = []
    for source, path in modules.items():
        if not any(_is_within(source, package) for package in packages):
            continue
        for dependency in _imports(path, source):
            if any(_is_within(dependency, package) for package in forbidden):
                violations.append(f"{source} -> {dependency}")
    assert violations == []


def test_service_adapters_and_processes_do_not_cross_ownership_boundaries():
    modules = _application_modules()
    violations = []
    for source, path in modules.items():
        dependencies = _imports(path, source)
        if _is_within(source, "app.service.adapters.inbound"):
            forbidden = (
                "app.service.adapters.outbound",
                "app.service.bootstrap",
            )
        elif _is_within(source, "app.service.adapters.outbound"):
            forbidden = (
                "app.service.adapters.inbound",
                "app.service.bootstrap",
                "app.worker",
            )
        elif _is_within(source, "app.worker"):
            forbidden = (
                "app.service",
                "sqlalchemy",
            )
        elif _is_within(source, "app.admin.cli"):
            forbidden = (
                "app.admin.bootstrap",
                "app.service.adapters",
                "app.worker",
                "pyarrow",
                "torch",
            )
        elif _is_within(source, "app.admin"):
            forbidden = (
                "app.worker",
                "pyarrow",
                "torch",
            )
        else:
            continue
        for dependency in dependencies:
            if any(_is_within(dependency, package) for package in forbidden):
                violations.append(f"{source} -> {dependency}")
    assert violations == [], "ownership boundary crossed:\n" + "\n".join(
        sorted(violations)
    )


def test_application_internal_import_graph_is_acyclic():
    modules = _application_modules()
    graph = {
        source: {
            dependency
            for dependency in _imports(path, source)
            if dependency in modules
        }
        for source, path in modules.items()
    }
    cycle = _find_cycle(graph)
    assert cycle is None, f"application import cycle: {' -> '.join(cycle or [])}"


def test_canonical_contracts_and_composition_roots_exist():
    assert (APP_ROOT / "contracts" / "flight" / "v4").is_dir()
    assert (APP_ROOT / "contracts" / "worker" / "v6").is_dir()
    assert (APP_ROOT / "contracts" / "metrics" / "v2").is_dir()
    assert (APP_ROOT / "contracts" / "metrics" / "fit_run" / "v2").is_dir()
    assert (APP_ROOT / "local" / "fit.py").is_file()
    assert (APP_ROOT / "cli" / "parser.py").is_file()
    assert (APP_ROOT / "cli" / "formatting.py").is_file()
    assert (APP_ROOT / "cli" / "parsers" / "service.py").is_file()
    for path in (
        APP_ROOT / "service" / "adapters" / "outbound" / "artifacts",
        APP_ROOT / "service" / "adapters" / "outbound" / "cuda",
        APP_ROOT / "service" / "adapters" / "outbound" / "worker",
        APP_ROOT / "service" / "adapters" / "outbound" / "postgres" / "ledger",
        APP_ROOT / "service" / "adapters" / "outbound" / "postgres" / "telemetry",
        APP_ROOT / "service" / "adapters" / "outbound" / "artifacts" / "telemetry",
        APP_ROOT / "service" / "application" / "messages",
        APP_ROOT / "service" / "application" / "telemetry",
        APP_ROOT / "worker" / "checkpoints",
        APP_ROOT / "worker" / "telemetry",
    ):
        assert (path / "__init__.py").is_file()
    for path in (
        APP_ROOT / "service" / "bootstrap" / "application.py",
        APP_ROOT / "service" / "bootstrap" / "data_plane.py",
        APP_ROOT / "service" / "bootstrap" / "control_plane.py",
        APP_ROOT / "worker" / "bootstrap" / "__main__.py",
        APP_ROOT / "admin" / "bootstrap" / "auth_tokens.py",
        APP_ROOT / "admin" / "bootstrap" / "db_migrations.py",
    ):
        assert path.is_file()


def test_telemetry_does_not_own_core_training_or_service_state():
    protected_modules = (
        "app.worker.training.epoch",
        "app.service.domain.records",
        "app.service.adapters.outbound.artifacts.publication",
    )
    modules = _application_modules()
    violations = []
    for module in protected_modules:
        for dependency in _imports(modules[module], module):
            if _is_within(dependency, "app.worker.telemetry") or _is_within(
                dependency,
                "app.service.application.telemetry",
            ):
                violations.append(f"{module} -> {dependency}")

    for module, path in modules.items():
        if not _is_within(
            module,
            "app.service.adapters.outbound.postgres.ledger",
        ):
            continue
        for dependency in _imports(path, module):
            if _is_within(
                dependency,
                "app.service.application.telemetry",
            ):
                violations.append(f"{module} -> {dependency}")

    assert violations == [], "telemetry owns core state:\n" + "\n".join(
        sorted(violations)
    )


def test_service_application_job_api_is_transport_neutral():
    forbidden_symbols = {
        "ACQUIRE_ACTION",
        "CANCEL_ACTION",
        "CONTRACT_PATH_VERSION",
        "CREATE_ACTION",
        "FIT_SCHEMA_ID",
        "INPUT_CLOSE_ACTION",
        "PREDICT_SCHEMA_ID",
        "JobActionContract",
        "encode_document",
        "response_factory",
        "response_document",
    }
    forbidden_wire_values = {"descriptorPath", "schemaId"}
    violations = []
    for relative in (
        "application/commands/jobs.py",
        "application/queries/status.py",
        "application/services/input_upload.py",
        "application/queries/outputs.py",
    ):
        path = APP_ROOT / "service" / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        used.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        for symbol in sorted(used & forbidden_symbols):
            violations.append(f"{relative}: {symbol}")
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        for value in sorted(strings & forbidden_wire_values):
            violations.append(f"{relative}: {value}")
        for value in sorted(
            item for item in strings if item.startswith("transformer.v4.")
        ):
            violations.append(f"{relative}: {value}")
    assert violations == [], "Flight presentation leaked into application:\n" + (
        "\n".join(violations)
    )


def test_flight_handlers_receive_use_cases_instead_of_raw_ledger():
    from app.service.adapters.inbound.flight.coordinator import JobCoordinator
    from app.service.adapters.inbound.flight.output import OutputHandler
    from app.service.adapters.inbound.flight.upload import UploadHandler

    signatures = {
        "JobCoordinator": JobCoordinator.__init__,
        "OutputHandler": OutputHandler.__init__,
        "UploadHandler": UploadHandler.__init__,
    }
    forbidden_parameters = {"ledger", "spool", "recovery_store"}
    violations = []
    for name, constructor in signatures.items():
        parameters = set(inspect.signature(constructor).parameters)
        for parameter in sorted(parameters & forbidden_parameters):
            violations.append(f"{name}.{parameter}")
    assert violations == []


def test_service_and_admin_imports_do_not_initialize_worker_runtime():
    service_program = """
import json
import sys
import app.service.bootstrap.application
print(json.dumps({
    'torch': 'torch' in sys.modules,
    'worker': any(name.startswith('app.worker') for name in sys.modules),
}))
"""
    service = subprocess.run(
        [sys.executable, "-c", service_program],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert service.stdout.strip() == '{"torch": false, "worker": false}'

    admin_program = """
import json
import sys
import app.admin.bootstrap.db_migrations as admin_module

def run(_args):
    print(json.dumps({
        'torch': 'torch' in sys.modules,
        'worker': any(name.startswith('app.worker') for name in sys.modules),
        'flight': 'pyarrow.flight' in sys.modules,
    }))

admin_module.run = run
sys.argv = ['transformer', 'db', 'migrations', 'status']
from app.main import main
main()
"""
    admin = subprocess.run(
        [sys.executable, "-c", admin_program],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert admin.stdout.strip() == (
        '{"torch": false, "worker": false, "flight": false}'
    )


def test_control_plane_cli_parser_does_not_initialize_ml_runtime():
    program = """
import json
import sys
from app.cli.parser import build_parser

parser = build_parser()
parser.parse_args(["flight", "serve", "--allow-plaintext"])
parser.parse_args(["db", "migrations", "status"])
print(json.dumps({
    "mlModules": sorted(
        name for name in sys.modules
        if name in ("torch", "cuda")
        or name.startswith(("torch.", "cuda."))
    ),
    "worker": any(name.startswith("app.worker") for name in sys.modules),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.stdout.strip() == '{"mlModules": [], "worker": false}'
