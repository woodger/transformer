import json
import uuid
from datetime import datetime
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

SCHEMA_ROOT = (
    Path(__file__).parents[1]
    / "contracts"
    / "flight"
    / "v1"
    / "schemas"
)

_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("uuid", raises=(AttributeError, ValueError))
def _is_uuid(value) -> bool:
    if not isinstance(value, str):
        return True
    return str(uuid.UUID(value)) == value.lower()


@_FORMAT_CHECKER.checks("date-time", raises=(AttributeError, ValueError))
def _is_timezone_aware_datetime(value) -> bool:
    if not isinstance(value, str):
        return True
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None


def read_contract_schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


@cache
def _contract_registry() -> Registry:
    resources = []
    for path in sorted(SCHEMA_ROOT.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def check_contract_schema(schema: dict) -> None:
    Draft202012Validator.check_schema(schema)


def validate_contract_document(value, schema: dict) -> None:
    Draft202012Validator(
        schema,
        registry=_contract_registry(),
        format_checker=_FORMAT_CHECKER,
    ).validate(value)
