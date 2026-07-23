from datetime import datetime
import json
from pathlib import Path
import re
import uuid


SCHEMA_ROOT = (
    Path(__file__).parents[1]
    / "contracts"
    / "flight"
    / "v1"
    / "schemas"
)


def read_contract_schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text())


def _json_equal(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _matches_type(value, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise AssertionError(
        f"unsupported JSON Schema type in contract test: {expected}"
    )


def validate_schema_subset(value, schema: dict, path: str = "$") -> None:
    """Validate the contract's deliberately small Draft 2020-12 subset.

    This keeps contract verification dependency-free while exercising the
    required, closed-object and conditional semantics used by result schemas.
    It is not intended to be a general JSON Schema implementation.
    """

    assert "$ref" not in schema, (
        f"{path}: direct result schemas must be self-contained"
    )

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        assert any(
            _matches_type(value, expected)
            for expected in expected_types
        ), (
            f"{path}: expected type {expected_types}, "
            f"got {type(value).__name__}"
        )

    if "const" in schema:
        assert _json_equal(value, schema["const"]), (
            f"{path}: expected const {schema['const']!r}, got {value!r}"
        )
    if "enum" in schema:
        assert any(
            _json_equal(value, candidate)
            for candidate in schema["enum"]
        ), f"{path}: {value!r} is not in {schema['enum']!r}"

    if isinstance(value, dict):
        required = set(schema.get("required", ()))
        missing = required - value.keys()
        assert not missing, (
            f"{path}: missing required fields {sorted(missing)!r}"
        )

        properties = schema.get("properties", {})
        unknown = value.keys() - properties.keys()
        additional = schema.get("additionalProperties", True)
        if additional is False:
            assert not unknown, (
                f"{path}: unexpected fields {sorted(unknown)!r}"
            )
        elif isinstance(additional, dict):
            for key in unknown:
                validate_schema_subset(
                    value[key],
                    additional,
                    f"{path}.{key}",
                )

        for key, property_schema in properties.items():
            if key in value:
                validate_schema_subset(
                    value[key],
                    property_schema,
                    f"{path}.{key}",
                )

    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], (
                f"{path}: too few items"
            )
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], (
                f"{path}: too many items"
            )
        if schema.get("uniqueItems"):
            canonical = [
                json.dumps(item, sort_keys=True)
                for item in value
            ]
            assert len(canonical) == len(set(canonical)), (
                f"{path}: duplicate items"
            )

        prefix = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix[:len(value)]):
            validate_schema_subset(
                value[index],
                item_schema,
                f"{path}[{index}]",
            )
        items = schema.get("items")
        if items is False:
            assert len(value) <= len(prefix), (
                f"{path}: items beyond prefix are forbidden"
            )
        elif isinstance(items, dict):
            start = len(prefix) if prefix else 0
            for index, item in enumerate(value[start:], start=start):
                validate_schema_subset(
                    item,
                    items,
                    f"{path}[{index}]",
                )

    if isinstance(value, str):
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], (
                f"{path}: string is too short"
            )
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"], (
                f"{path}: string is too long"
            )
        if "pattern" in schema:
            assert re.search(schema["pattern"], value), (
                f"{path}: pattern mismatch"
            )
        if schema.get("format") == "uuid":
            assert str(uuid.UUID(value)) == value.lower(), (
                f"{path}: invalid UUID"
            )
        elif schema.get("format") == "date-time":
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
            assert parsed.tzinfo is not None, (
                f"{path}: date-time must include timezone"
            )

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        if "minimum" in schema:
            assert value >= schema["minimum"], (
                f"{path}: below minimum"
            )
        if "maximum" in schema:
            assert value <= schema["maximum"], (
                f"{path}: above maximum"
            )
        if "exclusiveMinimum" in schema:
            assert value > schema["exclusiveMinimum"], (
                f"{path}: below exclusiveMinimum"
            )
        if "exclusiveMaximum" in schema:
            assert value < schema["exclusiveMaximum"], (
                f"{path}: above exclusiveMaximum"
            )

    for branch in schema.get("allOf", ()):
        condition = branch.get("if")
        if condition is None:
            validate_schema_subset(value, branch, path)
            continue
        try:
            validate_schema_subset(value, condition, path)
        except AssertionError:
            selected = branch.get("else")
        else:
            selected = branch.get("then")
        if selected is not None:
            validate_schema_subset(value, selected, path)
