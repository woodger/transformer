from app.contracts.flight.v11.codec import validate_request_document
from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1.digests import jcs_sha256


def job_config_sha256(document: JsonObject) -> str:
    validated = validate_request_document(document, "job-config")
    return jcs_sha256(validated)


__all__ = ["job_config_sha256"]
