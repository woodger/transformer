from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.service.adapters.outbound.opensearch.client import (
    OpenSearchMetricsClient,
)
from app.service.adapters.outbound.opensearch.config import (
    OpenSearchMetricsConfig,
    load_opensearch_metrics_config,
)
from app.service.application.ports.metrics import (
    BlockedMetricsDeliveryError,
    RetryableMetricsDeliveryError,
)


def _document(identifier: str, digest: str) -> dict[str, object]:
    return {
        "eventId": identifier,
        "documentSha256": digest,
        "schema": "inventory.metrics.point.v1",
    }


def _client(responses):
    client = object.__new__(OpenSearchMetricsClient)
    client.config = SimpleNamespace(
        max_bulk_documents=500,
        max_bulk_bytes=2 * 1024 * 1024,
    )
    requests = []
    iterator = iter(responses)

    def request(method, path, body, *, ndjson=False):
        requests.append((method, path, body, ndjson))
        return next(iterator)

    client._request = request
    return client, requests


def test_configuration_is_disabled_only_when_no_opensearch_key_is_present(
    tmp_path,
):
    env_file = tmp_path / "absent.env"
    assert load_opensearch_metrics_config(
        environ={},
        env_file=env_file,
    ) is None

    with pytest.raises(ValueError, match="incomplete OpenSearch"):
        load_opensearch_metrics_config(
            environ={"OPENSEARCH_ENDPOINT": "https://search.example:9200"},
            env_file=env_file,
        )


def test_configuration_requires_https_ca_and_a_dedicated_non_admin_writer(
    tmp_path,
):
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("test CA fixture", encoding="utf-8")
    values = {
        "OPENSEARCH_ENDPOINT": "https://hp260g9.home:9200",
        "OPENSEARCH_USERNAME": "transformer-metrics",
        "OPENSEARCH_PASSWORD": "secret",
        "OPENSEARCH_CA_FILE": str(ca_file),
        "OPENSEARCH_DEPLOYMENT_ID": "hp800g9.home",
    }

    config = load_opensearch_metrics_config(
        environ=values,
        env_file=tmp_path / "absent.env",
    )
    assert config is not None
    assert config.endpoint == "https://hp260g9.home:9200"
    assert "secret" not in repr(config)

    with pytest.raises(ValueError, match="must not use admin"):
        OpenSearchMetricsConfig(
            endpoint=config.endpoint,
            username="admin",
            password="secret",
            ca_file=config.ca_file,
            deployment_id=config.deployment_id,
        )
    with pytest.raises(ValueError, match="HTTPS origin"):
        OpenSearchMetricsConfig(
            endpoint="http://hp260g9.home:9200",
            username=config.username,
            password="secret",
            ca_file=config.ca_file,
            deployment_id=config.deployment_id,
        )


def test_bulk_uses_create_and_accepts_only_identical_conflicts():
    identifier = "a" * 64
    digest = "b" * 64
    client, requests = _client([
        {
            "items": [
                {"create": {"status": 409}},
            ],
        },
        {
            "docs": [
                {
                    "_id": identifier,
                    "found": True,
                    "_source": {"documentSha256": digest},
                }
            ]
        },
    ])

    client.create_documents(
        "metrics-points-v1",
        (_document(identifier, digest),),
        id_field="eventId",
    )

    bulk_lines = requests[0][2].decode("utf-8").splitlines()
    assert json.loads(bulk_lines[0]) == {
        "create": {
            "_id": identifier,
            "_index": "metrics-points-v1",
        }
    }
    assert requests[1][1] == "/metrics-points-v1/_mget"


def test_conflicting_document_with_different_digest_blocks_delivery():
    identifier = "a" * 64
    client, _ = _client([
        {"items": [{"create": {"status": 409}}]},
        {
            "docs": [
                {
                    "_id": identifier,
                    "found": True,
                    "_source": {"documentSha256": "c" * 64},
                }
            ]
        },
    ])

    with pytest.raises(BlockedMetricsDeliveryError, match="integrity"):
        client.create_documents(
            "metrics-points-v1",
            (_document(identifier, "b" * 64),),
            id_field="eventId",
        )


def test_retryable_bulk_item_does_not_report_the_chunk_as_delivered():
    client, _ = _client([
        {"items": [{"create": {"status": 429}}]},
    ])

    with pytest.raises(RetryableMetricsDeliveryError, match="status 429"):
        client.create_documents(
            "metrics-points-v1",
            (_document("a" * 64, "b" * 64),),
            id_field="eventId",
        )
