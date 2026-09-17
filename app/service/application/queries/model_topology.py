from __future__ import annotations

from typing import cast

from app.service.application.messages.model_topology import (
    GetModelTopologyQuery,
    ModelTopologyResult,
)
from app.service.application.ports.model_catalog import (
    CatalogMetadataVerifier,
    CatalogModelNotFound,
    ModelCatalogStore,
)
from app.service.application.services.model_topology import ModelTopologyBuilder
from app.service.domain.json_types import JsonObject


class GetModelTopology:
    def __init__(
        self,
        store: ModelCatalogStore,
        *,
        metadata_verifier: CatalogMetadataVerifier,
        builder: ModelTopologyBuilder,
    ) -> None:
        self._store = store
        self._metadata_verifier = metadata_verifier
        self._builder = builder

    def execute(self, query: GetModelTopologyQuery) -> ModelTopologyResult:
        entry = self._store.get_model(query.owner_subject, query.model_ref)
        if entry is None:
            raise CatalogModelNotFound(query.model_ref)
        self._metadata_verifier.verify(entry.model)
        topology = self._builder.build(entry.model)

        if self._store.get_model(query.owner_subject, query.model_ref) is None:
            raise CatalogModelNotFound(query.model_ref)
        return ModelTopologyResult(
            request_id=query.request_id,
            model_ref=entry.model.model_ref,
            model_definition_sha256=str(topology["modelDefinitionSha256"]),
            nodes=tuple(cast(list[JsonObject], topology["nodes"])),
            edges=tuple(cast(list[JsonObject], topology["edges"])),
        )


__all__ = ["GetModelTopology"]
