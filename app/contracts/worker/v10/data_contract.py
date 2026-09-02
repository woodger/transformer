from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast


@dataclass(frozen=True, slots=True)
class DataContractStructure:
    """Digest-free data-contract identity used for fit compatibility."""

    contract_id: str
    version: int
    profile: str
    seq_len: int
    feature_dim: int
    target_schema_id: str

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, object],
    ) -> Self:
        return cls(
            contract_id=cast(str, document.get("id")),
            version=cast(int, document.get("version")),
            profile=cast(str, document.get("profile")),
            seq_len=cast(
                int,
                document.get("seq_len", document.get("seqLen")),
            ),
            feature_dim=cast(
                int,
                document.get("feature_dim", document.get("featureDim")),
            ),
            target_schema_id=cast(
                str,
                document.get(
                    "target_schema_id",
                    document.get("targetSchemaId"),
                ),
            ),
        )


__all__ = ["DataContractStructure"]
