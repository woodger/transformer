from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1 import ModelContract
from app.contracts.semantic.v1.digests import jcs_sha256
from app.worker.application.artifacts import checkpoint_metadata
from app.worker.training.trainer import Trainer

_COPY_CHUNK_BYTES = 1024 * 1024


def load_model_contract(path: str) -> ModelContract:
    try:
        with open(path, encoding="utf-8") as source:
            loaded = cast(
                object,
                json.load(source, object_pairs_hook=_unique_object),
            )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("model contract could not be read") from exc
    document = loaded
    if isinstance(loaded, Mapping):
        wrapper = cast(Mapping[object, object], loaded)
        if "modelContract" in wrapper:
            document = wrapper["modelContract"]
    return ModelContract.from_document(document)


def checkpoint_model_contract(metadata: JsonObject) -> ModelContract:
    contract = ModelContract.from_document(metadata.get("modelContract"))
    data_contract = metadata.get("dataContract")
    semantic_digests = metadata.get("semanticDigests")
    if not isinstance(data_contract, Mapping) or not isinstance(
        semantic_digests,
        Mapping,
    ):
        raise ValueError("checkpoint semantic metadata is invalid")

    data_digest = data_contract.get("dataContractSha256")
    if not isinstance(data_digest, str) or (
        contract.digests(data_digest) != semantic_digests
    ):
        raise ValueError("checkpoint semantic digests are inconsistent")

    return contract


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_checkpoint_metadata(
    trainer: Trainer,
    model_contract: ModelContract,
    *,
    source_identity: str,
    manifest_sha256: str,
) -> JsonObject:
    model_config = model_contract.model_config
    data_contract: JsonObject = {
        "identity": "transformer.local-arrow",
        "revision": 1,
        "profile": "transformer.local",
        "dataContractSha256": manifest_sha256,
        "seqLen": model_config["seqLen"],
        "featureDim": model_config["featureDim"],
    }
    semantic_digests = model_contract.digests(manifest_sha256)
    job_config_sha256 = jcs_sha256({
        "kind": "transformer.local-fit",
        "source": source_identity,
        "modelContractSha256": semantic_digests["modelContractSha256"],
        "training": trainer.train_config.to_manifest(),
    })
    trainer.initialization = {"kind": "random"}
    manifest: JsonObject = {
        "jobId": str(uuid.uuid4()),
        "dataContract": data_contract,
        "modelContract": model_contract.to_document(),
        "semanticDigests": semantic_digests,
        "jobConfigSha256": job_config_sha256,
        "manifestSha256": manifest_sha256,
    }
    return checkpoint_metadata(trainer, manifest)


def stream_manifest_sha256(
    model_contract: ModelContract,
    *,
    received_frames: int,
    trained_frames: int,
) -> str:
    return jcs_sha256({
        "kind": "transformer.local-arrow-stream",
        "modelContract": model_contract.to_document(),
        "receivedFrames": received_frames,
        "trainedFrames": trained_frames,
    })


def normalized_source_identity(path: str) -> str:
    return os.path.realpath(os.fspath(Path(path)))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


__all__ = [
    "checkpoint_model_contract",
    "file_sha256",
    "load_model_contract",
    "local_checkpoint_metadata",
    "normalized_source_identity",
    "stream_manifest_sha256",
]
