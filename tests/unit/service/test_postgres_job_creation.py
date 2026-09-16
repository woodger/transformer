from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.semantic.v3 import ModelContract
from app.contracts.worker.v14.config import ModelConfig, TrainConfig
from app.contracts.worker.v14.model_definition import resolved_semantic_digests
from app.service.adapters.outbound.postgres.config import DatabaseConfig
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.session import Database
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from tests.fixture_documents import semantic_fixture_document

_JOB_ID = "00000000-0000-4000-8000-000000000001"
_EXECUTION_ID = "00000000-0000-4000-8000-000000000002"
_MODEL_REF = "mdl_00000000000000000000000000000001"
_MODEL_CONTRACT = ModelContract.from_document(
    semantic_fixture_document("single-regression")["modelContract"],
)
_SEMANTIC_DIGESTS = resolved_semantic_digests(
    _MODEL_CONTRACT,
    "a" * 64,
    ModelConfig.from_tuning(
        _MODEL_CONTRACT.model_tuning,
        seq_len=2,
        feature_dim=1,
    ),
)


def _database():
    return Database(
        DatabaseConfig(
            host="localhost",
            database="transformer",
            user="transformer",
            password="secret",
        ),
        engine=create_engine("sqlite+pysqlite:///:memory:"),
    )


def _create_job(
    ledger,
    connection,
    *,
    operation="predict",
    initialization=None,
    resolved_model_ref=None,
):
    return ledger.create_job(
        job_id=_JOB_ID,
        owner_subject="test-owner",
        client_execution_id=_EXECUTION_ID,
        operation=operation,
        requested_device="cpu",
        prediction_column="prediction",
        config_hash="b" * 64,
        source_encoding={
            "featureBlocks": [
                {"windowRows": 1, "nativeRowWidth": 1},
            ],
        },
        data_contract={
            "dataContractSha256": "a" * 64,
            "seqLen": 2,
            "featureDim": 1,
        },
        model_contract=_MODEL_CONTRACT.to_document(),
        semantic_digests=_SEMANTIC_DIGESTS,
        create_result={"jobId": _JOB_ID},
        model_label="daily" if operation == "fit" else None,
        resolved_model_ref=(
            resolved_model_ref
            if operation == "fit"
            else _MODEL_REF
        ),
        model_config=ModelConfig.from_tuning(
            _MODEL_CONTRACT.model_tuning,
            seq_len=2,
            feature_dim=1,
        ),
        training_config=TrainConfig() if operation == "fit" else None,
        initialization=initialization,
        now=1.0,
        connection=connection,
    )


def _session_that_raises(constraint_name, sqlstate):
    original = SimpleNamespace(
        diag=SimpleNamespace(constraint_name=constraint_name),
        sqlstate=sqlstate,
    )
    error = IntegrityError("INSERT", {}, original)
    return cast(
        Session,
        SimpleNamespace(
            add=lambda _record: None,
            flush=lambda: (_ for _ in ()).throw(error),
        ),
    )


def test_predict_job_keeps_absent_initialization_as_none():
    records = []
    session = cast(
        Session,
        SimpleNamespace(add=records.append, flush=lambda: None),
    )
    database = _database()
    try:
        stored = _create_job(Ledger(database), session)
    finally:
        database.close()

    assert stored["initialization"] is None


def test_fit_job_keeps_initialization_document():
    records = []
    session = cast(
        Session,
        SimpleNamespace(add=records.append, flush=lambda: None),
    )
    database = _database()
    try:
        stored = _create_job(
            Ledger(database),
            session,
            operation="fit",
            initialization={"source": "random"},
        )
    finally:
        database.close()

    assert stored["initialization"] == {"source": "random"}


def test_published_model_fit_keeps_resolved_parent_and_complete_lineage():
    records = []
    session = cast(
        Session,
        SimpleNamespace(add=records.append, flush=lambda: None),
    )
    initialization = {
        "source": "publishedModel",
        "parentModelRef": _MODEL_REF,
        "parentCheckpointSha256": "c" * 64,
        "parentDataContractSha256": "d" * 64,
        "dataContractSha256": "a" * 64,
        "parentTargetContractSha256": _SEMANTIC_DIGESTS[
            "targetContractSha256"
        ],
        "targetContractSha256": _SEMANTIC_DIGESTS[
            "targetContractSha256"
        ],
        "parentObjectiveSha256": _SEMANTIC_DIGESTS["objectiveSha256"],
        "objectiveSha256": _SEMANTIC_DIGESTS["objectiveSha256"],
        "parentModelDefinitionSha256": _SEMANTIC_DIGESTS[
            "modelDefinitionSha256"
        ],
        "modelDefinitionSha256": _SEMANTIC_DIGESTS[
            "modelDefinitionSha256"
        ],
    }
    database = _database()
    try:
        stored = _create_job(
            Ledger(database),
            session,
            operation="fit",
            initialization=initialization,
            resolved_model_ref=_MODEL_REF,
        )
    finally:
        database.close()

    assert stored["resolved_model_ref"] == _MODEL_REF
    assert stored["initialization"] == initialization


@pytest.mark.parametrize(
    "constraint_name",
    ("job_identities_pkey", "jobs_pkey"),
)
def test_job_identity_constraints_are_reported_as_duplicate(constraint_name):
    database = _database()
    try:
        with pytest.raises(ServiceError) as captured:
            _create_job(
                Ledger(database),
                _session_that_raises(constraint_name, "23505"),
            )
    finally:
        database.close()

    assert captured.value.code is ErrorCode.ALREADY_EXISTS
    assert captured.value.message == f"job already exists: {_JOB_ID}"


def test_unexpected_job_constraint_is_reported_as_internal_error():
    events = []
    logger = SimpleNamespace(
        event=lambda event, **fields: events.append((event, fields)),
    )
    database = _database()
    try:
        with pytest.raises(
            RuntimeError,
            match="job creation violated persistence invariants",
        ):
            _create_job(
                Ledger(database, logger=logger),
                _session_that_raises("jobs_operation_fields_ck", "23514"),
            )
    finally:
        database.close()

    assert events == [(
        "postgres.job.create_integrity_error",
        {
            "jobId": _JOB_ID,
            "constraintName": "jobs_operation_fields_ck",
            "sqlstate": "23514",
        },
    )]
