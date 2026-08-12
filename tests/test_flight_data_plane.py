from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pytest
from sqlalchemy import func, select

from app.database.models import InputUpload
from app.flight.config import FlightServiceConfig
from app.flight.constants import (
    CONTRACT_NAME,
    FIT_SCHEMA_ID,
    ErrorCode,
    ExecutionState,
    InputState,
)
from app.flight.coordinator import JobCoordinator
from app.flight.errors import ServiceError
from app.flight.output import OutputHandler
from app.flight.recovery_store import RecoveryStore
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool
from app.flight.upload import UploadHandler
from app.worker.data.arrow import read_committed_fit_arrow
from tests.flight_v4_helpers import (
    DATA_CONTRACT_SHA256,
    OWNER,
    close_input,
    commit_input,
    create_fit,
    create_predict,
)


def _auth(token="secret"):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode())],
        timeout=5.0,
    )


def _fit_schema():
    return pa.schema([
        pa.field("src", pa.list_(pa.float32(), 4), nullable=False),
        pa.field("tgt", pa.list_(pa.float32(), 6), nullable=False),
    ])


def _fit_batch(rows, *, offset=0.0):
    schema = _fit_schema()
    source = [
        [offset + row + value for value in range(4)]
        for row in range(rows)
    ]
    target = [[0.0, 0.1, 0.0, 0.0, 0.2, 1.0] for _ in range(rows)]
    return pa.RecordBatch.from_arrays(
        [
            pa.array(source, type=schema.field("src").type),
            pa.array(target, type=schema.field("tgt").type),
        ],
        schema=schema,
    )


def _noncanonical_fit_batch(rows):
    item = pa.field("item", pa.float32(), nullable=False)
    schema = pa.schema([
        pa.field("src", pa.list_(item, 4), nullable=False),
        pa.field("tgt", pa.list_(item, 6), nullable=False),
    ])
    return pa.RecordBatch.from_arrays(
        [
            pa.array(
                [[float(value) for value in range(4)]] * rows,
                type=schema.field("src").type,
            ),
            pa.array([[0.0] * 6] * rows, type=schema.field("tgt").type),
        ],
        schema=schema,
    )


def _upload_metadata(job, payload_id, ordinal, rows):
    return json.dumps({
        "contract": CONTRACT_NAME,
        "version": 4,
        "jobId": job["job_id"],
        "clientExecutionId": job["client_execution_id"],
        "fencingToken": str(job["fencing_token"]),
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": FIT_SCHEMA_ID,
        "dataContractSha256": DATA_CONTRACT_SHA256,
        "rows": rows,
    }, separators=(",", ":")).encode()


@pytest.fixture
def data_plane(tmp_path, postgres_ledger):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        port=0,
        allow_plaintext=True,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    recovery = RecoveryStore(tmp_path / "recovery").initialize()
    coordinator = JobCoordinator(
        config,
        postgres_ledger,
        spool,
        recovery_store=recovery,
        cuda_available=lambda: False,
    )
    upload = UploadHandler(
        config,
        postgres_ledger,
        spool,
        recovery,
        cuda_available=lambda: False,
    )
    output = OutputHandler(config, postgres_ledger, spool)
    server = TransformerFlightServer(
        config,
        coordinator,
        {"secret": OWNER, "other": "another-owner"},
        upload_handler=upload,
        output_handler=output,
    )
    client = flight.FlightClient(("localhost", server.port))
    try:
        yield config, spool, recovery, postgres_ledger, upload, client
    finally:
        client.close()
        server.shutdown()


def _put(client, job, payload_id, ordinal, batches):
    descriptor = flight.FlightDescriptor.for_path(
        "transformer",
        "v4",
        "jobs",
        job["job_id"],
        "inputs",
        str(ordinal),
    )
    writer, results = client.do_put(
        descriptor,
        _fit_schema(),
        options=_auth(),
    )
    rows = sum(batch.num_rows for batch in batches)
    writer.write_metadata(pa.py_buffer(
        _upload_metadata(job, payload_id, ordinal, rows)
    ))
    for batch in batches:
        writer.write_batch(batch)
    writer.done_writing()
    result = results.read()
    assert results.read() is None
    writer.close()
    return json.loads(result.to_pybytes())


def _active_upload_count(ledger):
    with ledger.connection() as connection:
        return connection.scalar(
            select(func.count()).select_from(InputUpload)
        )


def test_one_doput_commits_one_immutable_multi_batch_payload(data_plane):
    _, _, recovery, ledger, _, client = data_plane
    job = create_fit(ledger)
    payload_id = str(uuid.uuid4())

    result = _put(
        client,
        job,
        payload_id,
        0,
        [_fit_batch(2), _fit_batch(1, offset=10)],
    )

    assert result["status"] == "committed"
    assert result["rows"] == 3
    assert result["batches"] == 2
    assert result["inputRevision"] == 1
    assert result["nextInputOrdinal"] == 1
    assert result["queued"] is True
    record = ledger.list_inputs(job["job_id"])[0]
    assert record["payload_id"] == payload_id
    assert record["storage_class"] == "recovery"
    path = recovery.absolute_path(record["relative_path"])
    assert payload_id in os.path.basename(path)
    with pa.memory_map(path, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        assert reader.num_record_batches == 2
    batch = read_committed_fit_arrow(
        path,
        expected_rows=3,
        source_width=4,
    )
    assert batch.features.shape == (3, 4)
    assert batch.targets.shape == (3, 6)


def test_noncanonical_schema_is_rejected_before_commit_or_queue(data_plane):
    config, spool, recovery, ledger, _, _ = data_plane
    job = create_fit(ledger)
    queued = []
    upload = UploadHandler(
        config,
        ledger,
        spool,
        recovery,
        cuda_available=lambda: False,
        queue_notifier=queued.append,
    )

    with pytest.raises(ServiceError) as error:
        upload.handle(
            OWNER,
            _descriptor(job),
            _Reader(
                job,
                str(uuid.uuid4()),
                _noncanonical_fit_batch(1),
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected result")),
        )

    assert error.value.code is ErrorCode.INVALID_ARGUMENT
    assert "canonical physical schema" in error.value.message
    assert ledger.list_inputs(job["job_id"]) == []
    assert _active_upload_count(ledger) == 0
    assert queued == []
    current = ledger.get_job(job["job_id"], owner_subject=OWNER)
    assert current["input_state"] == InputState.OPEN.value
    assert current["execution_state"] == ExecutionState.WAITING_INPUT.value
    directory = Path(recovery.input_directory(job["job_id"]))
    assert not directory.exists()


def test_noncanonical_schema_returns_primary_flight_error(data_plane):
    _, _, _, ledger, _, client = data_plane
    job = create_fit(ledger)
    batch = _noncanonical_fit_batch(1)
    descriptor = _descriptor(job)
    writer, results = client.do_put(
        descriptor,
        batch.schema,
        options=_auth(),
    )
    writer.write_metadata(pa.py_buffer(
        _upload_metadata(job, str(uuid.uuid4()), 0, 1)
    ))
    writer.write_batch(batch)

    writer.done_writing()
    assert results.read() is None
    with pytest.raises(
        pa.ArrowInvalid,
        match="INVALID_ARGUMENT: fit input differs from the canonical",
    ):
        writer.close()

    assert ledger.list_inputs(job["job_id"]) == []
    current = ledger.get_job(job["job_id"], owner_subject=OWNER)
    assert current["input_state"] == InputState.OPEN.value
    assert current["execution_state"] == ExecutionState.WAITING_INPUT.value

    retried = _put(
        client,
        job,
        str(uuid.uuid4()),
        0,
        [_fit_batch(1)],
    )
    assert retried["status"] == "committed"
    assert retried["queued"] is True


def test_out_of_order_completion_does_not_advance_worker_over_gap(data_plane):
    _, _, _, ledger, _, client = data_plane
    job = create_fit(ledger)

    later = _put(client, job, str(uuid.uuid4()), 1, [_fit_batch(1)])
    first = _put(client, job, str(uuid.uuid4()), 0, [_fit_batch(1)])

    assert later["nextInputOrdinal"] == 0
    assert later["queued"] is False
    assert first["nextInputOrdinal"] == 2
    assert first["queued"] is True
    assert [item.ordinal for item in ledger.list_committed_inputs(
        job["job_id"]
    )] == [0, 1]


def test_exact_duplicate_is_idempotent_but_changed_payload_conflicts(
    data_plane,
):
    _, _, _, ledger, _, client = data_plane
    job = create_fit(ledger)
    payload_id = str(uuid.uuid4())
    batch = _fit_batch(1)

    first = _put(client, job, payload_id, 0, [batch])
    repeated = _put(client, job, payload_id, 0, [batch])

    assert repeated == {**first, "queued": False}
    with pytest.raises(Exception, match="ALREADY_EXISTS"):
        _put(client, job, payload_id, 0, [_fit_batch(1, offset=99)])
    assert len(ledger.list_inputs(job["job_id"])) == 1


class _Reader:
    def __init__(self, job, payload_id, batch, *, before_eof=None):
        self.schema = batch.schema
        self._chunks = iter([
            SimpleNamespace(
                data=None,
                app_metadata=pa.py_buffer(_upload_metadata(
                    job,
                    payload_id,
                    0,
                    batch.num_rows,
                )),
            ),
            SimpleNamespace(data=batch, app_metadata=None),
        ])
        self._before_eof = before_eof
        self._eof = False

    def read_chunk(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if not self._eof and self._before_eof is not None:
                self._eof = True
                self._before_eof()
            raise


def _descriptor(job):
    return flight.FlightDescriptor.for_path(
        "transformer", "v4", "jobs", job["job_id"], "inputs", "0"
    )


def test_partial_upload_leaves_neither_receipt_nor_candidate(data_plane):
    _, _, recovery, ledger, upload, _ = data_plane
    job = create_fit(ledger)
    reader = _Reader(job, str(uuid.uuid4()), _fit_batch(1))
    original = reader.read_chunk
    reads = 0

    def disconnect():
        nonlocal reads
        reads += 1
        if reads == 3:
            raise OSError("client disconnected")
        return original()

    reader.read_chunk = disconnect

    with pytest.raises(OSError, match="disconnected"):
        upload.handle(
            OWNER,
            _descriptor(job),
            reader,
            SimpleNamespace(write=lambda _: pytest.fail("unexpected result")),
        )

    assert ledger.list_inputs(job["job_id"]) == []
    assert _active_upload_count(ledger) == 0
    directory = Path(recovery.input_directory(job["job_id"]))
    assert not directory.exists() or list(directory.iterdir()) == []


def test_takeover_rejects_doput_that_finishes_after_fence_advance(data_plane):
    _, _, recovery, ledger, upload, _ = data_plane
    job = create_fit(ledger)

    def takeover():
        ledger.acquire_job(
            job["job_id"],
            owner_subject=OWNER,
            previous_client_execution_id=job["client_execution_id"],
            expected_fencing_token=1,
            client_execution_id=str(uuid.uuid4()),
            acquire_grace_seconds=30,
        )

    reader = _Reader(
        job,
        str(uuid.uuid4()),
        _fit_batch(1),
        before_eof=takeover,
    )
    with pytest.raises(ServiceError) as error:
        upload.handle(
            OWNER,
            _descriptor(job),
            reader,
            SimpleNamespace(write=lambda _: pytest.fail("unexpected result")),
        )

    assert error.value.code is ErrorCode.STALE_FENCE
    assert ledger.list_inputs(job["job_id"]) == []
    assert _active_upload_count(ledger) == 0
    directory = Path(recovery.input_directory(job["job_id"]))
    assert not directory.exists() or list(directory.iterdir()) == []


def test_takeover_also_fences_an_inflight_exact_doput_replay(data_plane):
    _, _, _, ledger, upload, client = data_plane
    job = create_fit(ledger)
    payload_id = str(uuid.uuid4())
    batch = _fit_batch(1)
    _put(client, job, payload_id, 0, [batch])

    def takeover():
        ledger.acquire_job(
            job["job_id"],
            owner_subject=OWNER,
            previous_client_execution_id=job["client_execution_id"],
            expected_fencing_token=1,
            client_execution_id=str(uuid.uuid4()),
            acquire_grace_seconds=30,
        )

    reader = _Reader(job, payload_id, batch, before_eof=takeover)
    with pytest.raises(ServiceError) as error:
        upload.handle(
            OWNER,
            _descriptor(job),
            reader,
            SimpleNamespace(write=lambda _: pytest.fail("unexpected result")),
        )

    assert error.value.code is ErrorCode.STALE_FENCE
    assert [item["payload_id"] for item in ledger.list_inputs(job["job_id"])] == [
        payload_id
    ]


def test_lost_put_result_keeps_the_durable_receipt(data_plane):
    _, _, recovery, ledger, upload, _ = data_plane
    job = create_fit(ledger)
    payload_id = str(uuid.uuid4())

    with pytest.raises(OSError, match="response channel lost"):
        upload.handle(
            OWNER,
            _descriptor(job),
            _Reader(job, payload_id, _fit_batch(1)),
            SimpleNamespace(
                write=lambda _: (_ for _ in ()).throw(
                    OSError("response channel lost")
                )
            ),
        )

    record = ledger.list_inputs(job["job_id"])[0]
    assert record["payload_id"] == payload_id
    assert os.path.isfile(recovery.absolute_path(record["relative_path"]))


def _write_output(path):
    output_type = pa.list_(pa.float32(), 6)
    schema = pa.schema([
        pa.field("out", output_type, nullable=False),
    ])
    table = pa.Table.from_arrays(
        [pa.array([[float(value) for value in range(6)]], type=output_type)],
        schema=schema,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
    raw = Path(path).read_bytes()
    return table, {
        "ordinal": 0,
        "rows": 1,
        "batches": 1,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_fingerprint": hashlib.sha256(
            schema.serialize().to_pybytes()
        ).hexdigest(),
    }


def test_output_is_unavailable_until_one_terminal_publication(data_plane):
    _, spool, _, ledger, _, client = data_plane
    job = create_predict(ledger)
    commit_input(ledger, job, 0, rows=1, storage_class="runtime")
    close_input(ledger, job)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v4", "jobs", job["job_id"], "outputs", "0"
    )

    with pytest.raises(pa.ArrowInvalid, match="FAILED_PRECONDITION"):
        client.get_flight_info(descriptor, options=_auth())

    running = ledger.claim_execution_job(job["job_id"], "cpu")
    path = spool.attempt_output_path(job["job_id"], running.attempt, 0)
    table, output = _write_output(path)
    output["relative_path"] = spool.relative_path(path)
    ledger.publish_outputs(
        job["job_id"],
        running.attempt,
        [output],
        attempt_id=running.attempt_id,
        result={"outputs": [{"ordinal": 0, "rows": 1}]},
    )

    info = client.get_flight_info(descriptor, options=_auth())
    downloaded = client.do_get(info.endpoints[0].ticket, options=_auth()).read_all()
    assert downloaded.equals(table)
    with pytest.raises(flight.FlightUnauthorizedError):
        client.do_get(info.endpoints[0].ticket, options=_auth("other")).read_all()
