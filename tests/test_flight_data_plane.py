import errno
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.ipc as ipc
import pytest
from sqlalchemy import func, select

import app.flight.spool as spool_module
import app.flight.upload_session as upload_session_module
from app.database.models import InputUpload
from app.flight.config import FlightServiceConfig
from app.flight.constants import CONTRACT_NAME, FIT_SCHEMA_ID, ErrorCode, JobState
from app.flight.coordinator import JobCoordinator
from app.flight.errors import ServiceError
from app.flight.output import OutputHandler
from app.flight.server import TransformerFlightServer
from app.flight.spool import Spool
from app.flight.upload import UploadHandler


def auth(token="secret", *, timeout=5.0):
    return flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {token}".encode())],
        timeout=timeout,
    )


def new_id():
    return str(uuid.uuid4())


def fit_batch(rows, *, offset=0.0):
    source = [[offset + i, offset + i + 1] for i in range(rows)]
    target = [[0.0, 0.1, 0.0, 0.0, 0.2, 1.0] for _ in range(rows)]
    return pa.record_batch({
        "src": pa.array(source, type=pa.list_(pa.float32())),
        "tgt": pa.array(target, type=pa.list_(pa.float32())),
    })


def upload_metadata(job_id, payload_id, ordinal, rows):
    return json.dumps({
        "contract": CONTRACT_NAME,
        "version": 1,
        "jobId": job_id,
        "payloadId": payload_id,
        "ordinal": ordinal,
        "schemaId": FIT_SCHEMA_ID,
        "rows": rows,
    }, separators=(",", ":")).encode()


def create_fit_job(ledger, *, owner="inventory", job_id=None):
    return ledger.create_job(
        job_id=job_id or new_id(),
        owner_subject=owner,
        operation="fit",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        model_label="returns.daily",
        model_config={
            "seq_len": 2,
            "hidden": 8,
            "layers": 1,
            "dropout": 0.0,
            "nhead": 2,
            "context_mode": "relaxed",
            "out_dim": 6,
            "feature_dim": None,
        },
        training_config={"epochs": 1},
    )


def active_upload_count(ledger):
    with ledger.connection() as connection:
        return connection.scalar(select(func.count()).select_from(InputUpload))


@pytest.fixture
def data_plane(tmp_path, postgres_ledger):
    config = FlightServiceConfig(
        runtime_dir=str(tmp_path / "runtime"),
        port=0,
        allow_plaintext=True,
        disk_min_free_bytes=1,
    ).validate()
    spool = Spool(config.runtime_dir, tmp_path / "models").initialize()
    ledger = postgres_ledger
    coordinator = JobCoordinator(
        config,
        ledger,
        spool,
        cuda_available=lambda: False,
    )
    upload = UploadHandler(config, ledger, spool)
    output = OutputHandler(config, ledger, spool)
    server = TransformerFlightServer(
        config,
        coordinator,
        {"secret": "inventory", "other": "another-owner"},
        upload_handler=upload,
        output_handler=output,
    )
    client = flight.FlightClient(("localhost", server.port))
    try:
        yield config, spool, ledger, upload, server, client
    finally:
        client.close()
        server.shutdown()


def put(
    client,
    job_id,
    payload_id,
    ordinal,
    batches,
    rows=None,
    *,
    timeout=5.0,
):
    schema = batches[0].schema if batches else pa.schema([
        ("src", pa.list_(pa.float32())),
        ("tgt", pa.list_(pa.float32())),
    ])
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "inputs", str(ordinal)
    )
    writer, results = client.do_put(
        descriptor,
        schema,
        options=auth(timeout=timeout),
    )
    writer.write_metadata(pa.py_buffer(upload_metadata(
        job_id,
        payload_id,
        ordinal,
        sum(batch.num_rows for batch in batches) if rows is None else rows,
    )))
    for batch in batches:
        writer.write_batch(batch)
    writer.done_writing()
    result = results.read()
    assert results.read() is None
    writer.close()
    return json.loads(result.to_pybytes())


def test_multi_batch_do_put_is_one_durable_logical_input(data_plane):
    _, spool, ledger, _, _, client = data_plane
    job = create_fit_job(ledger)
    payload_id = new_id()

    result = put(client, job["job_id"], payload_id, 0, [fit_batch(2), fit_batch(1, offset=10)])

    assert result["status"] == "committed"
    assert result["rows"] == 3
    assert result["batches"] == 2
    inputs = ledger.list_inputs(job["job_id"])
    assert len(inputs) == 1
    with pa.memory_map(spool.absolute_path(inputs[0]["relative_path"]), "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        assert reader.num_record_batches == 2
        assert sum(reader.get_batch(i).num_rows for i in range(2)) == 3


def test_metadata_only_do_put_commits_typed_empty_input(data_plane):
    _, spool, ledger, _, _, client = data_plane
    job = create_fit_job(ledger)

    result = put(client, job["job_id"], new_id(), 0, [])

    assert result["rows"] == 0
    assert result["batches"] == 0
    record = ledger.list_inputs(job["job_id"])[0]
    with pa.memory_map(spool.absolute_path(record["relative_path"]), "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        assert reader.num_record_batches == 0
        assert reader.schema.names == ["src", "tgt"]


def test_payload_larger_than_four_mib_is_accepted(data_plane):
    _, _, ledger, _, _, client = data_plane
    job = create_fit_job(ledger)
    rows = 150_000
    src_values = pa.array(np.zeros(rows * 2, dtype=np.float32))
    tgt_values = pa.array(np.zeros(rows * 6, dtype=np.float32))
    batch = pa.record_batch({
        "src": pa.FixedSizeListArray.from_arrays(src_values, 2),
        "tgt": pa.FixedSizeListArray.from_arrays(tgt_values, 6),
    })

    result = put(client, job["job_id"], new_id(), 0, [batch], timeout=20.0)

    assert result["bytes"] > 4 * 1024 * 1024
    assert ledger.list_inputs(job["job_id"])[0]["rows"] == rows


def test_exact_and_conflicting_duplicate_uploads_are_distinguished(data_plane):
    _, _, ledger, _, _, client = data_plane
    job = create_fit_job(ledger)
    payload_id = new_id()
    original = [fit_batch(1)]
    first = put(client, job["job_id"], payload_id, 0, original)

    repeated = put(client, job["job_id"], payload_id, 0, original)
    assert repeated == first

    with pytest.raises(Exception, match="ALREADY_EXISTS"):
        put(client, job["job_id"], payload_id, 0, [fit_batch(1, offset=99)])
    assert len(ledger.list_inputs(job["job_id"])) == 1


class DisconnectingReader:
    def __init__(self, schema, metadata, batch):
        self.schema = schema
        self._chunks = iter([
            SimpleNamespace(data=None, app_metadata=pa.py_buffer(metadata)),
            SimpleNamespace(data=batch, app_metadata=None),
        ])

    def read_chunk(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise OSError("client disconnected") from None


class CompleteReader:
    def __init__(self, schema, metadata, batches):
        self.schema = schema
        self._chunks = iter([
            SimpleNamespace(data=None, app_metadata=pa.py_buffer(metadata)),
            *[
                SimpleNamespace(data=batch, app_metadata=None)
                for batch in batches
            ],
        ])

    def read_chunk(self):
        return next(self._chunks)


class CancellingReader(CompleteReader):
    def __init__(self, schema, metadata, batches, cancel, *, cancel_after_read=2):
        super().__init__(schema, metadata, batches)
        self._cancel = cancel
        self._cancel_after_read = cancel_after_read
        self.reads = 0

    def read_chunk(self):
        chunk = super().read_chunk()
        self.reads += 1
        if self.reads == self._cancel_after_read:
            self._cancel()
        return chunk


class RepeatingEmptyReader:
    def __init__(self, schema, metadata, *, maximum_reads=100):
        self.schema = schema
        self.metadata = metadata
        self.maximum_reads = maximum_reads
        self.reads = 0

    def read_chunk(self):
        if self.reads == 0:
            self.reads += 1
            return SimpleNamespace(
                data=None,
                app_metadata=pa.py_buffer(self.metadata),
            )
        if self.reads >= self.maximum_reads:
            raise AssertionError("upload did not enforce physical staging size")
        self.reads += 1
        return SimpleNamespace(
            data=pa.RecordBatch.from_arrays(
                [
                    pa.array([], type=pa.list_(pa.float32())),
                    pa.array([], type=pa.list_(pa.float32())),
                ],
                names=["src", "tgt"],
            ),
            app_metadata=None,
        )


def test_partial_upload_never_becomes_committed(data_plane):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    payload_id = new_id()
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )

    with pytest.raises(OSError, match="disconnected"):
        upload.handle(
            "inventory",
            descriptor,
            DisconnectingReader(
                batch.schema,
                upload_metadata(job["job_id"], payload_id, 0, 1),
                batch,
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert ledger.list_inputs(job["job_id"]) == []
    assert active_upload_count(ledger) == 0
    input_directory = Path(spool.input_directory(job["job_id"]))
    assert not input_directory.exists() or list(input_directory.iterdir()) == []


def test_cancel_during_upload_never_publishes_a_committed_input(data_plane):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )

    reader = CancellingReader(
        batch.schema,
        upload_metadata(job["job_id"], new_id(), 0, 2),
        [batch, fit_batch(1, offset=10)],
        lambda: ledger.transition_job(job["job_id"], JobState.CANCELLED),
    )

    with pytest.raises(ServiceError) as error:
        upload.handle(
            "inventory",
            descriptor,
            reader,
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert error.value.code == ErrorCode.CANCELLED
    # Metadata plus one transport batch were read.  The second batch was not
    # consumed or staged after cancellation committed.
    assert reader.reads == 2
    assert ledger.get_job(job["job_id"])["state"] == JobState.CANCELLED.value
    assert ledger.list_inputs(job["job_id"]) == []
    assert not os.path.exists(spool.input_path(job["job_id"], 0))
    assert active_upload_count(ledger) == 0


def test_lost_put_result_after_durable_commit_does_not_rollback_input(data_plane):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    payload_id = new_id()
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )

    with pytest.raises(OSError, match="response channel lost"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], payload_id, 0, 1),
                [batch],
            ),
            SimpleNamespace(
                write=lambda _: (_ for _ in ()).throw(
                    OSError("response channel lost")
                )
            ),
        )

    committed = ledger.list_inputs(job["job_id"])
    assert len(committed) == 1
    assert committed[0]["payload_id"] == payload_id
    assert os.path.isfile(spool.absolute_path(committed[0]["relative_path"]))


def test_physical_ipc_size_is_limited_during_zero_row_batch_stream(data_plane):
    config, spool, ledger, _, _, _ = data_plane
    limited = replace(
        config,
        max_message_bytes=4096,
        target_batch_bytes=1024,
        max_batch_bytes=4096,
        max_payload_bytes=4096,
    ).validate()
    upload = UploadHandler(limited, ledger, spool)
    job = create_fit_job(ledger)
    schema = fit_batch(0).schema
    reader = RepeatingEmptyReader(
        schema,
        upload_metadata(job["job_id"], new_id(), 0, 0),
    )
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )

    with pytest.raises(ServiceError, match="staged IPC payload size"):
        upload.handle(
            "inventory",
            descriptor,
            reader,
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert reader.reads < reader.maximum_reads
    assert ledger.list_inputs(job["job_id"]) == []
    assert active_upload_count(ledger) == 0


def test_oversized_ordinal_is_rejected_before_staging(data_plane):
    config, spool, ledger, _, _, _ = data_plane
    limited = replace(config, max_payloads_per_job=10).validate()
    upload = UploadHandler(limited, ledger, spool)
    job = create_fit_job(ledger)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs",
        str(limited.max_payloads_per_job),
    )

    with pytest.raises(ServiceError, match="per-job payload limit"):
        upload.handle(
            "inventory",
            descriptor,
            SimpleNamespace(schema=fit_batch(0).schema),
            SimpleNamespace(),
        )

    assert ledger.list_inputs(job["job_id"]) == []


def test_pathological_json_integer_in_upload_metadata_is_invalid_not_internal(
    data_plane,
):
    _, _, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    schema = fit_batch(0).schema
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    body = b'{"ordinal":' + (b"9" * 5000) + b"}"

    with pytest.raises(ServiceError, match="valid UTF-8 JSON"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(schema, body, []),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert ledger.list_inputs(job["job_id"]) == []


def test_upload_metadata_document_has_a_size_limit(data_plane):
    _, _, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    schema = fit_batch(0).schema
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )

    with pytest.raises(ServiceError, match="exceeds 65536 bytes"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(schema, b" " * (64 * 1024 + 1), []),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert ledger.list_inputs(job["job_id"]) == []


def test_commit_failure_removes_published_file_and_reservation(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    payload_id = new_id()
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    monkeypatch.setattr(
        ledger,
        "commit_input",
        lambda **_: (_ for _ in ()).throw(RuntimeError("injected ledger failure")),
    )

    with pytest.raises(RuntimeError, match="injected ledger failure"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], payload_id, 0, 1),
                [batch],
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert not os.path.exists(spool.input_path(job["job_id"], 0))
    assert ledger.list_inputs(job["job_id"]) == []
    assert active_upload_count(ledger) == 0


def test_disk_full_during_durable_rename_has_stable_code_and_no_commit(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    monkeypatch.setattr(
        spool,
        "durable_replace",
        lambda *_: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "injected rename disk full")
        ),
    )

    with pytest.raises(ServiceError) as error:
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], new_id(), 0, 1),
                [batch],
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert error.value.code.value == "DISK_FULL"
    assert ledger.list_inputs(job["job_id"]) == []
    assert active_upload_count(ledger) == 0


def test_post_rename_directory_fsync_failure_leaves_no_live_orphan(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    # Ensure parent creation is not the injected failure point; the failure is
    # specifically after os.replace has published the final filename.
    os.makedirs(spool.input_directory(job["job_id"]), exist_ok=True)
    monkeypatch.setattr(
        spool_module,
        "fsync_directory",
        lambda _: (_ for _ in ()).throw(
            OSError(errno.EIO, "injected post-rename directory fsync failure")
        ),
    )

    with pytest.raises(OSError, match="post-rename directory fsync"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], new_id(), 0, 1),
                [batch],
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert ledger.list_inputs(job["job_id"]) == []
    assert not os.path.exists(spool.input_path(job["job_id"], 0))
    assert active_upload_count(ledger) == 0


def test_post_staging_watermark_does_not_double_count_payload_bytes(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    calls = []
    monkeypatch.setattr(
        spool,
        "ensure_free_space",
        lambda minimum, *, required_bytes=0: calls.append(
            (minimum, required_bytes)
        ),
    )

    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    upload.handle(
        "inventory",
        descriptor,
        CompleteReader(
            batch.schema,
            upload_metadata(job["job_id"], new_id(), 0, 1),
            [batch],
        ),
        SimpleNamespace(write=lambda _: None),
    )

    assert calls == [
        (upload.config.disk_min_free_bytes, 0),
        (upload.config.disk_min_free_bytes, 0),
    ]


def test_ipc_close_failure_still_cleans_temporary_and_reservation(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    original_new_file = upload_session_module.ipc.new_file

    class FailingCloseWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write_batch(self, value):
            self.wrapped.write_batch(value)

        def close(self):
            self.wrapped.close()
            raise OSError(errno.ENOSPC, "injected IPC close failure")

    monkeypatch.setattr(
        upload_session_module.ipc,
        "new_file",
        lambda *args, **kwargs: FailingCloseWriter(
            original_new_file(*args, **kwargs)
        ),
    )

    with pytest.raises(ServiceError, match="runtime directory is full"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], new_id(), 0, 1),
                [batch],
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    assert ledger.list_inputs(job["job_id"]) == []
    assert active_upload_count(ledger) == 0
    directory = Path(spool.input_directory(job["job_id"]))
    assert not directory.exists() or list(directory.iterdir()) == []


def test_post_commit_exception_preserves_ledger_referenced_input(
    data_plane,
    monkeypatch,
):
    _, spool, ledger, upload, _, _ = data_plane
    job = create_fit_job(ledger)
    batch = fit_batch(1)
    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job["job_id"], "inputs", "0"
    )
    original_commit = ledger.commit_input

    def commit_then_raise(**kwargs):
        original_commit(**kwargs)
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(ledger, "commit_input", commit_then_raise)

    with pytest.raises(RuntimeError, match="post-commit"):
        upload.handle(
            "inventory",
            descriptor,
            CompleteReader(
                batch.schema,
                upload_metadata(job["job_id"], new_id(), 0, 1),
                [batch],
            ),
            SimpleNamespace(write=lambda _: pytest.fail("unexpected PutResult")),
        )

    committed = ledger.list_inputs(job["job_id"])
    assert len(committed) == 1
    assert os.path.isfile(spool.absolute_path(committed[0]["relative_path"]))


def write_output(path, *, empty=False):
    table = pa.table({
        "out": pa.array(
            [] if empty else [[float(i) for i in range(6)]],
            type=pa.list_(pa.float32()),
        )
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    data = open(path, "rb").read()
    with pa.memory_map(path, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        batches = reader.num_record_batches
    return {
        "ordinal": 0,
        "rows": table.num_rows,
        "batches": batches,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "schema_fingerprint": hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest(),
    }


@pytest.mark.parametrize("empty", [False, True])
def test_get_flight_info_issues_opaque_ticket_and_do_get_streams_output(data_plane, empty):
    _, spool, ledger, _, _, client = data_plane
    job_id = new_id()
    ledger.create_job(
        job_id=job_id,
        owner_subject="inventory",
        operation="predict",
        requested_device="cpu",
        prediction_column="out",
        config_hash="a" * 64,
        input_model_ref="mdl_seed",
        model_config={
            "seq_len": 2, "hidden": 8, "layers": 1, "dropout": 0.0,
            "nhead": 2, "context_mode": "relaxed", "out_dim": 6,
            "feature_dim": 1,
        },
    )
    ledger.seal_job(job_id, manifest_hash="a" * 64, manifest=[])
    ledger.queue_job(job_id, selected_device="cpu")
    running = ledger.claim_next_job("cpu")
    path = spool.attempt_output_path(job_id, running["attempt"], 0)
    output = write_output(path, empty=empty)
    output["relative_path"] = spool.relative_path(path)
    ledger.publish_outputs(job_id, running["attempt"], [output], result={})

    descriptor = flight.FlightDescriptor.for_path(
        "transformer", "v1", "jobs", job_id, "outputs", "0"
    )
    info = client.get_flight_info(descriptor, options=auth())
    ticket = info.endpoints[0].ticket

    assert b"/" not in ticket.ticket
    assert job_id.encode() not in ticket.ticket
    table = client.do_get(ticket, options=auth()).read_all()
    assert table.num_rows == (0 if empty else 1)
    assert table.schema.field("out").type == pa.list_(pa.float32())

    with pytest.raises(flight.FlightUnauthorizedError):
        client.do_get(ticket, options=auth("other")).read_all()
