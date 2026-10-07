from __future__ import annotations

import os
from typing import BinaryIO

from app.contracts.json_types import JsonObject
from app.contracts.worker.v22 import validate_document
from app.worker.application.artifacts import (
    artifact_document,
    validate_workspace,
    write_json_once,
)
from app.worker.application.documents import integer_field, object_field, string_field
from app.worker.application.errors import WorkerExecutionError
from app.worker.application.events import WorkerEventEmitter
from app.worker.application.fit import execute_fit
from app.worker.application.inputs import DurableInputStream
from app.worker.application.predict import execute_predict


class WorkerApplication:
    """Выполнить один долговременный потоковый манифест команды Worker v22."""

    def __init__(
        self,
        emitter: WorkerEventEmitter,
        input_stream: BinaryIO | None = None,
    ) -> None:
        self.emitter = emitter
        self.input_stream = input_stream

    def run(self, manifest: JsonObject) -> None:
        validate_document(manifest, "command-manifest")
        self._validate_identity(manifest)
        workspace = validate_workspace(
            string_field(object_field(manifest, "workspace"), "root")
        )
        if self.input_stream is None:
            raise ValueError("worker control stream is unavailable")
        inputs = DurableInputStream(
            manifest,
            self.input_stream,
            self.emitter,
        )
        self.emitter.ready(
            next_ordinal=inputs.next_ordinal,
            input_revision=inputs.input_revision,
        )
        if string_field(manifest, "operation") == "fit":
            result = execute_fit(manifest, workspace, inputs, self.emitter)
        else:
            result = execute_predict(manifest, workspace, inputs, self.emitter)
        result_path = os.path.join(workspace, "worker-result.json")
        write_json_once(result_path, result)
        self.emitter.completed(artifact_document(result_path))

    def _validate_identity(self, manifest: JsonObject) -> None:
        expected = (
            self.emitter.job_id,
            self.emitter.attempt,
            self.emitter.attempt_id,
        )
        actual = (
            string_field(manifest, "jobId"),
            integer_field(manifest, "attempt"),
            string_field(manifest, "attemptId"),
        )
        if actual != expected:
            raise ValueError("worker command identity does not match argv")


__all__ = ["WorkerApplication", "WorkerExecutionError"]
