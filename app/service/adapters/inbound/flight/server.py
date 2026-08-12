import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NoReturn, Protocol, cast

import pyarrow.flight as flight

from app.service.adapters.inbound.flight.auth import (
    MIDDLEWARE_KEY,
    BearerAuthMiddlewareFactory,
    authenticated_subject,
)
from app.service.adapters.inbound.flight.configuration import FlightServerConfig
from app.service.adapters.inbound.flight.constants import (
    ACQUIRE_ACTION,
    ACTIONS,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CREATE_ACTION,
    HEALTH_ACTION,
    INPUT_CLOSE_ACTION,
    INPUTS_LIST_ACTION,
    MODEL_DESCRIBE_ACTION,
    OUTPUTS_LIST_ACTION,
    STATUS_ACTION,
)
from app.service.adapters.inbound.flight.contract import (
    ValidatedActionRequest,
    parse_action_body,
    validate_action_request,
)
from app.service.adapters.inbound.flight.coordinator import JobCoordinator
from app.service.adapters.inbound.flight.errors import (
    ServiceError,
    invalid,
    to_flight_exception,
)
from app.service.adapters.inbound.flight.output import OutputHandler
from app.service.adapters.inbound.flight.upload import PutMetadataWriter, UploadHandler
from app.service.adapters.inbound.flight.upload_session import FlightStreamReader
from app.service.adapters.observability import JsonLogger, OperationalMetrics
from app.service.domain.access import AuthIdentity

ACTION_DESCRIPTIONS = {
    CAPABILITIES_ACTION: "Return Flight v4 capabilities and limits.",
    HEALTH_ACTION: "Return liveness, readiness and device health.",
    CREATE_ACTION: "Create a durable streaming job.",
    ACQUIRE_ACTION: "Transfer externally fenced job ownership.",
    STATUS_ACTION: "Read durable job status.",
    INPUTS_LIST_ACTION: "List committed inputs by revision snapshot.",
    INPUT_CLOSE_ACTION: "Commit EOF and the immutable input manifest.",
    OUTPUTS_LIST_ACTION: "List atomically published outputs.",
    CANCEL_ACTION: "Cancel a job.",
    MODEL_DESCRIBE_ACTION: "Describe one immutable model generation.",
}


class _TokenCache(Protocol):
    def lookup(self, token: str) -> AuthIdentity | None: ...


class _CallContext(Protocol):
    def get_middleware(self, key: str) -> object | None: ...

    def is_cancelled(self) -> bool: ...


class _Action(Protocol):
    type: str
    body: object


class _ActionTypeFactory(Protocol):
    def __call__(self, name: str, description: str) -> object: ...


class _ExceptionFactory(Protocol):
    def __call__(self, message: str) -> Exception: ...


_FLIGHT_ERROR = cast(type[Exception], vars(flight)["FlightError"])


class TransformerFlightServer(
    flight.FlightServerBase,  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage, reportUntypedBaseClass]
):
    """Thin Flight adapter; Torch execution belongs to the worker subprocess."""

    def __init__(
        self,
        config: FlightServerConfig,
        coordinator: JobCoordinator,
        token_cache: _TokenCache | dict[str, str],
        *,
        upload_handler: UploadHandler | None = None,
        output_handler: OutputHandler | None = None,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
    ) -> None:
        self.config = config.validate()
        self.coordinator = coordinator
        self.upload_handler = upload_handler
        self.output_handler = output_handler
        self.metrics = metrics or OperationalMetrics()
        self.logger = logger or JsonLogger()
        middleware = {
            MIDDLEWARE_KEY: BearerAuthMiddlewareFactory(
                token_cache,
                self.metrics,
                self.logger,
            )
        }
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            (self.config.host, self.config.port),
            middleware=middleware,
            **tls_server_options(self.config),
        )

    def list_actions(self, context: _CallContext) -> list[object]:
        authenticated_subject(context)
        action_type = cast(_ActionTypeFactory, vars(flight)["ActionType"])
        return [
            action_type(name, ACTION_DESCRIPTIONS[name])
            for name in ACTIONS
        ]

    def do_action(
        self,
        context: _CallContext,
        action: _Action,
    ) -> Iterator[bytes]:
        started = time.monotonic()
        request: ValidatedActionRequest | None = None
        try:
            owner = authenticated_subject(context)
            if action.type not in ACTIONS:
                raise invalid(f"unsupported action: {action.type}")
            document = parse_action_body(action.body)
            request = validate_action_request(action.type, document)
            response = self.coordinator.dispatch(
                action.type,
                owner,
                request,
                document,
            )
            self._log_action(action.type, request, "OK", started)
            return iter([response])
        except ServiceError as exc:
            self._log_action(action.type, request, exc.code.value, started)
            raise to_flight_exception(exc) from None
        except _FLIGHT_ERROR:
            self._log_action(action.type, request, "FLIGHT_ERROR", started)
            raise
        except Exception as exc:
            self._log_action(action.type, request, "INTERNAL", started)
            self._raise_internal("DoAction", exc)

    def do_put(
        self,
        context: _CallContext,
        descriptor: object,
        reader: FlightStreamReader,
        writer: PutMetadataWriter,
    ) -> None:
        try:
            owner = authenticated_subject(context)
            if self.upload_handler is None:
                raise invalid("uploads are not configured")
            self.upload_handler.handle(owner, descriptor, reader, writer)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except _FLIGHT_ERROR:
            raise
        except Exception as exc:
            self._raise_internal("DoPut", exc)

    def get_flight_info(
        self,
        context: _CallContext,
        descriptor: object,
    ) -> object:
        try:
            owner = authenticated_subject(context)
            if self.output_handler is None:
                raise invalid("outputs are not configured")
            return self.output_handler.get_flight_info(owner, descriptor)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except _FLIGHT_ERROR:
            raise
        except Exception as exc:
            self._raise_internal("GetFlightInfo", exc)

    def do_get(
        self,
        context: _CallContext,
        ticket: object,
    ) -> object:
        try:
            owner = authenticated_subject(context)
            if self.output_handler is None:
                raise invalid("outputs are not configured")
            return self.output_handler.do_get(context, owner, ticket)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except _FLIGHT_ERROR:
            raise
        except Exception as exc:
            self._raise_internal("DoGet", exc)

    def _raise_internal(self, method: str, error: Exception) -> NoReturn:
        self.logger.event(
            "flight.rpc.internal_error",
            method=method,
            errorType=type(error).__name__,
        )
        factory = cast(_ExceptionFactory, vars(flight)["FlightInternalError"])
        raise factory("INTERNAL: internal service error") from None

    def _log_action(
        self,
        action: str,
        request: ValidatedActionRequest | None,
        status: str,
        started: float,
    ) -> None:
        fields: dict[str, object] = {
            "action": action,
            "status": status,
            "latencyMs": round((time.monotonic() - started) * 1000.0, 3),
        }
        if request is not None:
            fields["requestId"] = request["request_id"]
            request_mapping = cast(Mapping[str, object], request)
            job_id = request_mapping.get("job_id")
            if isinstance(job_id, str):
                fields["jobId"] = job_id
        self.logger.event("flight.action.completed", **fields)


def tls_server_options(config: FlightServerConfig) -> dict[str, object]:
    if not config.tls_enabled:
        return {}
    if config.tls_cert_file is None or config.tls_key_file is None:
        raise ValueError("TLS certificate and private key are required")
    certificate = Path(config.tls_cert_file).read_bytes()
    private_key = Path(config.tls_key_file).read_bytes()
    options: dict[str, object] = {
        "tls_certificates": [(certificate, private_key)]
    }
    if config.tls_require_client_cert:
        if config.tls_ca_file is None:
            raise ValueError("TLS CA certificate is required for mTLS")
        options.update({
            "verify_client": True,
            "root_certificates": Path(config.tls_ca_file).read_bytes(),
        })
    return options
