import time
from pathlib import Path

import pyarrow.flight as flight

from app.service.adapters.inbound.flight.auth import (
    MIDDLEWARE_KEY,
    BearerAuthMiddlewareFactory,
    authenticated_subject,
)
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
    parse_action_body,
    validate_action_request,
)
from app.service.adapters.inbound.flight.errors import (
    ServiceError,
    invalid,
    to_flight_exception,
)
from app.service.adapters.observability import JsonLogger, OperationalMetrics

ACTION_DESCRIPTIONS = {
    CAPABILITIES_ACTION: "Return Flight v3 capabilities and limits.",
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


class TransformerFlightServer(flight.FlightServerBase):
    """Thin Flight adapter; Torch execution belongs to the worker subprocess."""

    def __init__(
        self,
        config,
        coordinator,
        token_cache,
        *,
        upload_handler=None,
        output_handler=None,
        metrics: OperationalMetrics | None = None,
        logger: JsonLogger | None = None,
    ):
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
        super().__init__(
            (self.config.host, self.config.port),
            middleware=middleware,
            **tls_server_options(self.config),
        )

    def list_actions(self, context):
        authenticated_subject(context)
        return [
            flight.ActionType(name, ACTION_DESCRIPTIONS[name])
            for name in ACTIONS
        ]

    def do_action(self, context, action):
        started = time.monotonic()
        request = None
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
        except flight.FlightError:
            self._log_action(action.type, request, "FLIGHT_ERROR", started)
            raise
        except Exception as exc:
            self._log_action(action.type, request, "INTERNAL", started)
            self._raise_internal("DoAction", exc)

    def do_put(self, context, descriptor, reader, writer):
        try:
            owner = authenticated_subject(context)
            if self.upload_handler is None:
                raise invalid("uploads are not configured")
            self.upload_handler.handle(owner, descriptor, reader, writer)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except flight.FlightError:
            raise
        except Exception as exc:
            self._raise_internal("DoPut", exc)

    def get_flight_info(self, context, descriptor):
        try:
            owner = authenticated_subject(context)
            if self.output_handler is None:
                raise invalid("outputs are not configured")
            return self.output_handler.get_flight_info(owner, descriptor)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except flight.FlightError:
            raise
        except Exception as exc:
            self._raise_internal("GetFlightInfo", exc)

    def do_get(self, context, ticket):
        try:
            owner = authenticated_subject(context)
            if self.output_handler is None:
                raise invalid("outputs are not configured")
            return self.output_handler.do_get(context, owner, ticket)
        except ServiceError as exc:
            raise to_flight_exception(exc) from None
        except flight.FlightError:
            raise
        except Exception as exc:
            self._raise_internal("DoGet", exc)

    def _raise_internal(self, method: str, error: Exception):
        self.logger.event(
            "flight.rpc.internal_error",
            method=method,
            errorType=type(error).__name__,
        )
        raise flight.FlightInternalError(
            "INTERNAL: internal service error"
        ) from None

    def _log_action(self, action: str, request, status: str, started: float) -> None:
        fields = {
            "action": action,
            "status": status,
            "latencyMs": round((time.monotonic() - started) * 1000.0, 3),
        }
        if isinstance(request, dict):
            if request.get("request_id") is not None:
                fields["requestId"] = request["request_id"]
            if request.get("job_id") is not None:
                fields["jobId"] = request["job_id"]
        self.logger.event("flight.action.completed", **fields)


def tls_server_options(config) -> dict:
    if not config.tls_enabled:
        return {}
    certificate = Path(config.tls_cert_file).read_bytes()
    private_key = Path(config.tls_key_file).read_bytes()
    options = {"tls_certificates": [(certificate, private_key)]}
    if config.tls_require_client_cert:
        options.update({
            "verify_client": True,
            "root_certificates": Path(config.tls_ca_file).read_bytes(),
        })
    return options
