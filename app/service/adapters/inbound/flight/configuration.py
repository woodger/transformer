from typing import Protocol


class FlightUploadLimits(Protocol):
    @property
    def max_batch_bytes(self) -> int: ...

    @property
    def max_payload_bytes(self) -> int: ...

    @property
    def max_rows_per_payload(self) -> int: ...


class FlightServerConfig(FlightUploadLimits, Protocol):
    @property
    def host(self) -> str: ...

    @property
    def port(self) -> int: ...

    @property
    def tls_enabled(self) -> bool: ...

    @property
    def tls_cert_file(self) -> str | None: ...

    @property
    def tls_key_file(self) -> str | None: ...

    @property
    def tls_ca_file(self) -> str | None: ...

    @property
    def tls_require_client_cert(self) -> bool: ...

    def validate(self) -> "FlightServerConfig": ...
