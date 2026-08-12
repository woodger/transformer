from app.service.adapters.inbound.flight.output import _stream_batches
from app.service.bootstrap.data_plane import build_output_handler


class OutputHandler:
    """Build the current output adapter through the stable ``app.flight`` API."""

    def __new__(cls, config, ledger, spool, **kwargs):
        return build_output_handler(
            config,
            ledger,
            spool,
            **kwargs,
        )


__all__ = ["OutputHandler", "_stream_batches"]
