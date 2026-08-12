from app.service.bootstrap.data_plane import build_upload_handler


class UploadHandler:
    """Build the current upload adapter through the stable ``app.flight`` API."""

    def __new__(
        cls,
        config,
        ledger,
        spool,
        recovery_store=None,
        *,
        cuda_available=None,
        **kwargs,
    ):
        return build_upload_handler(
            config,
            ledger,
            spool,
            recovery_store,
            cuda_available=cuda_available or (lambda: False),
            **kwargs,
        )


__all__ = ["UploadHandler"]
