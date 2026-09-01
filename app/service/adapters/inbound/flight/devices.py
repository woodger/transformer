def device_from_api(value: str) -> str:
    if value == "gpu":
        return "cuda"
    if value in ("cpu", "auto"):
        return value
    raise ValueError(f"unsupported Flight device: {value}")


def device_to_api(value: str | None) -> str | None:
    if value == "cuda":
        return "gpu"
    if value in (None, "cpu", "auto"):
        return value
    raise ValueError(f"unsupported internal device: {value}")


__all__ = ["device_from_api", "device_to_api"]
