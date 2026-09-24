CONTRACT_NAME = "transformer-model-topology"
CONTRACT_REVISION = 2

DETAIL_ACTION = "transformer.model-topology.v2.detail"

MAX_NODES = 1_024
MAX_EDGES = 4_096
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_REVISION",
    "DETAIL_ACTION",
    "MAX_EDGES",
    "MAX_NODES",
    "MAX_RESPONSE_BYTES",
]
