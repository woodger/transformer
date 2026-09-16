OBJECTIVE_LANGUAGE_REVISION = 3

CONSTRAINTS = ("ClosedInterval", "Finite")
TRANSFORMATIONS = ("Identity", "Sigmoid", "Tanh")
RESOURCE_CLASSES = ("PositiveScalarPerObservation",)
DIRECT_OPERATORS = (
    "BinaryCrossEntropyWithLogits",
    "LogMSE",
    "SmoothL1",
)
AUXILIARY_OPERATORS = (
    "ExpectedValue",
    "GaussianNLL",
    "RiskAdjustedExpectedValue",
)
AGGREGATIONS = ("WeightedSum",)
REDUCTIONS = ("GlobalRowMean",)

MAX_TARGET_SLOTS = 128
MAX_OBJECTIVE_COMPONENTS = 256
MAX_PRIVATE_RESOURCES = 64

__all__ = [
    "AGGREGATIONS",
    "AUXILIARY_OPERATORS",
    "CONSTRAINTS",
    "DIRECT_OPERATORS",
    "MAX_OBJECTIVE_COMPONENTS",
    "MAX_PRIVATE_RESOURCES",
    "MAX_TARGET_SLOTS",
    "OBJECTIVE_LANGUAGE_REVISION",
    "REDUCTIONS",
    "RESOURCE_CLASSES",
    "TRANSFORMATIONS",
]
