OBJECTIVE_LANGUAGE_REVISION = 6

CONSTRAINTS = ("ClosedInterval", "Finite")
TRANSFORMATIONS = ("Identity", "Sigmoid", "Tanh")
RESOURCE_CLASSES = ("PositiveScalarPerObservation",)
DIRECT_OPERATORS = (
    "BinaryCrossEntropyWithLogits",
    "LogMSE",
    "PositiveClassWeightedBinaryCrossEntropyWithLogits",
    "SmoothL1",
)
AUXILIARY_OPERATORS = (
    "BernoulliConfidencePenalty",
    "ExpectedValue",
    "GaussianNLL",
    "RiskAdjustedExpectedValue",
)
AGGREGATIONS = ("WeightedSum",)
REDUCTIONS = ("GlobalRowMean",)
ENCODER_NORMALIZATION_ORDERS = ("postNorm", "preNorm")

MAX_TARGET_SLOTS = 128
MAX_OBJECTIVE_COMPONENTS = 256
MAX_PRIVATE_RESOURCES = 64

__all__ = [
    "AGGREGATIONS",
    "AUXILIARY_OPERATORS",
    "CONSTRAINTS",
    "DIRECT_OPERATORS",
    "ENCODER_NORMALIZATION_ORDERS",
    "MAX_OBJECTIVE_COMPONENTS",
    "MAX_PRIVATE_RESOURCES",
    "MAX_TARGET_SLOTS",
    "OBJECTIVE_LANGUAGE_REVISION",
    "REDUCTIONS",
    "RESOURCE_CLASSES",
    "TRANSFORMATIONS",
]
