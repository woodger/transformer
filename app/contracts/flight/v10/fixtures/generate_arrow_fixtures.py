#!/usr/bin/env python3
"""Generate the normative Arrow IPC fixtures for Flight contract v10."""

from argparse import ArgumentParser
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.flight.v10.arrow import (
    canonical_input_schema,
    canonical_prediction_schema,
)

FIXTURE_NAMES = (
    "fit-multi-batch.arrow",
    "predict-multi-batch.arrow",
    "predict-production-core-v2.arrow",
    "predict-production-core-v6-hour-boundary.arrow",
    "predict-missing-observation.arrow",
    "predict-missing-native-prefix.arrow",
    "predict-typed-empty.arrow",
    "prediction-output.arrow",
    "prediction-output-typed-empty.arrow",
)

SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 2, "nativeRowWidth": 1},
        {"position": 2, "windowRows": 1, "nativeRowWidth": 2},
    ],
}
SEQ_LEN = 2
FEATURE_DIM = 4

PRODUCTION_CORE_V2_SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 100, "nativeRowWidth": 891},
    ],
}
PRODUCTION_CORE_V2_FEATURE_DIM = 89_100

PRODUCTION_CORE_V6_SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 100, "nativeRowWidth": 648},
        {"position": 64_800, "windowRows": 8, "nativeRowWidth": 6},
    ],
}
PRODUCTION_CORE_V6_FEATURE_DIM = 64_848

MISSING_OBSERVATION_SOURCE_ENCODING = {
    "kind": "indexedFeatureBlocks",
    "featureBlocks": [
        {"position": 0, "windowRows": 2, "nativeRowWidth": 1},
    ],
}


class _RecordBatchWriter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def write_batch(self, batch: pa.RecordBatch) -> None: ...


def _write(path: Path, schema: pa.Schema, batches: list[pa.RecordBatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        # PyArrow does not publish a complete type for the IPC factory. The
        # local protocol captures the only writer operation used here.
        with cast(
            _RecordBatchWriter,
            ipc.new_file(sink, schema),  # pyright: ignore[reportUnknownMemberType]
        ) as writer:
            for batch in batches:
                writer.write_batch(batch)


def _batch(
    schema: pa.Schema,
    columns: list[object],
) -> pa.RecordBatch:
    arrays = [
        pa.array(values, type=field.type)
        for field, values in zip(schema, columns, strict=True)
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _constant_native_rows(
    count: int,
    width: int,
    *,
    base: float = 0.0,
) -> list[list[float]]:
    return [[base + row] * width for row in range(count)]


def generate(output_dir: Path) -> None:
    fit_schema = canonical_input_schema(
        "fit",
        SOURCE_ENCODING,
        SEQ_LEN,
        FEATURE_DIM,
    )
    fit_batches = [
        _batch(
            fit_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": [[0.1], [float("nan")], [0.3], [0.4]],
                        "observationOffsets": [[0, 1], [1, 2]],
                    },
                    "b1": {
                        "nativeRows": [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
                        "observationOffsets": [[0, 1], [1, 2]],
                    },
                }],
                [[
                    [0.01, 0.02, 0.03, 0.04, 0.20, 0.0],
                    [-0.10, 0.05, 0.00, 0.03, 0.35, 1.0],
                ]],
            ],
        ),
        _batch(
            fit_schema,
            [
                [0],
                [2],
                [{
                    "b0": {
                        "nativeRows": [[-1.0], [-2.0], [-3.0]],
                        "observationOffsets": [[0, 1]],
                    },
                    "b1": {
                        "nativeRows": [[-10.0, -11.0], [-12.0, -13.0]],
                        "observationOffsets": [[0, 1]],
                    },
                }],
                [[[0.25, -0.15, 0.10, -0.05, 0.00, 0.5]]],
            ],
        ),
    ]
    _write(output_dir / "fit-multi-batch.arrow", fit_schema, fit_batches)

    predict_schema = canonical_input_schema(
        "predict",
        SOURCE_ENCODING,
        SEQ_LEN,
        FEATURE_DIM,
    )
    predict_batches = [
        _batch(
            predict_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": [[10.0], [11.0], [12.0], [13.0]],
                        "observationOffsets": [[0, 1], [1, 2]],
                    },
                    "b1": {
                        "nativeRows": [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]],
                        "observationOffsets": [[0, 1], [1, 2]],
                    },
                }],
            ],
        ),
        _batch(
            predict_schema,
            [
                [1],
                [0],
                [{
                    "b0": {
                        "nativeRows": [[30.0], [31.0], [32.0]],
                        "observationOffsets": [[0, 1]],
                    },
                    "b1": {
                        "nativeRows": [[40.0, 41.0], [42.0, 43.0]],
                        "observationOffsets": [[0, 1]],
                    },
                }],
            ],
        ),
    ]
    _write(
        output_dir / "predict-multi-batch.arrow",
        predict_schema,
        predict_batches,
    )

    production_core_v2_schema = canonical_input_schema(
        "predict",
        PRODUCTION_CORE_V2_SOURCE_ENCODING,
        10,
        PRODUCTION_CORE_V2_FEATURE_DIM,
    )
    _write(
        output_dir / "predict-production-core-v2.arrow",
        production_core_v2_schema,
        [_batch(
            production_core_v2_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": _constant_native_rows(109, 891),
                        "observationOffsets": [list(range(10))],
                    },
                }],
            ],
        )],
    )

    production_core_v6_schema = canonical_input_schema(
        "predict",
        PRODUCTION_CORE_V6_SOURCE_ENCODING,
        10,
        PRODUCTION_CORE_V6_FEATURE_DIM,
    )
    _write(
        output_dir / "predict-production-core-v6-hour-boundary.arrow",
        production_core_v6_schema,
        [_batch(
            production_core_v6_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": _constant_native_rows(109, 648),
                        "observationOffsets": [list(range(10))],
                    },
                    "b1": {
                        "nativeRows": _constant_native_rows(
                            10,
                            6,
                            base=10_000.0,
                        ),
                        "observationOffsets": [[
                            0, 0, 0, 0, 0, 0, 1, 1, 1, 2,
                        ]],
                    },
                }],
            ],
        )],
    )

    missing_observation_schema = canonical_input_schema(
        "predict",
        MISSING_OBSERVATION_SOURCE_ENCODING,
        2,
        2,
    )
    _write(
        output_dir / "predict-missing-observation.arrow",
        missing_observation_schema,
        [_batch(
            missing_observation_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": _constant_native_rows(5, 1),
                        "observationOffsets": [[0, 2]],
                    },
                }],
            ],
        )],
    )
    _write(
        output_dir / "predict-missing-native-prefix.arrow",
        missing_observation_schema,
        [_batch(
            missing_observation_schema,
            [
                [0],
                [0],
                [{
                    "b0": {
                        "nativeRows": _constant_native_rows(2, 1),
                        "observationOffsets": [[0, 1]],
                    },
                }],
            ],
        )],
    )

    empty_predict_schema = canonical_input_schema(
        "predict",
        SOURCE_ENCODING,
        SEQ_LEN,
        FEATURE_DIM,
    )
    _write(
        output_dir / "predict-typed-empty.arrow",
        empty_predict_schema,
        [],
    )

    prediction_schema = canonical_prediction_schema("out")
    prediction_batches = [
        _batch(
            prediction_schema,
            [[
                [0.10, 0.20, -1.00, 1.00, 0.30, 0.40],
                [-0.20, 0.15, 0.50, -0.50, 0.00, 0.75],
            ]],
        ),
        _batch(
            prediction_schema,
            [[[0.00, 0.05, 0.00, 0.00, 0.10, 1.00]]],
        ),
    ]
    _write(
        output_dir / "prediction-output.arrow",
        prediction_schema,
        prediction_batches,
    )
    _write(
        output_dir / "prediction-output-typed-empty.arrow",
        prediction_schema,
        [],
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("arrow"),
        help="Destination directory (default: fixtures/arrow).",
    )
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
