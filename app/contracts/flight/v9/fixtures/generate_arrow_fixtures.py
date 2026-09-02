#!/usr/bin/env python3
"""Generate the normative Arrow IPC fixtures for Flight contract v9."""

from argparse import ArgumentParser
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import pyarrow as pa
import pyarrow.ipc as ipc

from app.contracts.flight.v9.arrow import (
    canonical_input_schema,
    canonical_prediction_schema,
)

FIXTURE_NAMES = (
    "fit-multi-batch.arrow",
    "predict-multi-batch.arrow",
    "predict-typed-empty.arrow",
    "prediction-output.arrow",
    "prediction-output-typed-empty.arrow",
)


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
    columns: list[list[list[float]]],
) -> pa.RecordBatch:
    arrays = [
        pa.array(values, type=field.type)
        for field, values in zip(schema, columns, strict=True)
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def generate(output_dir: Path) -> None:
    fit_schema = canonical_input_schema("fit", 4)
    fit_batches = [
        _batch(
            fit_schema,
            [
                [
                    [0.1, float("nan"), 0.3, 0.4],
                    [1.0, 2.0, 3.0, 4.0],
                ],
                [
                    [0.01, 0.02, 0.03, 0.04, 0.20, 0.0],
                    [-0.10, 0.05, 0.00, 0.03, 0.35, 1.0],
                ],
            ],
        ),
        _batch(
            fit_schema,
            [
                [[-1.0, -2.0, -3.0, -4.0]],
                [[0.25, -0.15, 0.10, -0.05, 0.00, 0.5]],
            ],
        ),
    ]
    _write(output_dir / "fit-multi-batch.arrow", fit_schema, fit_batches)

    predict_schema = canonical_input_schema("predict", 4)
    predict_batches = [
        _batch(
            predict_schema,
            [[[10.0, 11.0, 12.0, 13.0], [20.0, float("nan"), 22.0, 23.0]]],
        ),
        _batch(predict_schema, [[[30.0, 31.0, 32.0, 33.0]]]),
    ]
    _write(
        output_dir / "predict-multi-batch.arrow",
        predict_schema,
        predict_batches,
    )

    empty_predict_schema = canonical_input_schema("predict", 4)
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
