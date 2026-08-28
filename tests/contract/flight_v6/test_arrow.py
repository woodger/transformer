import io

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from app.worker.data.arrow import (
    empty_predictions_table,
    iter_framed_arrow,
    predictions_to_table,
    table_to_source_tensor,
    table_to_tensors,
    write_arrow,
)
from app.worker.data.tensors import reshape_source

FLOAT_LIST = pa.list_(pa.float32())


def test_reshape_source_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="flattened_features"):
        reshape_source(torch.zeros(2, 3, 4), seq_len=3)


def make_target(values=None):
    values = [0.0] * 6 if values is None else values
    return pa.array([values], type=FLOAT_LIST)


@pytest.mark.parametrize(
    "column_type",
    [
        pa.list_(pa.float32()),
        pa.large_list(pa.float64()),
        pa.list_(pa.float32(), 2),
    ],
)
def test_source_accepts_supported_list_types(column_type):
    table = pa.table({"src": pa.array([[1.0, 2.0]], type=column_type)})

    tensor = table_to_source_tensor(table)

    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [[1.0, 2.0]]


def test_source_conversion_preserves_chunked_sliced_arrow_values():
    column_type = pa.list_(pa.float32())
    first = pa.array(
        [[-1.0, -2.0], [1.0, 2.0]],
        type=column_type,
    ).slice(1)
    second = pa.array(
        [[3.0, 4.0], [5.0, 6.0]],
        type=column_type,
    )
    table = pa.table({
        "src": pa.chunked_array([first, second], type=column_type),
    })

    tensor = table_to_source_tensor(table)

    assert tensor.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]


def test_empty_training_table_preserves_typed_empty_tensors():
    table = pa.table({
        "src": pa.array([], type=FLOAT_LIST),
        "tgt": pa.array([], type=FLOAT_LIST),
    })

    batch = table_to_tensors(table)

    assert batch.features.shape == (0, 0)
    assert batch.targets.shape == (0, 6)


@pytest.mark.parametrize("name", ["src", "tgt"])
@pytest.mark.parametrize("value_type", [pa.int32(), pa.bool_(), pa.string()])
def test_columns_reject_non_float_value_types(name, value_type):
    if value_type == pa.string():
        values = [["0"] * (2 if name == "src" else 6)]
    elif value_type == pa.bool_():
        values = [[False] * (2 if name == "src" else 6)]
    else:
        values = [[0] * (2 if name == "src" else 6)]

    columns = {
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": make_target(),
    }
    columns[name] = pa.array(values, type=pa.list_(value_type))
    table = pa.table(columns)

    with pytest.raises(ValueError, match=r"list<float32> or list<float64>"):
        table_to_tensors(table)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pa.array([None], type=FLOAT_LIST), "null row"),
        (pa.array([[1.0, None]], type=FLOAT_LIST), "null element"),
    ],
)
def test_source_rejects_arrow_nulls(source, message):
    with pytest.raises(ValueError, match=message):
        table_to_source_tensor(pa.table({"src": source}))


def test_source_allows_nan():
    table = pa.table({
        "src": pa.array([[1.0, np.nan]], type=FLOAT_LIST),
    })

    tensor = table_to_source_tensor(table)

    assert torch.isnan(tensor[0, 1])


def test_nonempty_source_requires_positive_width():
    table = pa.table({"src": pa.array([[]], type=FLOAT_LIST)})

    with pytest.raises(ValueError, match="positive list length"):
        table_to_source_tensor(table)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_source_rejects_infinity(value):
    table = pa.table({
        "src": pa.array([[1.0, value]], type=FLOAT_LIST),
    })

    with pytest.raises(ValueError, match="non-finite value"):
        table_to_source_tensor(table)


@pytest.mark.parametrize("name", ["src", "tgt"])
def test_float64_values_must_be_representable_as_float32(name):
    columns = {
        "src": pa.array([[1.0, 2.0]], type=pa.list_(pa.float64())),
        "tgt": pa.array([[0.0] * 6], type=pa.list_(pa.float64())),
    }
    values = [1e100, 0.0] if name == "src" else [1e100] + [0.0] * 5
    columns[name] = pa.array([values], type=pa.list_(pa.float64()))

    with pytest.raises(ValueError, match="outside float32 range"):
        table_to_tensors(pa.table(columns))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_target_rejects_non_finite_values(value):
    target = [0.0] * 6
    target[2] = value
    table = pa.table({
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": make_target(target),
    })

    with pytest.raises(ValueError, match="non-finite value"):
        table_to_tensors(table)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (pa.array([None], type=FLOAT_LIST), "null row"),
        (pa.array([[0.0, 0.0, None, 0.0, 0.0, 0.0]], type=FLOAT_LIST), "null element"),
    ],
)
def test_target_rejects_arrow_nulls(target, message):
    table = pa.table({
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": target,
    })

    with pytest.raises(ValueError, match=message):
        table_to_tensors(table)


def test_target_requires_six_values():
    table = pa.table({
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": pa.array([[0.0] * 5], type=FLOAT_LIST),
    })

    with pytest.raises(ValueError, match="must have list length 6, got 5"):
        table_to_tensors(table)


@pytest.mark.parametrize("position", range(1, 6))
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_target_rejects_unit_interval_violation(position, value):
    target = [0.0] * 6
    target[position] = value
    table = pa.table({
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": make_target(target),
    })

    with pytest.raises(
        ValueError,
        match=rf"position {position} is outside \[0, 1\]",
    ):
        table_to_tensors(table)


@pytest.mark.parametrize("mean_return", [-1.1, 1.1])
def test_target_rejects_mean_return_outside_signed_unit_interval(mean_return):
    target = [0.0] * 6
    target[0] = mean_return
    table = pa.table({
        "src": pa.array([[1.0, 2.0]], type=FLOAT_LIST),
        "tgt": make_target(target),
    })

    with pytest.raises(ValueError, match=r"MeanReturn is outside \[-1, 1\]"):
        table_to_tensors(table)


@pytest.mark.parametrize(
    "name",
    ["src", "tgt"],
)
def test_columns_reject_inconsistent_row_width(name):
    columns = {
        "src": pa.array([[1.0, 2.0], [3.0, 4.0]], type=FLOAT_LIST),
        "tgt": pa.array([[0.0] * 6, [1.0] * 6], type=FLOAT_LIST),
    }
    columns[name] = pa.array(
        [[0.0] * (2 if name == "src" else 6), [0.0]],
        type=FLOAT_LIST,
    )

    with pytest.raises(ValueError, match="inconsistent list length at row 2"):
        table_to_tensors(pa.table(columns))


def test_prediction_tables_always_use_fixed_size_list_float32():
    non_empty = predictions_to_table(
        torch.tensor(
            [[-0.5, 0.2, 0.3, 0.4, 0.5, 0.6]],
            dtype=torch.float64,
        ),
        "predictions",
    )
    empty = empty_predictions_table("predictions")

    expected_type = pa.list_(pa.float32(), 6)
    for table in (non_empty, empty):
        field = table.schema.field("predictions")
        assert field.type == expected_type
        assert field.nullable is False
        assert field.type.value_field.nullable is True


def test_predictions_require_six_finite_values():
    with pytest.raises(ValueError, match=r"shape \[rows, 6\]"):
        predictions_to_table(torch.zeros((1, 5)), "predictions")

    invalid = torch.zeros((1, 6))
    invalid[0, 2] = torch.inf
    with pytest.raises(ValueError, match="only finite values"):
        predictions_to_table(invalid, "predictions")

    with pytest.raises(ValueError, match="row count 2 does not match input row count 1"):
        predictions_to_table(
            torch.zeros((2, 6)),
            "predictions",
            expected_rows=1,
        )


def test_prediction_file_is_atomically_replaced_and_parent_is_created(tmp_path):
    path = tmp_path / "nested" / "predictions.arrow"

    write_arrow(path, torch.tensor([[-0.5, 0.2, 0.3, 0.4, 0.5, 0.6]]), "out")
    write_arrow(path, torch.tensor([[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]), "out")

    with pa.memory_map(str(path), "r") as source:
        table = ipc.RecordBatchFileReader(source).read_all()
    assert table.column("out").to_pylist() == [
        pytest.approx([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ]
    assert not list(path.parent.glob("*.tmp"))


def test_framed_reader_rejects_oversized_payload_before_reading_it():
    stream = io.BytesIO((11).to_bytes(8, byteorder="big") + b"payload")

    with pytest.raises(ValueError, match="exceeds maximum 10 bytes"):
        next(iter_framed_arrow(stream, max_frame_bytes=10))

    assert stream.tell() == 8


def test_framed_reader_accepts_zero_length_terminator():
    stream = io.BytesIO((0).to_bytes(8, byteorder="big"))

    assert list(iter_framed_arrow(stream, max_frame_bytes=1)) == []


def test_framed_reader_rejects_incomplete_header():
    with pytest.raises(EOFError, match="Incomplete framed Arrow header"):
        list(iter_framed_arrow(io.BytesIO(b"short")))


def test_framed_reader_rejects_incomplete_payload():
    stream = io.BytesIO((4).to_bytes(8, byteorder="big") + b"xx")

    with pytest.raises(EOFError, match="Unexpected end of framed Arrow stream"):
        list(iter_framed_arrow(stream, max_frame_bytes=4))
