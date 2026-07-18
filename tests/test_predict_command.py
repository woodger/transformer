from types import SimpleNamespace

import pytest
import torch

from app.commands.predict import run


def test_predict_rejects_overwriting_input_file(tmp_path):
    path = tmp_path / "input.arrow"
    args = SimpleNamespace(data=str(path), preds_path=str(path))

    with pytest.raises(ValueError, match="must differ from input"):
        run(args, torch.device("cpu"))
