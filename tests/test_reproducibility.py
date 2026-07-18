import random
from types import SimpleNamespace

import numpy as np
import torch

import app.main as main_module
from app.runtime.reproducibility import configure_reproducibility


def test_configure_reproducibility_repeats_random_sequences():
    configure_reproducibility(1234)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    configure_reproducibility(1234)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_configure_reproducibility_sets_deterministic_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)

    configure_reproducibility(7, deterministic=True)
    configure_reproducibility(7, deterministic=False)

    assert calls == [True, False]


def test_main_applies_training_reproducibility_before_fit(monkeypatch):
    args = SimpleNamespace(
        action="fit",
        data="train.arrow",
        device="cpu",
        metrics_name=None,
        seed=17,
        deterministic=True,
    )
    calls = []

    monkeypatch.setattr(main_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        main_module,
        "configure_reproducibility",
        lambda seed, deterministic: calls.append(("seed", seed, deterministic)),
    )
    monkeypatch.setattr(
        main_module,
        "get_device",
        lambda device: calls.append(("device", device)) or torch.device("cpu"),
    )
    monkeypatch.setattr(main_module, "reset_metrics_log", lambda path: None)
    monkeypatch.setattr(main_module, "run_fit", lambda *args: calls.append(("fit",)))

    main_module.main()

    assert calls == [("seed", 17, True), ("device", "cpu"), ("fit",)]
