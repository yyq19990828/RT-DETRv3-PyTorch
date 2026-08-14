import random

import numpy as np
import torch

from detrs.engine.env import set_random_seed


def test_set_random_seed_covers_python_numpy_and_torch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    set_random_seed(53)
    first = (random.random(), np.random.random(), torch.rand(()).item())
    set_random_seed(53)
    repeated = (random.random(), np.random.random(), torch.rand(()).item())

    assert first == repeated
