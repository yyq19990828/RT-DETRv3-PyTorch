from types import SimpleNamespace

from ppdet_pytorch.cli import train as train_cli


def test_run_calls_parameterless_trainer_train(monkeypatch):
    calls = []

    class FakeTrainer:
        def __init__(self, cfg, mode):
            calls.append(("init", cfg, mode))

        def train(self, *args):
            calls.append(("train", args))

    monkeypatch.setattr(train_cli, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_cli, "init_parallel_env", lambda: None)
    flags = SimpleNamespace(
        ddp=False,
        enable_ce=False,
        seed=None,
        resume=None,
        eval=False,
    )
    config = {}

    train_cli.run(flags, config)

    assert calls == [("init", config, "train"), ("train", ())]


def test_run_applies_explicit_seed_per_rank(monkeypatch):
    calls = []

    class FakeTrainer:
        def __init__(self, cfg, mode):
            calls.append(("init", dict(cfg), mode))

        def train(self):
            calls.append(("train",))

    monkeypatch.setattr(train_cli, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_cli, "init_fleet_env", lambda unused: None)
    monkeypatch.setattr(train_cli.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(train_cli.torch.distributed, "get_rank", lambda: 2)
    monkeypatch.setattr(
        train_cli, "set_random_seed", lambda seed: calls.append(("seed", seed)))
    flags = SimpleNamespace(
        ddp=True,
        enable_ce=False,
        seed=7,
        resume=None,
        eval=False,
    )
    config = {"ddp": True, "find_unused_parameters": False}

    train_cli.run(flags, config)

    assert calls == [
        ("seed", 9),
        ("init", {"ddp": True, "find_unused_parameters": False, "seed": 7}, "train"),
        ("train",),
    ]


def test_run_keeps_enable_ce_seed_zero_compatibility(monkeypatch):
    seeds = []

    class FakeTrainer:
        def __init__(self, cfg, mode):
            pass

        def train(self):
            pass

    monkeypatch.setattr(train_cli, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_cli, "init_parallel_env", lambda: None)
    monkeypatch.setattr(train_cli, "set_random_seed", seeds.append)
    flags = SimpleNamespace(
        ddp=False,
        enable_ce=True,
        seed=None,
        resume=None,
        eval=False,
    )
    config = {}

    train_cli.run(flags, config)

    assert seeds == [0]
    assert config["seed"] == 0
