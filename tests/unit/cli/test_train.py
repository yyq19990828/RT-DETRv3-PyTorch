from types import SimpleNamespace

import pytest

from ppdet_pytorch.cli import train as train_cli
from ppdet_pytorch.core.workspace import AttrDict


def test_parse_args_accepts_supported_training_contract():
    args = train_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--resume",
            "epoch_3.pth",
            "--seed",
            "7",
            "--amp",
            "--ddp",
            "-o",
            "epoch=4",
            "TrainReader.batch_size=2",
        ]
    )

    assert args.resume == "epoch_3.pth"
    assert args.seed == 7
    assert args.amp is True
    assert args.ddp is True
    assert args.opt == {
        "epoch": 4,
        "TrainReader": {"batch_size": 2},
    }


def test_parse_args_requires_config(capsys):
    with pytest.raises(SystemExit) as error:
        train_cli.parse_args([])

    assert error.value.code == 2
    assert "--config" in capsys.readouterr().err


@pytest.mark.parametrize(
    "unsupported_args",
    [
        ["--eval"],
        ["--slim_config", "slim.yml"],
        ["--use_tensorboard", "True"],
        ["--use_wandb", "True"],
        ["--save_prediction_only"],
        ["--profiler_options", "batch_range=[1,2]"],
        ["--save_proposals"],
    ],
)
def test_parse_args_rejects_unimplemented_paddle_options(
    unsupported_args,
    capsys,
):
    with pytest.raises(SystemExit) as error:
        train_cli.parse_args(["--config", "model.yml", *unsupported_args])

    assert error.value.code == 2
    error_output = capsys.readouterr().err
    assert "error:" in error_output
    assert unsupported_args[0] in error_output


def test_main_accepts_argv_and_selects_cpu(
    isolated_workspace,
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "train.yml"
    config_path.write_text(
        "architecture: ContractModel\nnum_classes: 80\nuse_gpu: false\n",
        encoding="utf-8",
    )
    observed = []

    class FakeTrainer:
        def __init__(self, cfg, mode):
            observed.append((cfg.copy(), mode))

        def train(self):
            observed.append("train")

    monkeypatch.setattr(train_cli, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_cli, "init_parallel_env", lambda: None)
    monkeypatch.setattr(train_cli.torch.cuda, "is_available", lambda: False)

    exit_code = train_cli.main(["--config", str(config_path), "--seed", "5"])

    assert exit_code == 0
    assert observed[0][1] == "train"
    assert observed[0][0]["device"] == train_cli.torch.device("cpu")
    assert observed[0][0]["seed"] == 5
    assert observed[1] == "train"


def test_main_rejects_requested_cuda_before_querying_device_name(
    isolated_workspace,
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "train.yml"
    config_path.write_text(
        "architecture: ContractModel\nnum_classes: 80\nuse_gpu: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train_cli.torch.cuda,
        "get_device_name",
        lambda unused: pytest.fail("device name queried before CUDA validation"),
    )

    with pytest.raises(SystemExit) as error:
        train_cli.main(["--config", str(config_path)])

    assert error.value.code == 1


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
        train_cli, "set_random_seed", lambda seed: calls.append(("seed", seed))
    )
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


def test_run_rejects_unimplemented_semi_supervised_weights(monkeypatch):
    monkeypatch.setattr(train_cli, "init_parallel_env", lambda: None)
    monkeypatch.setattr(
        train_cli,
        "Trainer",
        lambda *args, **kwargs: pytest.fail("unsupported config constructed a trainer"),
    )
    flags = SimpleNamespace(
        ddp=False,
        enable_ce=False,
        seed=None,
        resume=None,
    )
    config = AttrDict(
        pretrain_teacher_weights="teacher.pth",
        pretrain_student_weights="student.pth",
    )

    with pytest.raises(NotImplementedError, match="teacher/student"):
        train_cli.run(flags, config)
