import pytest
import torch
from torch import nn

from detrs.cli import eval as eval_cli


class _AuxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor_points", torch.ones(1, 2))
        self.register_buffer("stride_tensor", torch.ones(1, 1))


class _EvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.aux_o2m_head = _AuxHead()

    def forward(self, batch):
        batch_size = batch["image"].shape[0]
        return {
            "bbox": torch.zeros((batch_size, 6)),
            "bbox_num": torch.ones(batch_size, dtype=torch.int32),
        }


def test_load_evaluation_weights_allows_only_derived_buffers(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save({"weight": torch.tensor(3.0)}, checkpoint_path)
    model = _EvalModel()

    eval_cli.load_evaluation_weights(model, checkpoint_path)

    assert model.weight.item() == pytest.approx(3.0)


def test_load_evaluation_weights_rejects_unknown_missing_key(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save({}, checkpoint_path)

    with pytest.raises(RuntimeError, match="missing=.*weight"):
        eval_cli.load_evaluation_weights(_EvalModel(), checkpoint_path)


def test_load_evaluation_weights_selects_exponential_ema(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save(
        {
            "model": {"weight": torch.tensor(2.0)},
            "ema": {
                "ema_state_dict": {"weight": torch.tensor(4.0)},
                "step": 3,
                "ema_decay_type": "exponential",
            },
        },
        checkpoint_path,
    )
    model = _EvalModel()

    eval_cli.load_evaluation_weights(model, checkpoint_path, use_ema=True)

    assert model.weight.item() == pytest.approx(4.0)


def test_get_ema_state_dict_corrects_non_exponential_weights():
    checkpoint = {
        "ema": {
            "ema_state_dict": {
                "weight": torch.tensor(0.75),
                "frozen": torch.tensor(7.0),
            },
            "step": 2,
            "current_decay": 0.5,
            "ema_decay_type": "normal",
            "ema_black_list": ["frozen"],
        }
    }

    state_dict = eval_cli._get_ema_state_dict(checkpoint)

    assert state_dict["weight"].item() == pytest.approx(1.0)
    assert state_dict["frozen"].item() == pytest.approx(7.0)


def test_load_evaluation_weights_selects_upstream_ema_module(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save(
        {
            "model": {"weight": torch.tensor(2.0)},
            "ema": {"module": {"weight": torch.tensor(6.0)}, "updates": 12},
        },
        checkpoint_path,
    )
    model = _EvalModel()

    eval_cli.load_evaluation_weights(model, checkpoint_path, use_ema=True)

    assert model.weight.item() == pytest.approx(6.0)


@pytest.mark.parametrize("module", [None, {}, {"weight": "not-a-tensor"}])
def test_rejects_malformed_upstream_ema_module(module):
    with pytest.raises(RuntimeError, match="EMA module must be a tensor state dict"):
        eval_cli._get_ema_state_dict({"ema": {"module": module, "updates": 1}})


def test_load_evaluation_weights_requires_requested_ema(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save({"model": {"weight": torch.tensor(2.0)}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="does not contain EMA"):
        eval_cli.load_evaluation_weights(_EvalModel(), checkpoint_path, use_ema=True)


def test_evaluate_uses_batch_dictionary_and_current_model_output():
    events = []

    class Metric:
        def update(self, batch, outputs):
            events.append((batch, outputs))

        def accumulate(self):
            events.append("accumulate")

        def get_results(self):
            return {"bbox": torch.arange(12).numpy()}

    batch = {"image": torch.ones((2, 3, 4, 4))}
    model = _EvalModel()
    metric = Metric()

    results = eval_cli.evaluate(
        model,
        [batch],
        metric,
        lambda value: value,
        torch.device("cpu"),
    )

    assert events[0][0] is batch
    assert set(events[0][1]) == {"bbox", "bbox_num"}
    assert events[1] == "accumulate"
    assert results["bbox"].shape == (12,)


def test_configure_dataset_preserves_non_overridden_path(tmp_path):
    dataset_root = tmp_path / "coco"
    cfg = type(
        "Config",
        (),
        {
            "EvalDataset": {
                "dataset_dir": str(dataset_root),
                "anno_path": "annotations/instances_val2017.json",
                "image_dir": "val2017",
            }
        },
    )()

    eval_cli._configure_dataset(cfg, anno_file=tmp_path / "subset.json")

    assert cfg.EvalDataset["dataset_dir"] == "."
    assert cfg.EvalDataset["anno_path"] == str((tmp_path / "subset.json").resolve())
    assert cfg.EvalDataset["image_dir"] == str((dataset_root / "val2017").resolve())


def test_format_results_names_standard_coco_statistics():
    results = eval_cli._format_results({"bbox": torch.arange(12).numpy()})

    assert results["bbox"]["AP"] == pytest.approx(0.0)
    assert results["bbox"]["AP50"] == pytest.approx(1.0)
    assert results["bbox"]["ARl"] == pytest.approx(11.0)


def test_parse_args_accepts_persistent_output_directory(tmp_path):
    args = eval_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert args.output_dir == str(tmp_path)


def test_parse_args_accepts_hyphen_and_underscore_aliases():
    args = eval_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--anno_file",
            "instances.json",
            "--image-dir",
            "images",
            "--batch_size",
            "8",
            "--num-workers",
            "0",
            "--use_ema",
        ]
    )

    assert args.anno_file == "instances.json"
    assert args.image_dir == "images"
    assert args.batch_size == 8
    assert args.num_workers == 0
    assert args.use_ema is True


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--batch-size", "0"], "--batch-size"),
        (["--num-workers", "-1"], "--num-workers"),
    ],
)
def test_parse_args_rejects_invalid_loader_values(extra_args, message, capsys):
    with pytest.raises(SystemExit) as error:
        eval_cli.parse_args(
            [
                "--config",
                "model.yml",
                "--checkpoint",
                "model.pth",
                *extra_args,
            ]
        )

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_main_accepts_argv_and_wires_current_eval_contract(
    isolated_workspace,
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "eval.yml"
    config_path.write_text(
        "architecture: ContractModel\n"
        "metric: COCO\n"
        "num_classes: 80\n"
        "EvalReader:\n  batch_size: 4\n"
        "EvalDataset:\n"
        "  dataset_dir: .\n"
        "  anno_path: instances.json\n"
        "  image_dir: images\n",
        encoding="utf-8",
    )
    output_directory = tmp_path / "results"
    observed = []

    class FakeDataset:
        dataset_dir = "."
        anno_path = "instances.json"

    class FakeTrainer:
        def __init__(self, cfg, mode):
            observed.append(("trainer", cfg.copy(), mode))
            self.model = object()
            self.dataset = FakeDataset()
            self.loader = []
            self._prepare_batch = lambda value: value

    monkeypatch.setattr(eval_cli, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        eval_cli,
        "load_evaluation_weights",
        lambda model, checkpoint, use_ema: observed.append(
            ("weights", model, checkpoint, use_ema)
        ),
    )
    monkeypatch.setattr(
        eval_cli,
        "COCOMetric",
        lambda annotation, output_eval: ("metric", annotation, output_eval),
    )
    monkeypatch.setattr(
        eval_cli,
        "evaluate",
        lambda model, loader, metric, prepare, device: {
            "bbox": torch.arange(12).numpy()
        },
    )

    exit_code = eval_cli.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            "model.pth",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--output-dir",
            str(output_directory),
            "--use-ema",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert observed[0][0] == "trainer"
    assert observed[0][1]["EvalReader"]["batch_size"] == 2
    assert observed[0][1]["worker_num"] == 0
    assert observed[0][1]["device"] == torch.device("cpu")
    assert observed[1][0] == "weights"
    assert observed[1][2:] == ("model.pth", True)
