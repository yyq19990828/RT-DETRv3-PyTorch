import pytest
import torch
from torch import nn

from ppdet_pytorch.cli import eval as eval_cli


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


def test_load_evaluation_weights_requires_requested_ema(tmp_path):
    checkpoint_path = tmp_path / "weights.pth"
    torch.save({"model": {"weight": torch.tensor(2.0)}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="does not contain EMA"):
        eval_cli.load_evaluation_weights(
            _EvalModel(), checkpoint_path, use_ema=True)


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
    assert cfg.EvalDataset["anno_path"] == str(
        (tmp_path / "subset.json").resolve()
    )
    assert cfg.EvalDataset["image_dir"] == str(
        (dataset_root / "val2017").resolve()
    )


def test_format_results_names_standard_coco_statistics():
    results = eval_cli._format_results({"bbox": torch.arange(12).numpy()})

    assert results["bbox"]["AP"] == pytest.approx(0.0)
    assert results["bbox"]["AP50"] == pytest.approx(1.0)
    assert results["bbox"]["ARl"] == pytest.approx(11.0)


def test_parse_args_accepts_persistent_output_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "rtdetrv3-eval",
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--output-dir",
            str(tmp_path),
        ],
    )

    args = eval_cli.parse_args()

    assert args.output_dir == str(tmp_path)
