from pathlib import Path

import pytest
import torch

from ppdet_pytorch import data  # noqa: F401
from ppdet_pytorch import optimizer as optimizer_module  # noqa: F401
from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.modeling import DFINE

CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "dfine"

VARIANTS = {
    "n": {
        "backbone": "B0",
        "return_idx": (2, 3),
        "hidden": 128,
        "decoder_hidden": 128,
        "layers": 3,
        "levels": 2,
        "parameters": 3_782_693,
        "epoch": 160,
        "stop": 148,
        "batch": 128,
        "backbone_lr": 0.0004,
        "base_size_repeat": None,
    },
    "s": {
        "backbone": "B0",
        "return_idx": (1, 2, 3),
        "hidden": 256,
        "decoder_hidden": 256,
        "layers": 3,
        "levels": 3,
        "parameters": 10_321_877,
        "epoch": 132,
        "stop": 120,
        "batch": 32,
        "backbone_lr": 0.0001,
        "base_size_repeat": 20,
    },
    "m": {
        "backbone": "B2",
        "return_idx": (1, 2, 3),
        "hidden": 256,
        "decoder_hidden": 256,
        "layers": 4,
        "levels": 3,
        "parameters": 19_590_064,
        "epoch": 132,
        "stop": 120,
        "batch": 32,
        "backbone_lr": 0.00002,
        "base_size_repeat": 6,
    },
    "l": {
        "backbone": "B4",
        "return_idx": (1, 2, 3),
        "hidden": 256,
        "decoder_hidden": 256,
        "layers": 6,
        "levels": 3,
        "parameters": 31_244_152,
        "epoch": 80,
        "stop": 72,
        "batch": 32,
        "backbone_lr": 0.0000125,
        "base_size_repeat": 4,
    },
    "x": {
        "backbone": "B5",
        "return_idx": (1, 2, 3),
        "hidden": 384,
        "decoder_hidden": 256,
        "layers": 6,
        "levels": 3,
        "parameters": 62_621_560,
        "epoch": 80,
        "stop": 72,
        "batch": 32,
        "backbone_lr": 0.0000025,
        "base_size_repeat": 3,
    },
}


def _path(variant):
    return CONFIG_ROOT / f"dfine_hgnetv2_{variant}_coco.yml"


@pytest.mark.parametrize("variant", VARIANTS)
def test_dfine_variant_loads_and_builds_exact_graph(variant, isolated_workspace):
    expected = VARIANTS[variant]
    config = load_config(_path(variant))
    model = create(config.architecture)
    reader = create("TrainReader")

    assert isinstance(model, DFINE)
    assert model.backbone.name == expected["backbone"]
    assert model.backbone.return_idx == expected["return_idx"]
    assert model.encoder.hidden_dim == expected["hidden"]
    assert model.decoder.hidden_dim == expected["decoder_hidden"]
    assert model.decoder.num_layers == expected["layers"]
    assert model.decoder.num_levels == expected["levels"]
    assert model.decoder.feat_channels == [expected["hidden"]] * expected["levels"]
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected["parameters"]
    )
    assert model.post_process.num_top_queries == 300
    assert config.epoch == expected["epoch"]
    assert config.amp is True
    assert config.use_ema is True
    assert config.validate is True
    assert config.ema_warmups == 1000
    assert config.norm_type == "sync_bn"
    assert config.log_iter == 100
    assert config.snapshot_epoch == 12
    assert reader.total_batch_size == expected["batch"]
    assert config.EvalReader["sample_transforms"][1]["Resize"]["backend"] == "pil"
    assert config.TestReader["sample_transforms"][1]["Resize"]["backend"] == "pil"
    assert (
        reader._sample_transforms.ordinary_transform_policy["stop_epoch"]
        == expected["stop"]
    )
    protocol = create(config.TrainingProtocol)
    assert protocol.family == "dfine"
    assert protocol.stop_epoch == expected["stop"]
    assert protocol.ema_restart_decay == config.ema_restart_decay
    resize_sizes = reader._batch_transforms.transforms_cls[0].target_size
    if expected["base_size_repeat"] is None:
        assert resize_sizes == 640
    else:
        assert resize_sizes.count(640) == expected["base_size_repeat"]

    optimizer = create("OptimizerBuilder")(config.base_lr, model)
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert len(parameter_groups) == len(trainable)
    backbone_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    )
    assert parameter_groups[id(backbone_parameter)]["lr"] == expected["backbone_lr"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_dfine_variant_cpu_train_and_eval_forward(variant, isolated_workspace):
    config = load_config(_path(variant))
    model = create(config.architecture)
    model.decoder.num_queries = 4
    model.decoder.num_denoising = 0
    model.decoder.eval_spatial_size = None
    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "gt_class": [torch.tensor([[1]]), torch.empty(0, 1, dtype=torch.int64)],
        "gt_bbox": [torch.tensor([[0.5, 0.5, 0.2, 0.3]]), torch.empty(0, 4)],
        "im_shape": torch.full((2, 2), 64.0),
        "scale_factor": torch.ones(2, 2),
    }

    model.train()
    losses = model(batch)
    assert losses["loss"].ndim == 0
    assert all(torch.isfinite(value) for value in losses.values())
    model.eval()
    with torch.inference_mode():
        predictions = model(batch)
    assert predictions["bbox"].shape == (600, 6)
    assert predictions["bbox_num"].tolist() == [300, 300]
    assert torch.isfinite(predictions["bbox"]).all()


def test_dfine_configs_repeat_without_registry_leakage(isolated_workspace):
    for variant, expected in VARIANTS.items():
        observed = []
        for _ in range(2):
            config = load_config(_path(variant))
            model = create(config.architecture)
            observed.append(
                (
                    model.backbone.name,
                    model.encoder.hidden_dim,
                    model.decoder.hidden_dim,
                    model.decoder.num_levels,
                )
            )
        expected_graph = (
            expected["backbone"],
            expected["hidden"],
            expected["decoder_hidden"],
            expected["levels"],
        )
        assert observed == [expected_graph, expected_graph]


def test_rejects_misspelled_component_before_trainer(isolated_workspace):
    config = load_config(_path("n"))
    config.DFINE["decoder"] = "DFINTransformer"
    with pytest.raises(ValueError, match="not registered"):
        create(config.architecture)


def test_rejects_n_three_levels(isolated_workspace):
    config = load_config(_path("n"))
    config.DFINETransformer["feat_channels"] = [128, 128, 128]
    config.DFINETransformer["feat_strides"] = [8, 16, 32]
    config.DFINETransformer["num_levels"] = 3
    config.DFINETransformer["num_points"] = [6, 6, 6]
    with pytest.raises(ValueError, match="encoder output"):
        create(config.architecture)


def test_rejects_illegal_include(isolated_workspace, tmp_path):
    outside = tmp_path / "outside.yml"
    outside.write_text("architecture: DFINE\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "bad.yml"
    config.write_text("_BASE_: ['../outside.yml']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the config root"):
        load_config(config)


def test_absolute_external_config_allows_include_within_config_root(
    isolated_workspace, tmp_path
):
    config_root = tmp_path / "external" / "configs"
    family_dir = config_root / "family"
    family_dir.mkdir(parents=True)
    (config_root / "base.yml").write_text("epoch: 3\n", encoding="utf-8")
    config = family_dir / "model.yml"
    config.write_text("_BASE_: ['../base.yml']\n", encoding="utf-8")

    assert load_config(config).epoch == 3


def test_rejects_unknown_component_field(isolated_workspace):
    config = load_config(_path("n"))
    config.DFINETransformer["num_layres"] = 3
    with pytest.raises(ValueError, match="Extraneous param"):
        create(config.architecture)
