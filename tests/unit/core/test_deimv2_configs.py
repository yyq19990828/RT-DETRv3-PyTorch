from pathlib import Path

import pytest

from detrs import data, engine  # noqa: F401
from detrs import optimizer as optimizer_module  # noqa: F401
from detrs.core.workspace import create, load_config
from detrs.modeling import DEIMV2
from detrs.modeling.losses import DEIMv2Criterion
from detrs.modeling.transformers.deimv2_decoder import RMSNorm
from detrs.modeling.transformers.dfine_support import (
    DEIMv2HungarianMatcher,
)

CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "deimv2"

VARIANTS = {
    "dinov3_x": {
        "file": "deimv2_dinov3_x_coco.yml",
        "parameters": 51_211_736,
        "epoch": 72,
        "policy": [4, 29, 50],
        "protocol_stop": 50,
        "matcher_epoch": 45,
        "eval_size": [640, 640],
        "backbone": "DINOv3STAs",
        "encoder": "DFINEHybridEncoder",
        "num_layers": 6,
        "losses": ["mal", "boxes", "local"],
    },
    "dinov3_l": {
        "file": "deimv2_dinov3_l_coco.yml",
        "parameters": 32_551_214,
        "epoch": 68,
        "policy": [4, 34, 60],
        "protocol_stop": 60,
        "matcher_epoch": 50,
        "eval_size": [640, 640],
        "backbone": "DINOv3STAs",
        "encoder": "DFINEHybridEncoder",
        "num_layers": 4,
        "losses": ["mal", "boxes", "local"],
    },
    "dinov3_m": {
        "file": "deimv2_dinov3_m_coco.yml",
        "parameters": 18_357_750,
        "epoch": 102,
        "policy": [4, 49, 90],
        "protocol_stop": 90,
        "matcher_epoch": 80,
        "eval_size": [640, 640],
        "backbone": "DINOv3STAs",
        "encoder": "DFINEHybridEncoder",
        "num_layers": 4,
        "losses": ["mal", "boxes", "local"],
    },
    "dinov3_s": {
        "file": "deimv2_dinov3_s_coco.yml",
        "parameters": 9_779_838,
        "epoch": 132,
        "policy": [4, 64, 120],
        "protocol_stop": 120,
        "matcher_epoch": 100,
        "eval_size": [640, 640],
        "backbone": "DINOv3STAs",
        "encoder": "DFINEHybridEncoder",
        "num_layers": 4,
        "losses": ["mal", "boxes", "local"],
    },
    "hgnetv2_n": {
        "file": "deimv2_hgnetv2_n_coco.yml",
        "parameters": 3_600_421,
        "epoch": 160,
        "policy": [4, 78, 148],
        "protocol_stop": 148,
        "matcher_epoch": 136,
        "eval_size": [640, 640],
        "backbone": "HGNetv2",
        "encoder": "DFINEHybridEncoder",
        "num_layers": 3,
        "losses": ["mal", "boxes", "local"],
    },
    "hgnetv2_pico": {
        "file": "deimv2_hgnetv2_pico_coco.yml",
        "parameters": 1_539_539,
        "epoch": 500,
        "policy": [4, 250, 400],
        "protocol_stop": 468,
        "matcher_epoch": 450,
        "eval_size": [640, 640],
        "backbone": "HGNetv2",
        "encoder": "LiteEncoder",
        "num_layers": 3,
        "losses": ["mal", "boxes"],
    },
    "hgnetv2_femto": {
        "file": "deimv2_hgnetv2_femto_coco.yml",
        "parameters": 983_945,
        "epoch": 500,
        "policy": [4, 250, 400],
        "protocol_stop": 468,
        "matcher_epoch": 450,
        "eval_size": [416, 416],
        "backbone": "HGNetv2",
        "encoder": "LiteEncoder",
        "num_layers": 3,
        "losses": ["mal", "boxes"],
    },
    "hgnetv2_atto": {
        "file": "deimv2_hgnetv2_atto_coco.yml",
        "parameters": 508_985,
        "epoch": 500,
        "policy": [4, 250, 400],
        "protocol_stop": 468,
        "matcher_epoch": 450,
        "eval_size": [320, 320],
        "backbone": "HGNetv2",
        "encoder": "LiteEncoder",
        "num_layers": 3,
        "losses": ["mal", "boxes"],
    },
}


def _path(variant):
    return CONFIG_ROOT / VARIANTS[variant]["file"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_deimv2_variant_builds_official_training_contract(variant, isolated_workspace):
    expected = VARIANTS[variant]
    config = load_config(_path(variant))
    model = create(config.architecture)
    protocol = create(config.TrainingProtocol)
    reader = create("TrainReader")

    assert isinstance(model, DEIMV2)
    assert isinstance(model.criterion, DEIMv2Criterion)
    assert model.criterion.losses == expected["losses"]
    assert model.criterion.gamma == 1.5
    assert model.criterion.use_uni_set is (
        expected["losses"] == ["mal", "boxes", "local"]
    )
    assert isinstance(model.criterion.matcher, DEIMv2HungarianMatcher)
    assert model.criterion.matcher.change_matcher is True
    assert model.criterion.matcher.iou_order_alpha == 4.0
    assert model.criterion.matcher.matcher_change_epoch == expected["matcher_epoch"]
    assert type(model.backbone).__name__ == expected["backbone"]
    assert type(model.encoder).__name__ == expected["encoder"]
    assert len(model.decoder.decoder.layers) == expected["num_layers"]
    assert isinstance(model.decoder.decoder.layers[0].norm1, RMSNorm)
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected["parameters"]
    )
    assert config.epoch == expected["epoch"]
    assert protocol.family == "deimv2"
    assert protocol.stop_epoch == expected["protocol_stop"]
    assert reader.dense_o2o_policy["policy_epochs"] == expected["policy"]
    assert config.eval_spatial_size == expected["eval_size"]
    assert config.DETRPostProcess["num_top_queries"] == 300


def test_deimv2_dinov3_branch_uses_imagenet_normalization(isolated_workspace):
    for variant in ("dinov3_x", "dinov3_l", "dinov3_m", "dinov3_s"):
        config = load_config(_path(variant))
        reader = config.TrainReader
        normalize = reader["batch_transforms"][0]["NormalizeImage"]
        assert normalize["mean"] == [0.485, 0.456, 0.406]
        assert normalize["std"] == [0.229, 0.224, 0.225]
        eval_normalize = config.EvalReader["sample_transforms"][2]["NormalizeImage"]
        assert eval_normalize["mean"] == [0.485, 0.456, 0.406]


def test_deimv2_hgnetv2_branch_keeps_unit_normalization(isolated_workspace):
    for variant in ("hgnetv2_n", "hgnetv2_pico", "hgnetv2_femto", "hgnetv2_atto"):
        config = load_config(_path(variant))
        normalize = config.TrainReader["batch_transforms"][0]["NormalizeImage"]
        assert normalize["mean"] == [0.0, 0.0, 0.0]
        assert normalize["std"] == [1.0, 1.0, 1.0]


def test_deimv2_sample_transforms_resize_to_variant_input_size(
    isolated_workspace,
):
    sizes = {
        "hgnetv2_femto": [416, 416],
        "hgnetv2_atto": [320, 320],
    }
    for variant, size in sizes.items():
        config = load_config(_path(variant))
        resize = config.TrainReader["sample_transforms"][-1]["Resize"]
        assert resize["target_size"] == size
    for variant in ("dinov3_s", "hgnetv2_n", "hgnetv2_pico"):
        config = load_config(_path(variant))
        resize = config.TrainReader["sample_transforms"][-1]["Resize"]
        assert resize["target_size"] == [640, 640]


def test_deimv2_deploy_prunes_eval_index_layers(isolated_workspace):
    config = load_config(_path("hgnetv2_atto"))
    model = create(config.architecture).eval()
    total_layers = len(model.decoder.decoder.layers)
    total_lqe = len(model.decoder.decoder.lqe_layers)
    assert model.decoder.eval_idx == total_layers - 1

    model.deploy()

    assert len(model.decoder.decoder.layers) == model.decoder.decoder.eval_idx + 1
    assert len(model.decoder.decoder.lqe_layers) == model.decoder.decoder.eval_idx + 1
    assert total_layers == 3
    assert total_lqe == 3
    assert not any("teacher" in name for name in model.state_dict())
