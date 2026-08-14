from pathlib import Path

import pytest
from torch import nn

from detrs import data, engine  # noqa: F401
from detrs import optimizer as optimizer_module  # noqa: F401
from detrs.core.workspace import create, load_config
from detrs.modeling import RTDETRV4
from detrs.modeling.losses import RTDETRV4Criterion

CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "rtdetrv4"

VARIANTS = {
    "s": {
        "backbone": "B0",
        "parameters": 10_519_253,
        "epoch": 132,
        "policy": [4, 64, 120],
        "weight": 5,
        "rho": 11,
        "delta": 1,
        "default": 20,
    },
    "m": {
        "backbone": "B2",
        "parameters": 19_787_440,
        "epoch": 102,
        "policy": [4, 49, 90],
        "weight": 5,
        "rho": 3.5,
        "delta": 0.25,
        "default": 15,
    },
    "l": {
        "backbone": "B4",
        "parameters": 31_487_224,
        "epoch": 58,
        "policy": [4, 29, 50],
        "weight": 15,
        "rho": 2,
        "delta": 0.1,
        "default": 15,
    },
    "x": {
        "backbone": "B5",
        "parameters": 63_011_160,
        "epoch": 58,
        "policy": [4, 29, 50],
        "weight": 20,
        "rho": 2,
        "delta": 0.25,
        "default": 20,
    },
}


def _path(variant):
    return CONFIG_ROOT / f"rtdetrv4_hgnetv2_{variant}_coco.yml"


@pytest.mark.parametrize("variant", VARIANTS)
def test_rtdetrv4_variant_builds_official_training_contract(
    variant, isolated_workspace
):
    expected = VARIANTS[variant]
    config = load_config(_path(variant))
    model = create(config.architecture)
    protocol = create(config.TrainingProtocol)
    reader = create("TrainReader")

    assert isinstance(model, RTDETRV4)
    assert isinstance(model.criterion, RTDETRV4Criterion)
    assert model.criterion.losses == ["mal", "boxes", "local"]
    assert model.criterion.weight_dict["loss_distill"] == expected["weight"]
    assert model.criterion.distill_adaptive_params == {
        "enabled": True,
        "rho": expected["rho"],
        "delta": expected["delta"],
        "default_weight": expected["default"],
    }
    assert model.encoder.feature_projector is not None
    assert model.encoder.feature_projector[0].out_features == 768
    assert isinstance(model.decoder.decoder.layers[0].activation, nn.SiLU)
    assert model.backbone.name == expected["backbone"]
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected["parameters"]
    )
    assert config.epoch == expected["epoch"]
    assert protocol.family == "rtdetrv4"
    assert protocol.stop_epoch == expected["policy"][-1]
    assert protocol.current_gam_weight == expected["weight"]
    assert protocol.gam_rho == expected["rho"]
    assert protocol.gam_delta == expected["delta"]
    assert protocol.gam_default_weight == expected["default"]
    assert reader.dense_o2o_policy["policy_epochs"] == expected["policy"]

    teacher = config.teacher_model
    assert teacher["weights_filename"] == (
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    assert teacher["weights_size_bytes"] == 342_860_279
    assert teacher["weights_sha256"] == (
        "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
    )


def test_gam_config_matches_criterion_adaptive_parameters(isolated_workspace):
    for variant in VARIANTS:
        config = load_config(_path(variant))
        adaptive = config.RTDETRV4Criterion["distill_adaptive_params"]
        protocol = config.TrainingProtocol
        assert protocol["gam_rho"] == adaptive["rho"]
        assert protocol["gam_delta"] == adaptive["delta"]
        assert protocol["gam_default_weight"] == adaptive["default_weight"]
        assert (
            protocol["current_gam_weight"]
            == config.RTDETRV4Criterion["weight_dict"]["loss_distill"]
        )


def test_student_eval_build_does_not_access_teacher_assets(isolated_workspace):
    config = load_config(_path("s"))
    config.teacher_model["dinov3_repo_path"] = "/definitely/missing"
    config.teacher_model["dinov3_weights_path"] = "/definitely/missing.pth"

    model = create(config.architecture).eval()

    assert isinstance(model, RTDETRV4)
    assert not any("teacher" in name for name in model.state_dict())


def test_deploy_removes_training_only_feature_projector(isolated_workspace):
    config = load_config(_path("s"))
    model = create(config.architecture).eval()
    assert any("feature_projector" in name for name in model.state_dict())

    assert model.deploy() is model

    assert model.encoder.feature_projector is None
    assert not any("feature_projector" in name for name in model.state_dict())
