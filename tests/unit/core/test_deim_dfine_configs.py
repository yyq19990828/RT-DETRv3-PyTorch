from pathlib import Path

import pytest
from torch import nn

from ppdet_pytorch import (
    data,  # noqa: F401
    engine,  # noqa: F401
)
from ppdet_pytorch import optimizer as optimizer_module  # noqa: F401
from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.modeling import DEIM
from ppdet_pytorch.modeling.losses import DEIMCriterion

CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "deim" / "dfine"

VARIANTS = {
    "n": {
        "epoch": 160,
        "flat": 7800,
        "no_aug": 12,
        "gamma": 1.0,
        "policy": [4, 78, 148],
        "batch": 128,
        "backbone_lr": 0.0004,
        "parameters": 3_782_693,
    },
    "s": {
        "epoch": 132,
        "flat": 64,
        "no_aug": 12,
        "gamma": 0.5,
        "policy": [4, 64, 120],
        "batch": 32,
        "backbone_lr": 0.0002,
        "parameters": 10_321_877,
    },
    "m": {
        "epoch": 102,
        "flat": 49,
        "no_aug": 12,
        "gamma": 0.5,
        "policy": [4, 49, 90],
        "batch": 32,
        "backbone_lr": 0.00004,
        "parameters": 19_590_064,
    },
    "l": {
        "epoch": 58,
        "flat": 29,
        "no_aug": 8,
        "gamma": 0.5,
        "policy": [4, 29, 50],
        "batch": 32,
        "backbone_lr": 0.000025,
        "parameters": 31_289_848,
    },
    "x": {
        "epoch": 58,
        "flat": 29,
        "no_aug": 8,
        "gamma": 0.5,
        "policy": [4, 29, 50],
        "batch": 32,
        "backbone_lr": 0.000005,
        "parameters": 62_715_480,
    },
}


def _path(variant):
    return CONFIG_ROOT / f"deim_hgnetv2_{variant}_coco.yml"


@pytest.mark.parametrize("variant", VARIANTS)
def test_deim_dfine_variant_builds_official_training_contract(
    variant, isolated_workspace
):
    expected = VARIANTS[variant]
    config = load_config(_path(variant))
    model = create(config.architecture)
    reader = create("TrainReader")
    protocol = create(config.TrainingProtocol)

    assert isinstance(model, DEIM)
    assert isinstance(model.criterion, DEIMCriterion)
    assert model.criterion.losses == ["mal", "boxes", "local"]
    assert model.criterion.gamma == 1.5
    assert model.criterion.use_uni_set is True
    assert isinstance(model.decoder.decoder.layers[0].activation, nn.SiLU)
    assert isinstance(model.decoder.query_pos_head.act, nn.SiLU)
    assert model.backbone.training is True
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected["parameters"]
    )
    assert config.epoch == expected["epoch"]
    assert config.amp is True
    assert config.use_ema is True
    assert protocol.family == "deim"
    assert protocol.stop_epoch == expected["policy"][-1]
    assert reader.total_batch_size == expected["batch"]
    assert reader.dense_o2o_policy["policy_epochs"] == expected["policy"]
    assert reader.dense_o2o_policy["mixup_epochs"] == expected["policy"][:2]
    assert reader.dense_o2o_policy["multiscale_stop_epoch"] == expected["policy"][-1]
    assert not any(
        type(transform).__name__ == "BatchRandomResize"
        for transform in reader._batch_transforms.transforms_cls
    )
    assert type(reader._sample_transforms.transforms_cls[1]).__name__ == (
        "DEIMDenseO2OMosaic"
    )
    assert type(reader._batch_transforms.transforms_cls[0]).__name__ == (
        "DEIMDenseO2OCollate"
    )

    scheduler = config.LearningRate["schedulers"][0]
    assert scheduler["name"] == "FlatCosineLRScheduler"
    assert scheduler["total_epochs"] == expected["epoch"]
    assert scheduler["warmup_iter"] == 2000
    assert scheduler["flat_epochs"] == expected["flat"]
    assert scheduler["no_aug_epochs"] == expected["no_aug"]
    assert scheduler["lr_gamma"] == expected["gamma"]

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


def test_deim_dfine_configs_repeat_without_registry_leakage(isolated_workspace):
    for variant in VARIANTS:
        observed = []
        for _ in range(2):
            config = load_config(_path(variant))
            model = create(config.architecture)
            observed.append(
                (
                    type(model).__name__,
                    tuple(model.criterion.losses),
                    type(model.decoder.query_pos_head.act).__name__,
                )
            )
        assert observed == [("DEIM", ("mal", "boxes", "local"), "SiLU")] * 2


def test_rejects_vfl_substitution_before_training(isolated_workspace):
    config = load_config(_path("n"))
    config.DEIMCriterion["losses"] = ["vfl", "boxes", "local"]

    with pytest.raises(ValueError, match="unsupported DEIM losses"):
        create(config.architecture)


def test_rejects_augmentation_after_stop(isolated_workspace):
    config = load_config(_path("s"))
    config.TrainReader["dense_o2o_policy"]["multiscale_stop_epoch"] = 119

    with pytest.raises(ValueError, match="multiscale_stop_epoch"):
        create("TrainReader")
