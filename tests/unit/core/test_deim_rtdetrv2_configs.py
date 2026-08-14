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

CONFIG_ROOT = Path(__file__).parents[3] / "configs" / "deim" / "rtdetrv2"

VARIANTS = {
    "r18vd_120e": (18, 3, 256, 0.5, 120, 64, 3, [4, 64, 117], 20_183_440),
    "r34vd_120e": (34, 4, 256, 0.5, 120, 64, 3, [4, 64, 117], 31_440_644),
    "r50vd_m_60e": (50, 3, 256, 0.5, 60, 34, 2, [4, 34, 58], 33_198_864),
    "r50vd_60e": (50, 6, 256, 1.0, 60, 34, 2, [4, 34, 58], 42_943_596),
    "r101vd_60e": (101, 6, 384, 1.0, 60, 34, 2, [4, 34, 58], 76_660_716),
}

PRETRAINED = {
    18: "ResNet18_vd_pretrained_from_paddle.pth",
    34: "ResNet34_vd_pretrained_from_paddle.pth",
    50: "ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


def _path(variant):
    return CONFIG_ROOT / f"deim_{variant}_coco.yml"


@pytest.mark.parametrize("variant", VARIANTS)
def test_deim_rtdetrv2_variant_builds_official_contract(variant, isolated_workspace):
    depth, layers, encoder_dim, expansion, epochs, flat, no_aug, policy, params = (
        VARIANTS[variant]
    )
    config = load_config(_path(variant))
    model = create(config.architecture)
    reader = create("TrainReader")
    protocol = create(config.TrainingProtocol)

    assert isinstance(model, DEIM)
    assert type(model.backbone).__name__ == "PResNet"
    assert type(model.encoder).__name__ == "RTDETRV2HybridEncoder"
    assert type(model.decoder).__name__ == "RTDETRTransformerv2"
    assert isinstance(model.criterion, DEIMCriterion)
    assert model.criterion.losses == ["mal", "boxes"]
    assert model.criterion.use_uni_set is False
    assert not {"loss_fgl", "loss_ddf"} & model.criterion.weight_dict.keys()
    assert model.backbone.out_channels[-1] == (512 if depth < 50 else 2048)
    assert len(model.backbone.res_layers) == 4
    assert model.encoder.hidden_dim == encoder_dim
    assert model.encoder.fpn_blocks[0].conv1.conv.out_channels == int(
        encoder_dim * expansion
    )
    assert model.decoder.hidden_dim == 256
    assert model.decoder.num_layers == layers
    assert model.post_process.use_focal_loss is True
    assert isinstance(model.decoder.decoder.layers[0].activation, nn.SiLU)
    assert isinstance(model.decoder.query_pos_head.act, nn.SiLU)
    assert sum(parameter.numel() for parameter in model.parameters()) == params
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())

    assert config.epoch == epochs
    assert Path(config.pretrain_weights).name == PRETRAINED[depth]
    assert protocol.family == "deim"
    assert protocol.stop_epoch == policy[-1]
    assert reader.total_batch_size == 16
    assert reader.dense_o2o_policy["policy_epochs"] == policy
    assert reader.dense_o2o_policy["mixup_epochs"] == policy[:2]
    assert reader.dense_o2o_policy["multiscale_stop_epoch"] == policy[-1]
    scheduler = config.LearningRate["schedulers"][0]
    assert scheduler["total_epochs"] == epochs
    assert scheduler["flat_epochs"] == flat
    assert scheduler["no_aug_epochs"] == no_aug

    optimizer = create("OptimizerBuilder")(config.base_lr, model)
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    backbone_norms = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and ("norm" in name or "bn" in name)
    ]
    assert backbone_norms
    assert all(
        parameter_groups[id(parameter)]["weight_decay"] == 0
        for parameter in backbone_norms
    )


def test_rejects_local_loss_before_training(isolated_workspace):
    config = load_config(_path("r18vd_120e"))
    config.DEIMCriterion["losses"].append("local")

    with pytest.raises(ValueError, match="does not support D-FINE local loss"):
        create(config.architecture)


def test_rejects_r50_variant_swap_on_strict_load(isolated_workspace):
    medium = create(load_config(_path("r50vd_m_60e")).architecture)
    large = create(load_config(_path("r50vd_60e")).architecture)

    with pytest.raises(RuntimeError, match="Missing key"):
        large.load_state_dict(medium.state_dict(), strict=True)
