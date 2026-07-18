import pytest
import torch
from torch import nn

from ppdet_pytorch.modeling.backbones.resnet import ConvNormLayer, ResNet
from ppdet_pytorch.optimizer.optimizer import (
    LearningRate,
    LinearWarmup,
    OptimizerBuilder,
    PiecewiseDecay,
)


class _ModelWithStageMultiplier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)
        self.stage = ConvNormLayer(
            ch_in=4,
            ch_out=4,
            filter_size=3,
            stride=1,
            freeze_norm=False,
            lr=0.1,
        )


class _TwoParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.0))
        self.stage = nn.Parameter(torch.tensor(1.0))
        self.stage._optimizer_lr_multiplier = 0.1


def _group_lr_by_parameter(optimizer):
    return {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_optimizer_builder_applies_model_lr_multipliers_once():
    model = _ModelWithStageMultiplier()
    optimizer = OptimizerBuilder(
        regularizer=False,
        optimizer={"type": "AdamW", "weight_decay": 0.0001},
    )(0.0004, model)

    group_lrs = _group_lr_by_parameter(optimizer)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert len(group_lrs) == len(trainable_parameters)
    assert optimizer.param_groups[0]["lr_multiplier"] == 1.0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0004)
    assert optimizer.param_groups[1]["lr_multiplier"] == 0.1
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.00004)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0001}
    assert all(
        group_lrs[id(parameter)] == pytest.approx(0.0004)
        for parameter in model.projection.parameters()
    )
    assert all(
        group_lrs[id(parameter)] == pytest.approx(0.00004)
        for parameter in model.stage.parameters()
    )


def test_resnet_stages_retain_configured_lr_multiplier():
    backbone = ResNet(
        depth=18,
        variant="d",
        lr_mult_list=[0.1, 0.2, 0.3, 0.4],
        freeze_norm=False,
        freeze_at=-1,
    )

    assert {
        getattr(parameter, "_optimizer_lr_multiplier", None)
        for parameter in backbone.conv1.parameters()
    } == {1.0}
    for stage, multiplier in zip(
        backbone.res_layers,
        [0.1, 0.2, 0.3, 0.4],
    ):
        assert {
            getattr(parameter, "_optimizer_lr_multiplier", None)
            for parameter in stage.parameters()
        } == {multiplier}


def test_lr_multiplier_changes_actual_sgd_update_by_declared_ratio():
    model = _TwoParameterModel()
    optimizer = OptimizerBuilder(
        regularizer=False,
        optimizer={"type": "SGD"},
    )(0.2, model)
    model.base.grad = torch.ones_like(model.base)
    model.stage.grad = torch.ones_like(model.stage)

    optimizer.step()

    assert 1.0 - model.base.item() == pytest.approx(0.2)
    assert 1.0 - model.stage.item() == pytest.approx(0.02)


@pytest.mark.filterwarnings(
    "ignore:The epoch parameter in `scheduler.step\\(\\)` was not necessary"
)
def test_lr_scheduler_uses_global_piecewise_steps_and_preserves_group_ratio():
    model = _ModelWithStageMultiplier()
    optimizer = OptimizerBuilder(
        regularizer=False,
        optimizer={"type": "AdamW", "weight_decay": 0.0001},
    )(0.0004, model)
    scheduler = LearningRate(
        base_lr=0.0004,
        schedulers=[
            PiecewiseDecay(gamma=[0.1], milestones=[1], use_warmup=True),
            LinearWarmup(steps=2, start_factor=0.5),
        ],
    )(step_per_epoch=4, optimizer=optimizer)

    base_lr_trace = []
    for _ in range(7):
        base_lr, stage_lr = [group["lr"] for group in optimizer.param_groups]
        base_lr_trace.append(base_lr)
        assert stage_lr / base_lr == pytest.approx(0.1)
        optimizer.step()
        scheduler.step()

    assert base_lr_trace == pytest.approx(
        [0.0002, 0.0003, 0.0004, 0.0004, 0.00004, 0.00004, 0.00004]
    )
