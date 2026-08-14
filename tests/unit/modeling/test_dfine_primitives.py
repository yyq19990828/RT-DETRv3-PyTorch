import copy

import pytest
import torch
import torch.nn as nn

from ppdet_pytorch.modeling.transformers.dfine_decoder import (
    LQE,
    Integral,
    TransformerDecoder,
)
from ppdet_pytorch.modeling.transformers.dfine_hybrid_encoder import DFINEHybridEncoder
from ppdet_pytorch.modeling.transformers.dfine_utils import (
    bbox2distance,
    distance2bbox,
    weighting_function,
)


class _DecoderLayer(nn.Module):
    def forward(
        self, target, reference_points, value, spatial_shapes, attn_mask, query_pos
    ):
        return target + query_pos * 0.01


class _Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, value):
        return self.linear(value)


def _decoder_inputs(num_layers=3, reg_max=8):
    torch.manual_seed(0)
    decoder = TransformerDecoder(
        hidden_dim=8,
        decoder_layer=_DecoderLayer(),
        decoder_layer_wide=_DecoderLayer(),
        num_layers=num_layers,
        num_head=2,
        reg_max=reg_max,
        reg_scale=torch.tensor([4.0]),
        up=torch.tensor([0.5]),
        eval_idx=1,
    )
    args = (
        torch.randn(2, 5, 8),
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 8),
        [[1, 2], [1, 3]],
        nn.ModuleList([_Head(8, 4 * (reg_max + 1)) for _ in range(num_layers)]),
        nn.ModuleList([_Head(8, 6) for _ in range(num_layers)]),
        _Head(4, 8),
        _Head(8, 4),
        Integral(reg_max),
        torch.tensor([0.5]),
        torch.tensor([4.0]),
    )
    return decoder, args


def test_utility_weighting_and_bbox_roundtrip():
    project = weighting_function(8, torch.tensor([0.5]), torch.tensor([4.0]))
    assert project.shape == (9,)
    assert torch.equal(project, -project.flip(0))

    points = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    distance = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(distance2bbox(points, distance, 4.0), points)

    xyxy = torch.tensor([[0.4, 0.3, 0.6, 0.7]])
    indices, right, left = bbox2distance(points, xyxy, 8, 4.0, torch.tensor([0.5]))
    assert indices.shape == right.shape == left.shape == (4,)
    assert torch.allclose(right + left, torch.ones(4))


def test_integral_and_lqe_outputs():
    integral = Integral(8)
    logits = torch.randn(2, 3, 36)
    project = weighting_function(8, torch.tensor([0.5]), torch.tensor([4.0]))
    assert integral(logits, project).shape == (2, 3, 4)

    lqe = LQE(4, 16, 2, 8)
    scores = torch.randn(2, 3, 5)
    assert lqe(scores, logits).shape == scores.shape


def test_decoder_train_eval_each_aux_layer_and_deploy():
    decoder, args = _decoder_inputs()
    decoder.train()
    train_outputs = decoder(*args)
    assert train_outputs[0].shape == (3, 2, 5, 4)
    assert train_outputs[1].shape == (3, 2, 5, 6)
    assert train_outputs[2].shape == (3, 2, 5, 36)

    eval_decoder = copy.deepcopy(decoder).eval()
    eval_outputs = eval_decoder(*args)
    assert eval_outputs[0].shape[0] == 1

    deploy_decoder = copy.deepcopy(eval_decoder)
    deploy_decoder.convert_to_deploy()
    deploy_outputs = deploy_decoder(*args)
    assert torch.allclose(eval_outputs[0], deploy_outputs[0], rtol=1e-5, atol=1e-6)
    assert torch.allclose(eval_outputs[1], deploy_outputs[1], rtol=1e-5, atol=1e-6)


def test_encoder_returns_list_and_optional_training_projected_f5_tuple():
    feats = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    encoder = DFINEHybridEncoder(
        in_channels=[4, 8],
        feat_strides=[8, 16],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
        use_encoder_idx=[1],
        num_encoder_layers=1,
        expansion=0.5,
        depth_mult=0.34,
    )
    encoder.eval()
    outputs = encoder(feats)
    assert isinstance(outputs, list) and len(outputs) == 2

    distill_encoder = DFINEHybridEncoder(
        in_channels=[4, 8],
        feat_strides=[8, 16],
        hidden_dim=8,
        nhead=2,
        dim_feedforward=16,
        use_encoder_idx=[1],
        num_encoder_layers=1,
        expansion=0.5,
        depth_mult=0.34,
        distill_teacher_dim=12,
        project_f5=True,
    )
    distill_encoder.train()
    family_outputs, projected_f5 = distill_encoder(feats)
    assert len(family_outputs) == 2
    assert projected_f5.shape == (2, 12, 4, 4)
    distill_encoder.eval()
    assert isinstance(distill_encoder(feats), list)


def test_rejects_invalid_reg_max():
    for reg_max in (1, 2, 7, 9):
        with pytest.raises(ValueError, match="reg_max"):
            Integral(reg_max)


def test_rejects_level_count():
    with pytest.raises(ValueError, match="level"):
        DFINEHybridEncoder(in_channels=[4, 8], feat_strides=[8])
    encoder = DFINEHybridEncoder(
        in_channels=[4, 8],
        feat_strides=[8, 16],
        hidden_dim=8,
        nhead=2,
        use_encoder_idx=[1],
    )
    with pytest.raises(ValueError, match="feature levels"):
        encoder([torch.randn(1, 4, 8, 8)])


def test_rejects_zero_teacher_dimension():
    with pytest.raises(ValueError, match="distill_teacher_dim"):
        DFINEHybridEncoder(distill_teacher_dim=0, project_f5=True)


def test_rejects_teacher_dimension_when_projection_is_disabled():
    with pytest.raises(ValueError, match="project_f5 must be enabled"):
        DFINEHybridEncoder(distill_teacher_dim=12, project_f5=False)
