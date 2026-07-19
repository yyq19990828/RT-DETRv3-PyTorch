"""Regression tests for layout-stable BatchNorm backward."""

import torch
import torch.nn as nn

from ppdet_pytorch.modeling.backbones.resnet import ConvNormLayer
from ppdet_pytorch.modeling.batch_norm import ContiguousGradBatchNorm2d


def test_batch_norm_backward_matches_contiguous_reference():
    aligned = ContiguousGradBatchNorm2d(4, track_running_stats=False)
    reference = nn.BatchNorm2d(4, track_running_stats=False)
    reference.load_state_dict(aligned.state_dict())
    aligned.train()
    reference.train()

    generator = torch.Generator().manual_seed(2026)
    input_value = torch.randn(1, 4, 5, 5, generator=generator)
    aligned_input = input_value.clone().requires_grad_(True)
    reference_input = input_value.clone().requires_grad_(True)
    aligned_output = aligned(aligned_input)
    reference_output = reference(reference_input)

    gradient_nlc = torch.zeros(1, 25, 4)
    gradient_nlc[:, ::3] = torch.randn(1, 9, 4, generator=generator)
    gradient_nchw = gradient_nlc.transpose(1, 2).reshape(1, 4, 5, 5)
    assert not gradient_nchw.is_contiguous()

    aligned_output.backward(gradient_nchw)
    reference_output.backward(gradient_nchw.contiguous())

    torch.testing.assert_close(aligned_input.grad, reference_input.grad)
    assert set(aligned.state_dict()) == set(reference.state_dict())


def test_resnet_frozen_batch_norm_keeps_global_statistics_in_train_mode():
    layer = ConvNormLayer(3, 4, 3, stride=1, freeze_norm=True)
    input_value = torch.randn(2, 3, 5, 5)

    layer.eval()
    expected = layer(input_value)
    running_mean = layer.norm.running_mean.clone()
    running_var = layer.norm.running_var.clone()
    layer.train()
    actual = layer(input_value)

    assert layer.training
    assert not layer.norm.training
    assert all(not parameter.requires_grad for parameter in layer.norm.parameters())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.norm.running_mean, running_mean)
    torch.testing.assert_close(layer.norm.running_var, running_var)
