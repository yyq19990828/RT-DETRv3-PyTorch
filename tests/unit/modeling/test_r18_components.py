from types import SimpleNamespace

import pytest
import torch

from ppdet_pytorch.modeling.architectures import rtdetrv3 as rtdetrv3_module
from ppdet_pytorch.modeling.architectures.rtdetrv3 import RTDETRV3
from ppdet_pytorch.modeling.backbones.resnet import ResNet
from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
from ppdet_pytorch.modeling.post_process import DETRPostProcess
from ppdet_pytorch.modeling.transformers.hybrid_encoder import (
    HybridEncoder,
    TransformerLayer,
)
from ppdet_pytorch.modeling.transformers.rtdetr_transformerv3 import (
    RTDETRTransformerv3,
)


def _small_hybrid_encoder():
    return HybridEncoder(
        in_channels=[8, 16, 32],
        feat_strides=[8, 16, 32],
        hidden_dim=8,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        encoder_layer=TransformerLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
        ),
        eval_size=[32, 32],
    )


class _PostProcessAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_process = DETRPostProcess(
            num_classes=2,
            num_top_queries=3,
            use_focal_loss=True,
        )

    def forward(self, boxes, logits, image_shape, scale_factor):
        predictions, counts, _ = self.post_process(
            (boxes, logits, None),
            image_shape,
            scale_factor,
        )
        return predictions, counts


def test_resnet18_vd_current_api_forward_and_backward():
    backbone = ResNet(
        depth=18,
        variant="d",
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
    )
    image = torch.randn(1, 3, 64, 64, requires_grad=True)

    features = backbone({"image": image})
    sum(feature.mean() for feature in features).backward()

    assert [tuple(feature.shape) for feature in features] == [
        (1, 128, 8, 8),
        (1, 256, 4, 4),
        (1, 512, 2, 2),
    ]
    assert [(shape.channels, shape.stride) for shape in backbone.out_shape] == [
        (128, 8),
        (256, 16),
        (512, 32),
    ]
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_hybrid_encoder_cached_position_embedding_moves_with_module():
    encoder = _small_hybrid_encoder()

    assert "pos_embed2" in dict(encoder.named_buffers())
    assert "pos_embed2" not in encoder.state_dict()

    encoder.to(dtype=torch.float64)
    assert encoder.pos_embed2.dtype == torch.float64


def test_transformer_cached_anchors_move_without_entering_state_dict():
    transformer = RTDETRTransformerv3(
        num_classes=2,
        hidden_dim=8,
        num_queries=4,
        backbone_feat_channels=[8, 16, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=2,
        nhead=2,
        num_decoder_layers=1,
        dim_feedforward=16,
        num_denoising=0,
        eval_size=[32, 32],
    )

    buffers = dict(transformer.named_buffers())
    assert {"anchors", "valid_mask"} <= buffers.keys()
    assert {"anchors", "valid_mask"}.isdisjoint(transformer.state_dict())

    transformer.to(dtype=torch.float64)
    assert transformer.anchors.dtype == torch.float64
    assert transformer.valid_mask.dtype == torch.bool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_encoder_cpu_trace_reloads_and_runs_on_cuda(tmp_path):
    encoder = _small_hybrid_encoder().eval()
    inputs = [
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 32, 1, 1),
    ]
    with torch.inference_mode():
        traced = torch.jit.trace(
            encoder,
            (inputs,),
            strict=False,
            check_trace=False,
        )
    artifact = tmp_path / "hybrid-encoder.pt"
    torch.jit.save(traced, str(artifact))

    cuda_encoder = torch.jit.load(str(artifact), map_location="cuda").eval()
    outputs = cuda_encoder([value.cuda() for value in inputs])

    assert all(output.device.type == "cuda" for output in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_post_process_cpu_trace_reloads_and_runs_on_cuda(tmp_path):
    adapter = _PostProcessAdapter().eval()
    inputs = (
        torch.rand(2, 4, 4),
        torch.randn(2, 4, 2),
        torch.full((2, 2), 32.0),
        torch.ones(2, 2),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(adapter, inputs)
    artifact = tmp_path / "post-process.pt"
    torch.jit.save(traced, str(artifact))

    cuda_adapter = torch.jit.load(str(artifact), map_location="cuda").eval()
    predictions, counts = cuda_adapter(*(value.cuda() for value in inputs))

    assert predictions.device.type == "cuda"
    assert counts.device.type == "cuda"


def test_detr_post_process_current_api_decodes_top_queries():
    post_process = DETRPostProcess(
        num_classes=2,
        num_top_queries=3,
        use_focal_loss=True,
    )
    boxes = torch.tensor([[[0.50, 0.50, 0.40, 0.20], [0.25, 0.25, 0.20, 0.40]]])
    logits = torch.tensor([[[4.0, -4.0], [1.0, 3.0]]])
    image_shape = torch.tensor([[100.0, 200.0]])
    scale_factor = torch.ones(1, 2)

    predictions, counts, masks = post_process(
        (boxes, logits, None),
        image_shape,
        scale_factor,
    )

    assert predictions.shape == (3, 6)
    assert counts.tolist() == [3]
    assert masks is None
    assert torch.isfinite(predictions).all()
    assert set(predictions[:, 0].tolist()) <= {0.0, 1.0}
    assert torch.all((predictions[:, 1] >= 0) & (predictions[:, 1] <= 1))
    assert torch.all(predictions[:, 2:][..., 0::2] <= 200)
    assert torch.all(predictions[:, 2:][..., 1::2] <= 100)


def test_detr_post_process_softmax_handles_masks_without_topk_truncation():
    post_process = DETRPostProcess(
        num_classes=2,
        num_top_queries=3,
        use_focal_loss=False,
        with_mask=True,
    )
    boxes = torch.tensor([[[0.50, 0.50, 0.40, 0.20], [0.25, 0.25, 0.20, 0.40]]])
    logits = torch.tensor([[[4.0, -4.0, 0.0], [1.0, 3.0, -2.0]]])
    masks = torch.ones(1, 2, 2, 2)

    predictions, counts, mask_predictions = post_process(
        (boxes, logits, masks),
        torch.tensor([[4.0, 4.0]]),
        torch.ones(1, 2),
    )

    expected_scores = torch.softmax(logits, dim=-1)[..., :-1].max(dim=-1).values
    assert predictions.shape == (2, 6)
    assert counts.tolist() == [2]
    assert torch.allclose(predictions[:, 1], expected_scores.flatten())
    assert mask_predictions is not None
    assert mask_predictions.shape == (2, 4, 4)


def test_rtdetrv3_from_config_uses_backbone_shape_without_neck(monkeypatch):
    calls = []
    backbone = SimpleNamespace(out_shape=["backbone-shape"])
    transformer = SimpleNamespace(hidden_dim=32, nhead=8)

    def fake_create(component, **kwargs):
        calls.append((component, kwargs))
        if component == "backbone":
            return backbone
        if component == "transformer":
            return transformer
        if component == "detr_head":
            return "head"
        raise AssertionError(f"unexpected component: {component}")

    monkeypatch.setattr(rtdetrv3_module, "create", fake_create)

    components = RTDETRV3.from_config(
        {
            "backbone": "backbone",
            "neck": None,
            "transformer": "transformer",
            "detr_head": "detr_head",
            "aux_o2m_head": None,
        }
    )

    assert components["neck"] is None
    assert components["aux_o2m_head"] is None
    assert calls[1] == ("transformer", {"input_shape": backbone.out_shape})


def test_dinov3_head_current_eval_api_selects_decoder_layer():
    head = DINOv3Head(loss=None, eval_idx=-1)
    head.eval()
    decoder_boxes = torch.rand(2, 1, 4, 4)
    decoder_logits = torch.randn(2, 1, 4, 2)
    encoder_boxes = torch.rand(1, 4, 4)
    encoder_logits = torch.randn(1, 4, 2)

    boxes, logits, masks = head(
        (
            decoder_boxes,
            decoder_logits,
            encoder_boxes,
            encoder_logits,
            None,
        )
    )

    assert torch.equal(boxes, decoder_boxes[-1])
    assert torch.equal(logits, decoder_logits[-1])
    assert masks is None
