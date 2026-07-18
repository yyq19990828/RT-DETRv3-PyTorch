import torch

from ppdet_pytorch.modeling.backbones.resnet import ResNet
from ppdet_pytorch.modeling.heads.detr_head import DINOv3Head
from ppdet_pytorch.modeling.post_process import DETRPostProcess


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
