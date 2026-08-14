import pytest
import torch

from ppdet_pytorch.core.workspace import register
from ppdet_pytorch.modeling.post_process import DETRPostProcess
from ppdet_pytorch.modeling.transformers.dfine_support import (
    DFINEHungarianMatcher,
    box_cxcywh_to_xyxy,
    get_contrastive_denoising_training_group,
    repository_batch_to_dfine_targets,
)
from ppdet_pytorch.modeling.transformers.matchers import HungarianMatcher


def _targets():
    return [
        {
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10]]),
        },
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]


def test_matcher_exact_assignment_for_nonempty_and_empty_images():
    matcher = DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
    )
    outputs = {
        "pred_logits": torch.tensor(
            [
                [[-3.0, 5.0], [4.0, -2.0], [-1.0, -1.0]],
                [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
            ]
        ),
        "pred_boxes": torch.tensor(
            [
                [[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10], [0.5] * 4],
                [[0.1] * 4, [0.2] * 4, [0.3] * 4],
            ]
        ),
    }

    indices = matcher(outputs, _targets())["indices"]

    assert torch.equal(indices[0][0], torch.tensor([0, 1]))
    assert torch.equal(indices[0][1], torch.tensor([0, 1]))
    assert indices[1][0].numel() == indices[1][1].numel() == 0


def test_matcher_all_empty_targets():
    matcher = DFINEHungarianMatcher(
        {"cost_class": 1, "cost_bbox": 1, "cost_giou": 1}, use_focal_loss=True
    )
    targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]
    outputs = {
        "pred_logits": torch.zeros(2, 3, 2),
        "pred_boxes": torch.full((2, 3, 4), 0.5),
    }
    assert all(pair[0].numel() == 0 for pair in matcher(outputs, targets)["indices"])


def test_existing_matcher_has_exact_pinned_assignment_for_nonempty_targets():
    outputs = {
        "pred_logits": torch.tensor([[[-3.0, 5.0], [4.0, -2.0], [-1.0, -1.0]]]),
        "pred_boxes": torch.tensor(
            [[[0.75, 0.75, 0.20, 0.20], [0.20, 0.20, 0.10, 0.10], [0.5] * 4]]
        ),
    }
    targets = _targets()[:1]
    pinned = DFINEHungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
    )(outputs, targets)["indices"]
    existing = HungarianMatcher(
        matcher_coeff={"class": 2, "bbox": 5, "giou": 2},
        use_focal_loss=True,
    )(
        outputs["pred_boxes"],
        outputs["pred_logits"],
        [targets[0]["boxes"]],
        [targets[0]["labels"].unsqueeze(-1)],
    )

    assert torch.equal(existing[0][0], pinned[0][0])
    assert torch.equal(existing[0][1], pinned[0][1])


def test_denoising_exact_indices_and_attention_groups_with_mixed_empty_targets():
    embedding = torch.nn.Embedding(3, 4)
    query_logits, query_boxes, attention_mask, metadata = (
        get_contrastive_denoising_training_group(
            _targets(),
            num_classes=2,
            num_queries=3,
            class_embed=embedding,
            num_denoising=4,
            label_noise_ratio=0,
            box_noise_scale=0,
        )
    )

    assert query_logits.shape == (2, 8, 4)
    assert query_boxes.shape == (2, 8, 4)
    assert metadata["dn_num_group"] == 2
    assert metadata["dn_num_split"] == [8, 3]
    assert torch.equal(metadata["dn_positive_idx"][0], torch.tensor([0, 1, 4, 5]))
    assert metadata["dn_positive_idx"][1].numel() == 0
    assert attention_mask[8:, :8].all()
    assert attention_mask[:4, 4:8].all()
    assert attention_mask[4:8, :4].all()
    assert not attention_mask[:4, :4].any()


def test_denoising_all_empty_returns_pinned_metadata():
    targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
    ]
    result = get_contrastive_denoising_training_group(
        targets, 2, 300, torch.nn.Embedding(3, 4)
    )
    assert result[:3] == (None, None, None)
    assert result[3] == {
        "dn_positive_idx": None,
        "dn_num_group": 0,
        "dn_num_split": [0, 300],
    }


def test_target_bridge_preserves_normalized_boxes_and_adds_original_size():
    normalized = torch.tensor([[0.25, 0.5, 0.2, 0.4]])
    targets = repository_batch_to_dfine_targets(
        {
            "gt_class": [torch.tensor([[2]]), torch.empty(0, 1, dtype=torch.int64)],
            "gt_bbox": [normalized, torch.empty(0, 4)],
            "im_shape": torch.tensor([[240.0, 320.0], [300.0, 200.0]]),
            "scale_factor": torch.tensor([[0.5, 0.5], [1.5, 2.0]]),
        }
    )
    assert targets[0]["boxes"] is normalized
    assert torch.equal(targets[0]["labels"], torch.tensor([2]))
    assert torch.equal(targets[0]["orig_size"], torch.tensor([640.0, 480.0]))
    assert torch.equal(targets[1]["orig_size"], torch.tensor([100.0, 200.0]))


def test_existing_postprocess_matches_pinned_top300_and_repository_surface():
    logits = torch.arange(600, dtype=torch.float32).reshape(1, 2, 300) / 100
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25], [0.25, 0.25, 0.1, 0.2]]])
    postprocess = DETRPostProcess(
        num_classes=300, num_top_queries=300, use_focal_loss=True
    )
    bbox, bbox_num, mask = postprocess(
        (boxes, logits, None),
        im_shape=torch.tensor([[240.0, 320.0]]),
        scale_factor=torch.tensor([[0.5, 0.5]]),
    )

    scores, flat_indices = torch.topk(logits.sigmoid().flatten(1), 300, dim=-1)
    expected_labels = flat_indices % 300
    expected_queries = flat_indices // 300
    original_size = torch.tensor([[640.0, 480.0, 640.0, 480.0]])
    expected_boxes = box_cxcywh_to_xyxy(boxes) * original_size.unsqueeze(1)
    expected_boxes = expected_boxes.gather(
        1, expected_queries.unsqueeze(-1).expand(-1, -1, 4)
    )
    expected = torch.cat(
        [expected_labels.unsqueeze(-1).float(), scores.unsqueeze(-1), expected_boxes],
        dim=-1,
    ).reshape(-1, 6)

    torch.testing.assert_close(bbox, expected, rtol=0, atol=0)
    assert torch.equal(bbox_num, torch.tensor([300], dtype=torch.int32))
    assert mask is None


def test_rejects_duplicate_registration():
    with pytest.raises(ValueError, match="already registered: DFINEHungarianMatcher"):
        register(DFINEHungarianMatcher)


@pytest.mark.parametrize("field", ["target", "prediction"])
def test_rejects_nonfinite_boxes(field):
    matcher = DFINEHungarianMatcher(
        {"cost_class": 1, "cost_bbox": 1, "cost_giou": 1}, use_focal_loss=True
    )
    targets = _targets()
    outputs = {
        "pred_logits": torch.zeros(2, 3, 2),
        "pred_boxes": torch.full((2, 3, 4), 0.5),
    }
    if field == "target":
        targets[0]["boxes"][0, 0] = float("nan")
        message = "nonfinite boxes"
    else:
        outputs["pred_boxes"][0, 0, 0] = float("inf")
        message = "predictions must be finite"
    with pytest.raises(ValueError, match=message):
        matcher(outputs, targets)


def test_rejects_malformed_target_lengths():
    with pytest.raises(ValueError, match="mismatched label and box lengths"):
        repository_batch_to_dfine_targets(
            {
                "gt_class": [torch.tensor([[0], [1]])],
                "gt_bbox": [torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
            }
        )
