import pytest
import torch

from detrs.modeling.losses.deim_loss import DEIMCriterion
from detrs.modeling.transformers.dfine_decoder import DFINETransformer
from detrs.modeling.transformers.dfine_support import DFINEHungarianMatcher
from detrs.modeling.transformers.rtdetr_transformerv2 import (
    RTDETRTransformerv2,
)


def _criterion(*, losses=("mal", "boxes"), mal_alpha=None, use_uni_set=False):
    return DEIMCriterion(
        matcher=DFINEHungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
        ),
        weight_dict={"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2},
        losses=list(losses),
        gamma=1.5,
        num_classes=3,
        mal_alpha=mal_alpha,
        use_uni_set=use_uni_set,
    )


def _outputs():
    generator = torch.Generator().manual_seed(31)

    def prediction():
        return {
            "pred_logits": torch.randn(
                2, 4, 3, generator=generator, requires_grad=True
            ),
            "pred_boxes": (
                torch.rand(2, 4, 4, generator=generator) * 0.4 + 0.2
            ).requires_grad_(),
        }

    return {
        **prediction(),
        "aux_outputs": [prediction()],
        "enc_aux_outputs": [],
        "enc_meta": {"class_agnostic": False},
    }


def _targets():
    return [
        {
            "labels": torch.tensor([0, 2]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]),
        },
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.3, 0.6, 0.2, 0.1]])},
    ]


def test_rtdetrv2_contract_does_not_require_pre_or_local_outputs():
    losses = _criterion()(_outputs(), _targets())
    assert set(losses) == {
        "loss_mal",
        "loss_bbox",
        "loss_giou",
        "loss_mal_aux_0",
        "loss_bbox_aux_0",
        "loss_giou_aux_0",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()


def test_mal_alpha_only_scales_negative_weight():
    outputs = _outputs()
    targets = _targets()
    matcher = _criterion().matcher(outputs, targets)["indices"]
    regular = _criterion(mal_alpha=None).loss_labels_mal(outputs, targets, matcher, 3)[
        "loss_mal"
    ]
    scaled = _criterion(mal_alpha=0.5).loss_labels_mal(outputs, targets, matcher, 3)[
        "loss_mal"
    ]
    assert scaled < regular
    assert scaled > regular * 0.5


def test_rejects_local_on_rtdetrv2_without_distribution_outputs():
    with pytest.raises(ValueError, match="local loss requires"):
        _criterion(losses=("mal", "boxes", "local"))(_outputs(), _targets())


def test_rejects_vfl_substitution():
    with pytest.raises(ValueError, match="unsupported DEIM losses"):
        _criterion(losses=("vfl", "boxes"))


def test_rejects_gamma_other_than_official_mal_value():
    with pytest.raises(ValueError, match="gamma must be 1.5"):
        DEIMCriterion(
            matcher=None,
            weight_dict={"loss_mal": 1},
            losses=["mal"],
            gamma=2.0,
            num_classes=3,
        )


def test_rejects_union_indices_that_duplicate_query_rows():
    main = [
        (torch.tensor([0, 1]), torch.tensor([0, 1])),
        (torch.tensor([2]), torch.tensor([0])),
    ]
    auxiliary = [
        [
            (torch.tensor([0, 1]), torch.tensor([1, 1])),
            (torch.tensor([2, 3]), torch.tensor([0, 0])),
        ],
        [
            (torch.tensor([0, 2]), torch.tensor([1, 0])),
            (torch.tensor([2]), torch.tensor([0])),
        ],
    ]

    union = DEIMCriterion._get_go_indices(main, auxiliary)

    assert torch.equal(union[0][0], torch.tensor([0, 1, 2]))
    assert torch.equal(union[0][1], torch.tensor([1, 1, 0]))
    assert torch.equal(union[1][0], torch.tensor([2, 3]))
    assert torch.equal(union[1][1], torch.tensor([0, 0]))


def test_class_agnostic_encoder_matches_zeroed_labels_before_loss():
    outputs = _outputs()
    outputs["enc_aux_outputs"] = [
        {
            "pred_logits": torch.zeros(2, 4, 1, requires_grad=True),
            "pred_boxes": torch.full((2, 4, 4), 0.4, requires_grad=True),
        }
    ]
    outputs["enc_meta"]["class_agnostic"] = True

    losses = _criterion()(outputs, _targets())

    assert {"loss_mal_enc_0", "loss_bbox_enc_0", "loss_giou_enc_0"} <= losses.keys()
    assert all(torch.isfinite(value) for value in losses.values())


def test_mal_clamps_negative_giou_quality_before_fractional_power():
    criterion = DEIMCriterion(
        matcher=None,
        weight_dict={"loss_mal": 1},
        losses=["mal"],
        gamma=1.5,
        num_classes=3,
        boxes_weight_format="giou",
    )
    outputs = {
        "pred_logits": torch.zeros(1, 1, 3, requires_grad=True),
        "pred_boxes": torch.tensor([[[0.9, 0.9, 0.1, 0.1]]], requires_grad=True),
    }
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.1, 0.1, 0.1, 0.1]])}
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    metadata = criterion.get_loss_meta_info("mal", outputs, targets, indices)

    loss = criterion.loss_labels_mal(outputs, targets, indices, 1, **metadata)[
        "loss_mal"
    ]

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(outputs["pred_logits"].grad).all()


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"dn_positive_idx": []}, {"dn_num_group": 1}],
)
def test_rejects_malformed_denoising_metadata(metadata):
    outputs = _outputs()
    outputs["dn_outputs"] = []
    outputs["dn_meta"] = metadata

    with pytest.raises(ValueError, match="dn_meta"):
        _criterion()(outputs, _targets())


def test_rejects_malformed_denoising_pre_output():
    outputs = _outputs()
    outputs["dn_outputs"] = []
    outputs["dn_meta"] = {"dn_positive_idx": [[], []], "dn_num_group": 1}
    outputs["dn_pre_outputs"] = {
        "pred_logits": torch.zeros(1, 2, 3),
        "pred_boxes": torch.zeros(1, 2, 4),
    }

    with pytest.raises(ValueError, match="inconsistent batch size"):
        _criterion()(outputs, _targets())


def test_empty_targets_keep_mal_and_zero_box_losses_finite():
    empty_targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
        for _ in range(2)
    ]

    losses = _criterion()(_outputs(), empty_targets)

    assert losses["loss_mal"] > 0
    assert losses["loss_bbox"] == 0
    assert losses["loss_giou"] == 0
    assert all(torch.isfinite(value) for value in losses.values())


@pytest.mark.parametrize("family", ["dfine", "rtdetrv2"])
def test_real_decoder_graph_one_step_backward(family):
    torch.manual_seed(101)
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])},
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]
    if family == "dfine":
        decoder = DFINETransformer(
            num_classes=3,
            hidden_dim=8,
            num_queries=4,
            feat_channels=(8, 16),
            feat_strides=(8, 16),
            num_levels=2,
            num_points=(2, 2),
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            num_denoising=0,
            reg_max=8,
        ).train()
        features = [torch.randn(2, 8, 2, 2), torch.randn(2, 16, 1, 1)]
        losses = ("mal", "boxes", "local")
        weights = {
            "loss_mal": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        }
        use_uni_set = True
    else:
        decoder = RTDETRTransformerv2(
            variant="r18vd",
            num_classes=3,
            num_queries=4,
            num_denoising=0,
            eval_spatial_size=None,
        ).train()
        features = [
            torch.randn(2, 256, 2, 2),
            torch.randn(2, 256, 1, 1),
            torch.randn(2, 256, 1, 1),
        ]
        losses = ("mal", "boxes")
        weights = {"loss_mal": 1, "loss_bbox": 5, "loss_giou": 2}
        use_uni_set = False
    criterion = DEIMCriterion(
        matcher=DFINEHungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
        ),
        weight_dict=weights,
        losses=losses,
        gamma=1.5,
        num_classes=3,
        reg_max=8,
        use_uni_set=use_uni_set,
    )

    outputs = decoder(features, targets)
    result = criterion(outputs, targets)
    total = sum(result.values())
    total.backward()

    assert torch.isfinite(total)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )
