import pytest
import torch
from torch import nn

from detrs.core.workspace import global_config
from detrs.modeling.transformers import DFINETransformer


def _model(**kwargs):
    config = {
        "num_classes": 3,
        "hidden_dim": 8,
        "num_queries": 4,
        "feat_channels": (8, 16),
        "feat_strides": (8, 16),
        "num_levels": 2,
        "num_points": (2, 2),
        "nhead": 2,
        "num_layers": 2,
        "dim_feedforward": 16,
        "num_denoising": 4,
        "reg_max": 8,
    }
    config.update(kwargs)
    return DFINETransformer(**config)


def _features(batch=2):
    return [torch.randn(batch, 8, 2, 2), torch.randn(batch, 16, 1, 1)]


def test_deim_silu_applies_to_decoder_and_prediction_mlps():
    model = _model(activation="silu", mlp_act="silu")

    assert isinstance(model.decoder.layers[0].activation, nn.SiLU)
    assert isinstance(model.decoder.lqe_layers[0].reg_conf.act, nn.SiLU)
    assert isinstance(model.query_pos_head.act, nn.SiLU)
    assert isinstance(model.enc_bbox_head.act, nn.SiLU)
    assert isinstance(model.pre_bbox_head.act, nn.SiLU)
    assert isinstance(model.dec_bbox_head[0].act, nn.SiLU)


def _targets():
    return [
        {
            "labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
        },
        {
            "labels": torch.tensor([0, 2], dtype=torch.long),
            "boxes": torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.7, 0.6, 0.2, 0.1]]),
        },
    ]


def test_train_outputs_cover_aux_raw_encoder_and_denoising_contracts():
    torch.manual_seed(0)
    model = _model().train()
    output = model(_features(), _targets())

    assert set(output) == {
        "pred_logits",
        "pred_boxes",
        "pred_corners",
        "ref_points",
        "up",
        "reg_scale",
        "aux_outputs",
        "enc_aux_outputs",
        "pre_outputs",
        "enc_meta",
        "dn_outputs",
        "dn_pre_outputs",
        "dn_meta",
    }
    assert output["pred_logits"].shape == (2, 4, 3)
    assert output["pred_boxes"].shape == (2, 4, 4)
    assert output["pred_corners"].shape == (2, 4, 36)
    assert output["ref_points"].shape == (2, 4, 4)
    assert len(output["aux_outputs"]) == 1
    assert len(output["enc_aux_outputs"]) == 1
    assert len(output["dn_outputs"]) == 2
    for item in output["aux_outputs"] + output["dn_outputs"]:
        assert set(item) == {
            "pred_logits",
            "pred_boxes",
            "pred_corners",
            "ref_points",
            "teacher_corners",
            "teacher_logits",
        }
    assert set(output["pre_outputs"]) == {"pred_logits", "pred_boxes"}
    assert set(output["dn_pre_outputs"]) == {"pred_logits", "pred_boxes"}
    assert output["dn_meta"]["dn_num_split"][1] == 4


def test_denoising_can_be_disabled_and_eval_returns_final_predictions_only():
    model = _model(num_denoising=0).train()
    output = model(_features(), targets=None)
    assert not any(key.startswith("dn_") for key in output)

    model.eval()
    with torch.inference_mode():
        output = model(_features())
    assert set(output) == {"pred_logits", "pred_boxes"}
    assert output["pred_logits"].shape == (2, 4, 3)


def test_deploy_conversion_prunes_heads_and_preserves_eval_output():
    model = _model(num_layers=3, eval_idx=1, num_denoising=4).eval()
    features = _features()
    with torch.inference_mode():
        expected = model(features)
        assert model.convert_to_deploy() is model
        assert model.convert_to_deploy() is model
        actual = model(features)
    assert len(model.decoder.layers) == 2
    assert "decoder.project" in model.state_dict()
    assert isinstance(model.denoising_class_embed, torch.nn.Identity)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_traced_cached_anchors_accept_another_batch_size():
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _model(
                eval_spatial_size=(16, 16), num_layers=1, num_denoising=0
            ).eval()

        def forward(self, first, second):
            output = self.model([first, second])
            return output["pred_logits"], output["pred_boxes"]

    adapter = Adapter()
    traced = torch.jit.trace(adapter, tuple(_features(batch=1)), strict=False)
    candidate_features = _features(batch=4)
    expected = adapter(*candidate_features)
    actual = traced(*candidate_features)
    for expected_value, actual_value in zip(expected, actual):
        torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"feat_channels": (8, 16, 32), "num_levels": 2}, "feat_channels"),
        ({"feat_strides": (8,), "num_levels": 2}, "feat_strides"),
        ({"num_levels": 0, "feat_channels": (), "feat_strides": ()}, "feature level"),
        ({"reg_max": 7}, "reg_max"),
        ({"reg_max": 2}, "reg_max"),
    ],
)
def test_invalid_levels_and_reg_max_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model(**kwargs)


def test_state_keys_and_registration_are_canonical():
    model = _model(eval_spatial_size=(16, 16), num_denoising=0)
    keys = set(model.state_dict())
    assert {
        "up",
        "reg_scale",
        "enc_output.proj.weight",
        "enc_score_head.weight",
        "enc_bbox_head.layers.2.weight",
        "query_pos_head.layers.0.weight",
        "pre_bbox_head.layers.2.weight",
        "dec_score_head.0.weight",
        "dec_bbox_head.0.layers.2.weight",
        "decoder.layers.0.self_attn.in_proj_weight",
    } <= keys
    assert {"anchors", "valid_mask"} <= keys
    assert global_config["DFINETransformer"].cls is DFINETransformer
