from collections import OrderedDict

import pytest
import torch

from ppdet_pytorch.core.workspace import create
from ppdet_pytorch.modeling.backbones.hgnetv2 import FrozenBatchNorm2d, HGNetv2


@pytest.mark.parametrize(
    ("name", "use_lab", "return_idx", "channels"),
    [
        ("B0", True, (2, 3), [512, 1024]),
        ("B2", True, (1, 2, 3), [384, 768, 1536]),
        ("B4", False, (1, 2, 3), [512, 1024, 2048]),
        ("B5", False, (1, 2, 3), [512, 1024, 2048]),
    ],
)
def test_hgnetv2_variant_shapes(name, use_lab, return_idx, channels):
    model = HGNetv2(
        name=name,
        use_lab=use_lab,
        return_idx=return_idx,
        freeze_at=-1,
        freeze_norm=False,
    ).eval()
    with torch.no_grad():
        outputs = model({"image": torch.randn(1, 3, 64, 64)})

    assert [output.shape[1] for output in outputs] == channels
    assert [output.shape[-1] for output in outputs] == [
        64 // (2 ** (index + 2)) for index in return_idx
    ]
    assert [(shape.channels, shape.stride) for shape in model.out_shape] == list(
        zip(channels, [2 ** (index + 2) for index in return_idx])
    )


def test_hgnetv2_registry_and_freeze_contract():
    model = create(
        {
            "name": "HGNetv2",
            "use_lab": False,
            "return_idx": [1, 2, 3],
            "freeze_stem_only": False,
            "freeze_at": 1,
            "freeze_norm": True,
        },
        name="B4",
    )

    assert not any(parameter.requires_grad for parameter in model.stem.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.stages[0].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.stages[1].parameters()
    )
    assert any(parameter.requires_grad for parameter in model.stages[2].parameters())
    assert not any(
        isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()
    )
    assert any(isinstance(module, FrozenBatchNorm2d) for module in model.modules())


@pytest.mark.parametrize("name", ["B1", "B3", "B6", "unknown"])
def test_hgnetv2_rejects_unsupported_variant(name):
    with pytest.raises(ValueError, match="unsupported HGNetv2 variant"):
        HGNetv2(name=name)


@pytest.mark.parametrize("return_idx", [(), (1, 1), (2, 1), (4,)])
def test_hgnetv2_rejects_invalid_return_idx(return_idx):
    with pytest.raises(ValueError, match="return_idx"):
        HGNetv2(name="B0", return_idx=return_idx)


def test_hgnetv2_rejects_wrong_layout_before_mutation(tmp_path):
    model = HGNetv2(name="B0", use_lab=True, freeze_at=-1, freeze_norm=False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = OrderedDict((key, value.clone()) for key, value in before.items())
    key = "stem.stem1.conv.weight"
    state[key] = state[key].transpose(0, 1)
    checkpoint = tmp_path / "bad-layout.pth"
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="stem.stem1.conv.weight.*shape"):
        model.load_pretrained(checkpoint)
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )


def test_hgnetv2_rejects_wrong_variant_before_mutation(tmp_path):
    source = HGNetv2(name="B0", use_lab=True, freeze_at=-1, freeze_norm=False)
    target = HGNetv2(name="B2", use_lab=True, freeze_at=-1, freeze_norm=False)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    checkpoint = tmp_path / "b0.pth"
    torch.save(source.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="checkpoint keys do not match B2"):
        target.load_pretrained(checkpoint)
    assert all(
        torch.equal(value, target.state_dict()[key]) for key, value in before.items()
    )
