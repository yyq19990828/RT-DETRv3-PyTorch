import os
from argparse import Namespace

import yaml
from PIL import Image, ImageFont

from ppdet_pytorch.utils.checkpoint import get_latest_checkpoint
from ppdet_pytorch.utils.cli import ArgsParser
from ppdet_pytorch.utils.config import AttrDict, save_config
from ppdet_pytorch.utils.visualizer import draw_bbox


def test_args_parser_preserves_namespace_and_parses_nested_options():
    namespace = Namespace(seed=17)

    parsed = ArgsParser().parse_args(
        ["-c", "model.yml", "-o", "epochs=12", "Model.depth=18"],
        namespace=namespace,
    )

    assert parsed is namespace
    assert parsed.config == "model.yml"
    assert parsed.seed == 17
    assert parsed.opt == {"epochs": 12, "Model": {"depth": 18}}


def test_save_config_creates_parent_and_round_trips(tmp_path):
    output_path = tmp_path / "nested" / "config.yml"

    save_config(
        AttrDict(epochs=12, Model={"depth": 18}),
        str(output_path),
    )

    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == {
        "epochs": 12,
        "Model": {"depth": 18},
    }


def test_draw_bbox_uses_default_font_fallback(monkeypatch):
    fallback_font = ImageFont.load_default()

    def missing_font(*_args, **_kwargs):
        raise OSError("missing font")

    monkeypatch.setattr(ImageFont, "truetype", missing_font)
    monkeypatch.setattr(ImageFont, "load_default", lambda: fallback_font)
    image = Image.new("RGB", (32, 32), "white")

    result = draw_bbox(
        image,
        im_id=1,
        catid2name={7: "object"},
        bboxes=[
            {
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 2, 20, 20],
                "score": 0.9,
            }
        ],
        threshold=0.5,
    )

    assert result is image
    assert result.getpixel((2, 2)) != (255, 255, 255)


def test_get_latest_checkpoint_accepts_string_directory(tmp_path):
    first = tmp_path / "checkpoint_1.pth"
    latest = tmp_path / "checkpoint_2.pth"
    first.write_bytes(b"first")
    latest.write_bytes(b"latest")
    os.utime(first, (1, 1))
    os.utime(latest, (2, 2))

    assert get_latest_checkpoint(str(tmp_path)) == latest
