from pathlib import Path

import pytest

from ppdet_pytorch.cli import export as export_cli
from ppdet_pytorch.core.workspace import AttrDict


def test_parse_args_validates_export_boundaries(capsys):
    base = ["--config", "model.yml", "--checkpoint", "model.pth"]

    for invalid in (
        ["--batch-size", "0"],
        ["--input-size", "640", "0"],
        ["--opset-version", "16"],
    ):
        with pytest.raises(SystemExit):
            export_cli.parse_args(base + invalid)

    assert "error:" in capsys.readouterr().err


def test_input_size_uses_override_or_test_reader_shape():
    cfg = AttrDict(TestReader={"inputs_def": {"image_shape": [3, 608, 640]}})

    assert export_cli._input_size(cfg) == (608, 640)
    assert export_cli._input_size(cfg, [320, 480]) == (320, 480)


def test_output_paths_are_deterministic(tmp_path):
    cfg = AttrDict(filename="rtdetrv3_r18vd_6x_coco")

    paths = export_cli._output_paths(cfg, tmp_path, "both")

    assert paths == {
        "onnx": tmp_path / "rtdetrv3_r18vd_6x_coco.onnx",
        "torchscript": tmp_path / "rtdetrv3_r18vd_6x_coco.torchscript.pt",
    }
    assert all(isinstance(path, Path) for path in paths.values())
