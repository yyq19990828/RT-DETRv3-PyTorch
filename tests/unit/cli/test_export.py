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


@pytest.mark.parametrize(
    ("image_shape", "message"),
    [
        ([3, 640], "channels, height, width"),
        (["3", 640, 640], "must be integers"),
        ([1, 640, 640], "channels must be 3"),
        ([3, 0, 640], "must be positive"),
    ],
)
def test_input_size_rejects_invalid_config_shape(image_shape, message):
    cfg = AttrDict(TestReader={"inputs_def": {"image_shape": image_shape}})

    with pytest.raises(ValueError, match=message):
        export_cli._input_size(cfg)


def test_input_size_requires_config_shape_without_override():
    with pytest.raises(ValueError, match="must define"):
        export_cli._input_size(AttrDict())


def test_output_paths_are_deterministic(tmp_path):
    cfg = AttrDict(filename="rtdetrv3_r18vd_6x_coco")

    paths = export_cli._output_paths(cfg, tmp_path, "both")

    assert paths == {
        "onnx": tmp_path / "rtdetrv3_r18vd_6x_coco.onnx",
        "torchscript": tmp_path / "rtdetrv3_r18vd_6x_coco.torchscript.pt",
    }
    assert all(isinstance(path, Path) for path in paths.values())


def test_main_wires_both_export_formats_and_verification(monkeypatch, tmp_path):
    cfg = AttrDict(filename="fixture", eval_size=[640, 640])
    observed = []

    class FakeAdapter:
        def eval(self):
            observed.append(("adapter_eval",))
            return self

        def __call__(self, *inputs):
            observed.append(("reference", inputs))
            return "reference"

    monkeypatch.setattr(export_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        export_cli,
        "apply_overrides",
        lambda loaded_cfg, overrides: observed.append(
            ("overrides", loaded_cfg, overrides)
        ),
    )
    monkeypatch.setattr(
        export_cli,
        "build_model",
        lambda loaded_cfg, checkpoint, device, use_ema: (
            observed.append(("model", loaded_cfg, checkpoint, device.type, use_ema))
            or object()
        ),
    )
    monkeypatch.setattr(
        export_cli, "DetectionExportAdapter", lambda model: FakeAdapter()
    )
    monkeypatch.setattr(
        export_cli,
        "make_example_inputs",
        lambda batch, height, width: ("image", "im_shape", batch, height, width),
    )
    monkeypatch.setattr(
        export_cli,
        "export_onnx",
        lambda adapter, inputs, path, opset_version, dynamic_batch: observed.append(
            ("export_onnx", path, opset_version, dynamic_batch)
        ),
    )
    monkeypatch.setattr(
        export_cli,
        "export_torchscript",
        lambda adapter, inputs, path: observed.append(("export_torchscript", path)),
    )
    monkeypatch.setattr(
        export_cli,
        "run_onnx",
        lambda path, inputs: ("onnx", path, inputs),
    )
    monkeypatch.setattr(
        export_cli,
        "run_torchscript",
        lambda path, inputs: ("torchscript", path, inputs),
    )
    monkeypatch.setattr(
        export_cli,
        "validate_detection_outputs",
        lambda reference, actual: (
            observed.append(("verify", reference, actual)) or {"max_abs_diff": 0.0}
        ),
    )

    result = export_cli.main(
        [
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--format",
            "both",
            "--output-dir",
            str(tmp_path),
            "--input-size",
            "320",
            "480",
            "--batch-size",
            "2",
            "--fixed-batch",
            "--use-ema",
            "-o",
            "architecture=Fixture",
        ]
    )

    assert result == 0
    assert cfg.eval_size == [320, 480]
    assert ("model", cfg, "model.pth", "cpu", True) in observed
    assert ("export_onnx", tmp_path / "fixture.onnx", 17, False) in observed
    assert (
        "export_torchscript",
        tmp_path / "fixture.torchscript.pt",
    ) in observed
    assert len([event for event in observed if event[0] == "verify"]) == 2


def test_main_refuses_to_overwrite_existing_export(monkeypatch, tmp_path):
    cfg = AttrDict(filename="fixture", eval_size=[640, 640])
    (tmp_path / "fixture.onnx").write_bytes(b"existing")
    monkeypatch.setattr(export_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(export_cli, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(
        export_cli,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )

    with pytest.raises(FileExistsError, match="use --force"):
        export_cli.main(
            [
                "--config",
                "model.yml",
                "--checkpoint",
                "model.pth",
                "--format",
                "onnx",
                "--output-dir",
                str(tmp_path),
                "--input-size",
                "640",
                "640",
            ]
        )
