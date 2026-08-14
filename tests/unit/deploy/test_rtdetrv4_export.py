from pathlib import Path

import torch

from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.deploy import (
    DetectionExportAdapter,
    export_onnx,
    export_torchscript,
    make_example_inputs,
    run_onnx,
    run_torchscript,
)

ROOT = Path(__file__).resolve().parents[3]


def _adapter():
    config = load_config(ROOT / "configs/rtdetrv4/rtdetrv4_hgnetv2_s_coco.yml")
    model = create(config.architecture).eval().deploy()
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    model.decoder.num_queries = 4
    assert model.encoder.feature_projector is None
    assert not any(
        "teacher" in name or "distill" in name for name in model.state_dict()
    )
    return DetectionExportAdapter(model).eval()


def test_rtdetrv4_torchscript_is_student_only_and_accepts_batch_four(
    isolated_workspace, tmp_path
):
    adapter = _adapter()

    example = make_example_inputs(1, 64, 64)
    output = tmp_path / "rtdetrv4-s.pt"
    with torch.inference_mode():
        expected = adapter(*example)
    export_torchscript(adapter, example, output)

    scripted = torch.jit.load(str(output))
    assert not any(
        "teacher" in name or "distill" in name or "feature_projector" in name
        for name, _ in scripted.named_parameters()
    )
    actual = run_torchscript(output, make_example_inputs(4, 64, 64))
    assert actual[1].shape[0] == 4
    torch.testing.assert_close(run_torchscript(output, example), expected)


def test_rtdetrv4_onnx_is_student_only_and_accepts_batch_four(
    isolated_workspace, tmp_path
):
    onnx = __import__("onnx")
    adapter = _adapter()
    example = make_example_inputs(1, 64, 64)
    output = tmp_path / "rtdetrv4-s.onnx"

    export_onnx(adapter, example, output, opset_version=17)

    graph = onnx.load(str(output)).graph
    assert not any(
        "teacher" in value.name
        or "distill" in value.name
        or "feature_projector" in value.name
        for value in graph.initializer
    )
    actual = run_onnx(output, make_example_inputs(4, 64, 64))
    assert actual[1].shape[0] == 4
