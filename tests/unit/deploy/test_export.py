import json

import pytest
import torch

from ppdet_pytorch.deploy import (
    TORCHSCRIPT_METADATA_FILE,
    DetectionExportAdapter,
    export_onnx,
    export_torchscript,
    make_example_inputs,
    run_onnx,
    run_torchscript,
    validate_detection_outputs,
)


class _DetectionModel(torch.nn.Module):
    def forward(self, inputs):
        image = inputs["image"]
        im_shape = inputs["im_shape"]
        scale_factor = inputs["scale_factor"]
        batch_size = image.shape[0]
        labels = torch.zeros(batch_size, device=image.device)
        scores = image.mean(dim=(1, 2, 3))
        boxes = torch.stack(
            (
                scale_factor[:, 0],
                scale_factor[:, 1],
                im_shape[:, 1],
                im_shape[:, 0],
            ),
            dim=1,
        )
        bbox = torch.cat((labels[:, None], scores[:, None], boxes), dim=1)
        bbox_num = torch.ones(batch_size, dtype=torch.int32, device=image.device)
        return {"bbox": bbox, "bbox_num": bbox_num}


def _adapter():
    return DetectionExportAdapter(_DetectionModel()).eval()


def test_make_example_inputs_rejects_non_positive_shapes():
    with pytest.raises(ValueError, match="positive"):
        make_example_inputs(0, 8, 8)


def test_make_example_inputs_are_deterministic_without_changing_global_rng():
    torch.manual_seed(7)
    expected_next = torch.rand(1)
    torch.manual_seed(7)

    first = make_example_inputs(2, 8, 12)
    second = make_example_inputs(2, 8, 12)

    assert torch.equal(first[0], second[0])
    assert torch.equal(torch.rand(1), expected_next)
    assert first[0].std() > 0


def test_torchscript_export_reloads_with_another_batch_size(tmp_path):
    adapter = _adapter()
    example = make_example_inputs(1, 8, 12)
    output_path = tmp_path / "model.pt"

    export_torchscript(adapter, example, output_path)
    extra_files = {TORCHSCRIPT_METADATA_FILE: b""}
    torch.jit.load(str(output_path), _extra_files=extra_files)
    candidate_inputs = make_example_inputs(4, 8, 12)
    reference = adapter(*candidate_inputs)
    candidate = run_torchscript(output_path, candidate_inputs)

    assert validate_detection_outputs(reference, candidate)["batch_size"] == 4
    assert json.loads(extra_files[TORCHSCRIPT_METADATA_FILE]) == {
        "input_size": [8, 12],
        "schema_version": 1,
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_onnx_export_runs_with_dynamic_batch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    adapter = _adapter()
    example = make_example_inputs(1, 8, 12)
    output_path = tmp_path / "model.onnx"

    export_onnx(adapter, example, output_path, dynamic_batch=True)
    candidate_inputs = make_example_inputs(4, 8, 12)
    reference = adapter(*candidate_inputs)
    candidate = run_onnx(output_path, candidate_inputs)

    assert validate_detection_outputs(reference, candidate)["batch_size"] == 4
    assert not list(tmp_path.glob(".*.tmp"))


def test_onnx_export_failure_preserves_existing_output(tmp_path, monkeypatch):
    output_path = tmp_path / "model.onnx"
    output_path.write_bytes(b"previous")

    def fail_export(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="export failed"):
        export_onnx(
            _adapter(),
            make_example_inputs(1, 8, 12),
            output_path,
        )

    assert output_path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.tmp"))


def test_torchscript_validation_failure_preserves_existing_output(tmp_path):
    output_path = tmp_path / "model.pt"
    output_path.write_bytes(b"previous")

    def fail_validation(path):
        assert path != output_path
        raise AssertionError("parity failed")

    with pytest.raises(AssertionError, match="parity failed"):
        export_torchscript(
            _adapter(),
            make_example_inputs(1, 8, 12),
            output_path,
            validate=fail_validation,
        )

    assert output_path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.tmp"))


def test_validate_detection_outputs_accepts_empty_and_rejects_label_changes():
    empty = (
        torch.empty((0, 6)),
        torch.zeros((1,), dtype=torch.int32),
    )
    assert validate_detection_outputs(empty, empty)["detections"] == 0

    reference = (
        torch.tensor([[1.0, 0.9, 0.0, 0.0, 1.0, 1.0]]),
        torch.ones((1,), dtype=torch.int32),
    )
    candidate = (
        torch.tensor([[2.0, 0.9, 0.0, 0.0, 1.0, 1.0]]),
        torch.ones((1,), dtype=torch.int32),
    )
    with pytest.raises(AssertionError, match="labels"):
        validate_detection_outputs(reference, candidate)


def test_validate_detection_outputs_matches_reordered_near_ties():
    reference = (
        torch.tensor(
            [
                [1.0, 0.90000, 0.0, 0.0, 10.0, 10.0],
                [2.0, 0.89999, 20.0, 20.0, 30.0, 30.0],
            ]
        ),
        torch.tensor([2], dtype=torch.int32),
    )
    candidate = (
        torch.tensor(
            [
                [2.0, 0.90000, 20.01, 20.0, 30.0, 30.0],
                [1.0, 0.89999, 0.0, 0.0, 10.0, 10.0],
            ]
        ),
        torch.tensor([2], dtype=torch.int32),
    )

    metrics = validate_detection_outputs(reference, candidate)

    assert metrics["order_equal"] is False
    assert metrics["reordered_detections"] == 2
    assert metrics["detections"] == 2
    assert metrics["score_max_abs"] == pytest.approx(1e-5, abs=1e-7)
    assert metrics["box_max_abs"] == pytest.approx(0.01, abs=1e-6)


def test_validate_detection_outputs_keeps_identical_duplicate_order():
    bbox = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0, 10.0, 10.0],
            [1.0, 0.9, 0.0, 0.0, 10.0, 10.0],
        ]
    )
    outputs = (bbox, torch.tensor([2], dtype=torch.int32))

    metrics = validate_detection_outputs(outputs, outputs)

    assert metrics["order_equal"] is True
    assert metrics["reordered_detections"] == 0


def test_validate_detection_outputs_does_not_match_across_images():
    reference = (
        torch.tensor(
            [
                [1.0, 0.9, 0.0, 0.0, 1.0, 1.0],
                [2.0, 0.8, 2.0, 2.0, 3.0, 3.0],
            ]
        ),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    candidate = (
        reference[0].flip(0),
        reference[1].clone(),
    )

    with pytest.raises(AssertionError, match="image 0"):
        validate_detection_outputs(reference, candidate)


def test_validate_detection_outputs_rejects_excessive_coordinate_error():
    reference = (
        torch.tensor([[1.0, 0.9, 0.0, 0.0, 1.0, 1.0]]),
        torch.ones((1,), dtype=torch.int32),
    )
    candidate = (
        torch.tensor([[1.0, 0.9, 0.03, 0.0, 1.0, 1.0]]),
        torch.ones((1,), dtype=torch.int32),
    )

    with pytest.raises(AssertionError, match="within tolerances"):
        validate_detection_outputs(reference, candidate)
