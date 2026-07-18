"""Tensor-only adapters for ONNX and TorchScript deployment."""

import math
import os
import tempfile
from pathlib import Path

import torch
from torch import nn


class DetectionExportAdapter(nn.Module):
    """Expose RT-DETRv3's batch-dict contract as tensor inputs/outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, im_shape, scale_factor):
        outputs = self.model(
            {
                "image": image,
                "im_shape": im_shape,
                "scale_factor": scale_factor,
            }
        )
        return outputs["bbox"], outputs["bbox_num"]


def make_example_inputs(batch_size, height, width, device="cpu"):
    """Create deterministic inputs matching the eager inference contract."""
    if batch_size < 1 or height < 1 or width < 1:
        raise ValueError("batch size, height, and width must be positive")
    device = torch.device(device)
    image = torch.zeros((batch_size, 3, height, width), device=device)
    im_shape = torch.tensor(
        [[height, width]], dtype=torch.float32, device=device
    ).repeat(batch_size, 1)
    scale_factor = torch.ones((batch_size, 2), device=device)
    return image, im_shape, scale_factor


def _temporary_path(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=".{}-".format(output_path.name),
        suffix=".tmp",
    )
    os.close(descriptor)
    return output_path, Path(temporary_name)


def export_onnx(
    adapter,
    example_inputs,
    output_path,
    opset_version=17,
    dynamic_batch=True,
):
    """Export and validate an ONNX model, then publish it atomically."""
    output_path, temporary_path = _temporary_path(output_path)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "im_shape": {0: "batch"},
            "scale_factor": {0: "batch"},
            "bbox": {0: "detections"},
            "bbox_num": {0: "batch"},
        }
    try:
        adapter.eval()
        torch.onnx.export(
            adapter,
            example_inputs,
            str(temporary_path),
            input_names=["image", "im_shape", "scale_factor"],
            output_names=["bbox", "bbox_num"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
        import onnx

        onnx.checker.check_model(onnx.load(str(temporary_path)))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def export_torchscript(adapter, example_inputs, output_path):
    """Trace, reload-check, and atomically publish a TorchScript model."""
    output_path, temporary_path = _temporary_path(output_path)
    try:
        adapter.eval()
        with torch.inference_mode():
            traced = torch.jit.trace(
                adapter,
                example_inputs,
                strict=False,
                check_trace=True,
            )
        traced.save(str(temporary_path))
        torch.jit.load(str(temporary_path), map_location="cpu")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def run_onnx(model_path, inputs):
    """Run an exported model with ONNX Runtime's CPU provider."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feed = {
        name: value.detach().cpu().numpy()
        for name, value in zip(
            ("image", "im_shape", "scale_factor"),
            inputs,
        )
    }
    return tuple(torch.from_numpy(value) for value in session.run(None, feed))


def run_torchscript(model_path, inputs):
    """Reload and run a TorchScript export on CPU."""
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    cpu_inputs = tuple(value.detach().cpu() for value in inputs)
    with torch.inference_mode():
        return model(*cpu_inputs)


def _max_absolute_error(left, right):
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).item())


def validate_detection_outputs(
    reference,
    candidate,
    score_atol=2e-5,
    box_atol=1e-2,
):
    """Validate row ordering, labels, scores, boxes, and per-image counts."""
    reference_bbox, reference_num = (value.detach().cpu() for value in reference)
    candidate_bbox, candidate_num = (value.detach().cpu() for value in candidate)
    if reference_bbox.shape != candidate_bbox.shape:
        raise AssertionError(
            "bbox shape mismatch: {} != {}".format(
                tuple(reference_bbox.shape), tuple(candidate_bbox.shape)
            )
        )
    if not torch.equal(reference_num, candidate_num):
        raise AssertionError("bbox_num differs")
    if not torch.equal(reference_bbox[:, 0], candidate_bbox[:, 0]):
        raise AssertionError("detection labels or row ordering differ")

    score_error = _max_absolute_error(reference_bbox[:, 1], candidate_bbox[:, 1])
    box_error = _max_absolute_error(reference_bbox[:, 2:], candidate_bbox[:, 2:])
    if not math.isfinite(score_error) or score_error > score_atol:
        raise AssertionError(
            "score max abs error {} exceeds {}".format(score_error, score_atol)
        )
    if not math.isfinite(box_error) or box_error > box_atol:
        raise AssertionError(
            "box max abs error {} exceeds {}".format(box_error, box_atol)
        )
    return {
        "score_max_abs": score_error,
        "box_max_abs": box_error,
        "detections": int(reference_bbox.shape[0]),
        "batch_size": int(reference_num.shape[0]),
    }
