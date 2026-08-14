"""Tensor-only adapters for ONNX and TorchScript deployment."""

import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn

TORCHSCRIPT_METADATA_FILE = "rtdetrv3-export.json"


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
        if not isinstance(outputs, dict) or not {"bbox", "bbox_num"} <= outputs.keys():
            raise ValueError(
                "export model must return inference outputs bbox and bbox_num"
            )
        return outputs["bbox"], outputs["bbox_num"]


def make_example_inputs(batch_size, height, width, device="cpu"):
    """Create deterministic inputs matching the eager inference contract."""
    if batch_size < 1 or height < 1 or width < 1:
        raise ValueError("batch size, height, and width must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(20260813)
    image = torch.rand(
        (batch_size, 3, height, width), device=device, generator=generator
    )
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
    validate=None,
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
        if validate is not None:
            validate(temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def export_torchscript(adapter, example_inputs, output_path, validate=None):
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
        image = example_inputs[0]
        metadata = {
            "input_size": [int(image.shape[-2]), int(image.shape[-1])],
            "schema_version": 1,
        }
        torch.jit.save(
            traced,
            str(temporary_path),
            _extra_files={
                TORCHSCRIPT_METADATA_FILE: json.dumps(metadata, sort_keys=True)
            },
        )
        loaded_metadata = {TORCHSCRIPT_METADATA_FILE: b""}
        torch.jit.load(
            str(temporary_path),
            map_location="cpu",
            _extra_files=loaded_metadata,
        )
        if json.loads(loaded_metadata[TORCHSCRIPT_METADATA_FILE]) != metadata:
            raise RuntimeError("TorchScript export metadata verification failed")
        if validate is not None:
            validate(temporary_path)
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
    metadata_files = {TORCHSCRIPT_METADATA_FILE: b""}
    model = torch.jit.load(
        str(model_path), map_location="cpu", _extra_files=metadata_files
    ).eval()
    metadata = json.loads(metadata_files[TORCHSCRIPT_METADATA_FILE])
    input_size = [int(inputs[0].shape[-2]), int(inputs[0].shape[-1])]
    if input_size != metadata["input_size"]:
        raise ValueError(
            "TorchScript input size {} does not match fixed export size {}".format(
                input_size, metadata["input_size"]
            )
        )
    cpu_inputs = tuple(value.detach().cpu() for value in inputs)
    with torch.inference_mode():
        return model(*cpu_inputs)


def _max_absolute_error(left, right):
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).item())


def _match_image_detections(
    reference_bbox,
    candidate_bbox,
    image_index,
    score_atol,
    box_atol,
):
    """Match an image's unordered detections within explicit tolerances."""
    count = reference_bbox.shape[0]
    if count == 0:
        return [], 0

    score_errors = torch.abs(reference_bbox[:, None, 1] - candidate_bbox[None, :, 1])
    box_errors = torch.amax(
        torch.abs(reference_bbox[:, None, 2:] - candidate_bbox[None, :, 2:]),
        dim=-1,
    )
    compatible = (
        (reference_bbox[:, None, 0] == candidate_bbox[None, :, 0])
        & (score_errors <= score_atol)
        & (box_errors <= box_atol)
    )
    diagonal_compatible = compatible.diagonal()
    reordered = int((~diagonal_compatible).sum().item())
    if reordered == 0:
        return list(range(count)), 0

    adjacency = [
        torch.nonzero(row, as_tuple=False).flatten().tolist() for row in compatible
    ]
    candidate_match = [-1] * count

    def augment(reference_index, seen):
        for candidate_index in adjacency[reference_index]:
            if seen[candidate_index]:
                continue
            seen[candidate_index] = True
            previous_reference = candidate_match[candidate_index]
            if previous_reference == -1 or augment(previous_reference, seen):
                candidate_match[candidate_index] = reference_index
                return True
        return False

    for reference_index in range(count):
        augment(reference_index, [False] * count)

    reference_match = [-1] * count
    for candidate_index, reference_index in enumerate(candidate_match):
        if reference_index != -1:
            reference_match[reference_index] = candidate_index
    unmatched = [index for index, match in enumerate(reference_match) if match == -1]
    if unmatched:
        first = unmatched[0]
        raise AssertionError(
            "image {} detections could not be matched one-to-one by "
            "labels/scores/boxes within tolerances: matched {}/{}, "
            "first unmatched row={} label={} score={}".format(
                image_index,
                count - len(unmatched),
                count,
                first,
                int(reference_bbox[first, 0].item()),
                float(reference_bbox[first, 1].item()),
            )
        )
    return reference_match, reordered


def validate_detection_outputs(
    reference,
    candidate,
    score_atol=2e-5,
    box_atol=2e-2,
):
    """Validate per-image detection sets, values, and output structure."""
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
    if reference_bbox.ndim != 2 or reference_bbox.shape[1] != 6:
        raise AssertionError("bbox must have shape [N, 6]")
    if reference_num.ndim != 1:
        raise AssertionError("bbox_num must have shape [B]")
    counts = [int(value) for value in reference_num.tolist()]
    if any(value < 0 for value in counts) or sum(counts) != len(reference_bbox):
        raise AssertionError("bbox_num does not match bbox rows")
    if (
        not torch.isfinite(reference_bbox).all()
        or not torch.isfinite(candidate_bbox).all()
    ):
        raise AssertionError("detection outputs contain non-finite values")

    matched_reference = []
    matched_candidate = []
    reordered = 0
    start = 0
    for image_index, count in enumerate(counts):
        stop = start + count
        reference_group = reference_bbox[start:stop]
        candidate_group = candidate_bbox[start:stop]
        matches, group_reordered = _match_image_detections(
            reference_group,
            candidate_group,
            image_index,
            score_atol,
            box_atol,
        )
        matched_reference.append(reference_group)
        matched_candidate.append(candidate_group[matches])
        reordered += group_reordered
        start = stop

    aligned_reference = torch.cat(matched_reference, dim=0)
    aligned_candidate = torch.cat(matched_candidate, dim=0)
    score_error = _max_absolute_error(aligned_reference[:, 1], aligned_candidate[:, 1])
    box_error = _max_absolute_error(aligned_reference[:, 2:], aligned_candidate[:, 2:])
    return {
        "score_max_abs": score_error,
        "box_max_abs": box_error,
        "detections": int(reference_bbox.shape[0]),
        "batch_size": int(reference_num.shape[0]),
        "order_equal": reordered == 0,
        "reordered_detections": reordered,
    }
