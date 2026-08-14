import pytest
import torch

from ppdet_pytorch.deploy import (
    DetectionExportAdapter,
    export_torchscript,
    make_example_inputs,
    run_torchscript,
)


class _InferenceModel(torch.nn.Module):
    def forward(self, inputs):
        image = inputs["image"]
        batch_size = image.shape[0]
        bbox = torch.cat(
            [
                torch.zeros((batch_size, 1), device=image.device),
                image.mean((1, 2, 3)).unsqueeze(1),
                torch.ones((batch_size, 4), device=image.device),
            ],
            dim=1,
        )
        return {
            "bbox": bbox,
            "bbox_num": torch.ones(batch_size, dtype=torch.int32, device=image.device),
        }


class _TrainingModel(torch.nn.Module):
    def forward(self, inputs):
        return {"pred_logits": inputs["image"].mean((2, 3))}


def test_rejects_wrong_size(tmp_path):
    path = tmp_path / "model.pt"
    adapter = DetectionExportAdapter(_InferenceModel()).eval()
    export_torchscript(adapter, make_example_inputs(1, 8, 12), path)

    with pytest.raises(ValueError, match="fixed export size"):
        run_torchscript(path, make_example_inputs(1, 10, 12))


def test_rejects_dynamic_height(tmp_path):
    pytest.importorskip("onnx")
    import onnx

    from ppdet_pytorch.deploy import export_onnx

    path = tmp_path / "model.onnx"
    export_onnx(
        DetectionExportAdapter(_InferenceModel()).eval(),
        make_example_inputs(1, 8, 12),
        path,
    )
    graph = onnx.load(str(path))
    image = next(value for value in graph.graph.input if value.name == "image")
    dimensions = image.type.tensor_type.shape.dim
    assert dimensions[0].dim_param == "batch"
    assert [dimension.dim_value for dimension in dimensions[2:]] == [8, 12]
    assert not dimensions[2].dim_param and not dimensions[3].dim_param


def test_rejects_training_output_without_publishing(tmp_path):
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="inference outputs"):
        export_torchscript(
            DetectionExportAdapter(_TrainingModel()).eval(),
            make_example_inputs(1, 8, 12),
            path,
        )

    assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
