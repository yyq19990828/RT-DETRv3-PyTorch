import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ppdet_pytorch.cli import infer as infer_cli
from ppdet_pytorch.core.workspace import AttrDict
from ppdet_pytorch.data.utils import default_collate_fn
from ppdet_pytorch.deploy import TORCHSCRIPT_METADATA_FILE


def _config_with_test_reader():
    cfg = AttrDict()
    cfg.num_classes = 80
    cfg.TestReader = {
        "sample_transforms": [
            {"Decode": {}},
            {
                "Resize": {
                    "target_size": [8, 12],
                    "keep_ratio": False,
                    "interp": cv2.INTER_LINEAR,
                }
            },
            {
                "NormalizeImage": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "norm_type": "none",
                }
            },
            {"Permute": {}},
        ]
    }
    return cfg


def test_parse_args_accepts_current_and_legacy_flag_spellings():
    args = infer_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--infer_img",
            "image.jpg",
            "--output_dir",
            "results",
            "--save_results",
            "--batch_size",
            "4",
            "--draw-threshold",
            "0.25",
        ]
    )

    assert args.infer_img == "image.jpg"
    assert args.output_dir == "results"
    assert args.save_results is True
    assert args.batch_size == 4
    assert args.threshold == pytest.approx(0.25)


def test_parse_args_uses_backend_specific_default_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    onnx_args = infer_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--onnx-model",
            "model.onnx",
            "--infer-img",
            "image.jpg",
        ]
    )
    torchscript_args = infer_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--torchscript-model",
            "model.pt",
            "--infer-img",
            "image.jpg",
        ]
    )
    checkpoint_args = infer_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--checkpoint",
            "model.pth",
            "--infer-img",
            "image.jpg",
        ]
    )

    assert onnx_args.device == "cpu"
    assert torchscript_args.device == "cuda"
    assert checkpoint_args.device == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fallback_args = infer_cli.parse_args(
        [
            "--config",
            "model.yml",
            "--torchscript-model",
            "model.pt",
            "--infer-img",
            "image.jpg",
        ]
    )
    assert fallback_args.device == "cpu"


def test_parse_args_accepts_explicit_torchscript_cpu_and_cuda():
    base_args = [
        "--config",
        "model.yml",
        "--torchscript-model",
        "model.pt",
        "--infer-img",
        "image.jpg",
    ]

    assert infer_cli.parse_args([*base_args, "--device", "cpu"]).device == "cpu"
    assert infer_cli.parse_args([*base_args, "--device", "cuda:0"]).device == "cuda:0"


def test_parse_args_accepts_explicit_onnx_cpu_and_cuda():
    base_args = [
        "--config",
        "model.yml",
        "--onnx-model",
        "model.onnx",
        "--infer-img",
        "image.jpg",
    ]

    assert infer_cli.parse_args([*base_args, "--device", "cpu"]).device == "cpu"
    assert infer_cli.parse_args([*base_args, "--device", "cuda:1"]).device == "cuda:1"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--onnx-model", "model.onnx", "--checkpoint", "model.pth"],
        ["--onnx-model", "model.onnx", "--use-ema"],
        ["--onnx-model", "model.onnx", "--device", "mps"],
    ],
)
def test_parse_args_rejects_invalid_exported_model_combinations(extra_args, capsys):
    with pytest.raises(SystemExit):
        infer_cli.parse_args(
            ["--config", "model.yml", *extra_args, "--infer-img", "image.jpg"]
        )

    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--threshold", "1.1"], "--threshold"),
        (["--batch-size", "0"], "--batch-size"),
        (["--imgsz", "0"], "--imgsz"),
    ],
)
def test_parse_args_rejects_invalid_inference_values(extra_args, message, capsys):
    base_args = [
        "--config",
        "model.yml",
        "--checkpoint",
        "model.pth",
        "--infer-img",
        "image.jpg",
    ]

    with pytest.raises(SystemExit):
        infer_cli.parse_args(base_args + extra_args)

    assert message in capsys.readouterr().err


def test_get_image_list_is_filtered_and_deterministic(tmp_path):
    (tmp_path / "b.PNG").write_bytes(b"image")
    (tmp_path / "a.jpg").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    images = infer_cli.get_image_list(infer_dir=tmp_path)

    assert [path.name for path in images] == ["a.jpg", "b.PNG"]


def test_create_preprocessors_uses_test_reader_and_does_not_mutate_config(
    tmp_path,
):
    cfg = _config_with_test_reader()
    image_path = tmp_path / "sample.png"
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    assert cv2.imwrite(str(image_path), image)

    sample_transform, batch_transform = infer_cli.create_preprocessors(
        cfg, image_size=16
    )
    batch = infer_cli.prepare_image_batch(
        [image_path],
        [7],
        sample_transform,
        batch_transform,
        torch.device("cpu"),
    )

    assert batch["image"].shape == (1, 3, 16, 16)
    assert batch["image"].dtype == torch.float32
    assert batch["image"][0, :, 0, 0].tolist() == pytest.approx(
        [30 / 255, 20 / 255, 10 / 255]
    )
    assert batch["im_shape"].tolist() == [[16.0, 16.0]]
    assert batch["scale_factor"][0].tolist() == pytest.approx([1.6, 0.8])
    assert batch["im_id"].tolist() == [[7]]
    assert cfg.TestReader["sample_transforms"][1]["Resize"]["target_size"] == [
        8,
        12,
    ]


def test_configure_input_size_keeps_model_cache_aligned():
    cfg = AttrDict(eval_size=[640, 640])

    infer_cli.configure_input_size(cfg, 608)

    assert cfg.eval_size == [608, 608]


def test_split_detections_uses_bbox_num_and_threshold():
    outputs = {
        "bbox": torch.tensor(
            [
                [2.0, 0.8, 1.0, 2.0, 5.0, 8.0],
                [4.0, 0.2, 3.0, 4.0, 6.0, 9.0],
                [7.0, 0.4, 0.0, 1.0, 2.0, 3.0],
            ]
        ),
        "bbox_num": torch.tensor([2, 1], dtype=torch.int32),
    }

    detections = infer_cli.split_detections(outputs, threshold=0.3)

    assert len(detections) == 2
    assert detections[0]["labels"].tolist() == [2]
    assert detections[0]["scores"].tolist() == pytest.approx([0.8])
    assert detections[0]["boxes"].tolist() == [[1.0, 2.0, 5.0, 8.0]]
    assert detections[1]["labels"].tolist() == [7]


def test_split_detections_allows_empty_threshold_results():
    outputs = {
        "bbox": torch.tensor([[2.0, 0.8, 1.0, 2.0, 5.0, 8.0]]),
        "bbox_num": torch.tensor([1], dtype=torch.int32),
    }

    detections = infer_cli.split_detections(outputs, threshold=1.0)

    assert detections[0]["labels"].numel() == 0
    assert detections[0]["scores"].numel() == 0
    assert detections[0]["boxes"].shape == (0, 4)


def test_split_detections_rejects_inconsistent_output():
    with pytest.raises(RuntimeError, match="bbox_num"):
        infer_cli.split_detections(
            {
                "bbox": torch.zeros((2, 6)),
                "bbox_num": torch.tensor([1]),
            }
        )


@pytest.mark.parametrize(
    ("bbox", "bbox_num", "message"),
    [
        (torch.zeros((1, 6)), torch.tensor([[1]]), "shape"),
        (torch.zeros((1, 6)), torch.tensor([1.0]), "integer counts"),
        (torch.zeros((0, 6)), torch.tensor([-1, 1]), "non-negative"),
    ],
)
def test_split_detections_rejects_invalid_bbox_counts(bbox, bbox_num, message):
    with pytest.raises(RuntimeError, match=message):
        infer_cli.split_detections({"bbox": bbox, "bbox_num": bbox_num})


def test_predict_images_passes_batch_dict_to_current_model(tmp_path):
    image_paths = [tmp_path / name for name in ("one.jpg", "two.jpg", "three.jpg")]
    observed_batch_sizes = []

    def sample_transform(sample):
        return {
            "image": np.ones((3, 4, 4), dtype=np.float32),
            "im_shape": np.array([4.0, 4.0], dtype=np.float32),
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "im_id": sample["im_id"],
        }

    class Model(torch.nn.Module):
        def forward(self, batch):
            batch_size = batch["image"].shape[0]
            observed_batch_sizes.append(batch_size)
            rows = torch.tensor(
                [[1.0, 0.9, 0.0, 0.0, 2.0, 2.0]],
                device=batch["image"].device,
            ).repeat(batch_size, 1)
            return {
                "bbox": rows,
                "bbox_num": torch.ones(
                    batch_size,
                    dtype=torch.int32,
                    device=batch["image"].device,
                ),
            }

    detections = infer_cli.predict_images(
        Model(),
        image_paths,
        sample_transform,
        default_collate_fn,
        torch.device("cpu"),
        batch_size=2,
        threshold=0.3,
    )

    assert observed_batch_sizes == [2, 1]
    assert len(detections) == 3
    assert all(item["labels"].tolist() == [1] for item in detections)


def test_predict_images_rejects_missing_detection_group(tmp_path):
    image_paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]

    def sample_transform(sample):
        return {
            "image": np.ones((3, 4, 4), dtype=np.float32),
            "im_shape": np.array([4.0, 4.0], dtype=np.float32),
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "im_id": sample["im_id"],
        }

    class Model(torch.nn.Module):
        def forward(self, batch):
            return {
                "bbox": torch.tensor([[1.0, 0.9, 0.0, 0.0, 2.0, 2.0]]),
                "bbox_num": torch.tensor([1]),
            }

    with pytest.raises(RuntimeError, match="batch size"):
        infer_cli.predict_images(
            Model(),
            image_paths,
            sample_transform,
            default_collate_fn,
            torch.device("cpu"),
            batch_size=2,
        )


def test_detections_to_records_uses_category_mapping(tmp_path):
    detections = [
        {
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.75]),
            "boxes": torch.tensor([[2.0, 3.0, 7.0, 11.0]]),
        }
    ]

    records = infer_cli.detections_to_records(
        [Path("image.jpg")],
        detections,
        {1: 17},
        {1: "cat"},
    )

    assert records == [
        {
            "image_id": 0,
            "image": "image.jpg",
            "category_id": 17,
            "category_name": "cat",
            "bbox": [2.0, 3.0, 5.0, 8.0],
            "score": pytest.approx(0.75),
        }
    ]


def test_detections_to_records_rejects_missing_group():
    with pytest.raises(RuntimeError, match="input images"):
        infer_cli.detections_to_records(
            [Path("one.jpg"), Path("two.jpg")],
            [],
            {},
            {},
        )


def test_build_model_wires_config_checkpoint_and_eval_mode(monkeypatch):
    cfg = AttrDict(
        architecture="FakeDetector",
        FakeDetector={"width": 8},
        TestReader={"sample_transforms": [{"Decode": {}}]},
    )
    observed = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.transforms = None

        def load_meanstd(self, transforms):
            self.transforms = transforms

    model = Model()

    def fake_create(model_config):
        observed["model_config"] = model_config
        return model

    def fake_load_weights(loaded_model, checkpoint_path, use_ema=False):
        observed["checkpoint"] = (loaded_model, checkpoint_path, use_ema)

    monkeypatch.setattr(infer_cli, "create", fake_create)
    monkeypatch.setattr(infer_cli, "load_evaluation_weights", fake_load_weights)

    result = infer_cli.build_model(
        cfg,
        "model.pth",
        torch.device("cpu"),
        use_ema=True,
    )

    assert result is model
    assert observed["model_config"] == {"width": 8, "name": "FakeDetector"}
    assert observed["checkpoint"] == (model, "model.pth", True)
    assert model.transforms == [{"Decode": {}}]
    assert model.training is False
    assert model.weight.device.type == "cpu"


def test_build_model_requires_architecture_block():
    with pytest.raises(ValueError, match="architecture block"):
        infer_cli.build_model(AttrDict(), "model.pth", torch.device("cpu"))


def test_onnx_inference_runner_reuses_session_and_maps_batch(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    sessions = []

    class FakeSession:
        def __init__(self, path, providers):
            sessions.append((path, providers, self))

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name=name,
                    shape=["batch", 3, 8, 12] if name == "image" else ["batch", 2],
                )
                for name in ("image", "im_shape", "scale_factor")
            ]

        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in ("bbox", "bbox_num")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, feed):
            assert output_names is None
            batch_size = feed["image"].shape[0]
            return [
                np.zeros((batch_size, 6), dtype=np.float32),
                np.ones((batch_size,), dtype=np.int32),
            ]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=FakeSession,
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )
    runner = infer_cli.OnnxInferenceRunner(model_path, torch.device("cpu"))
    batch = {
        "image": torch.zeros((2, 3, 8, 12)),
        "im_shape": torch.tensor([[8.0, 12.0], [8.0, 12.0]]),
        "scale_factor": torch.ones((2, 2)),
        "im_id": torch.tensor([[0], [1]]),
    }

    first = runner(batch)
    second = runner(batch)

    assert runner.eval() is runner
    assert len(sessions) == 1
    assert sessions[0][:2] == (
        str(model_path),
        ["CPUExecutionProvider"],
    )
    assert runner.device == torch.device("cpu")
    assert runner.providers == ("CPUExecutionProvider",)
    assert first["bbox"].shape == (2, 6)
    assert first["bbox_num"].tolist() == [1, 1]
    assert second["bbox_num"].tolist() == [1, 1]
    with pytest.raises(RuntimeError, match="expects fixed spatial size 8x12"):
        runner({**batch, "image": torch.zeros((2, 3, 7, 12))})


def test_onnx_inference_runner_selects_cuda_device_and_cpu_fallback(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")
    sessions = []

    class FakeSession:
        def __init__(self, path, providers):
            sessions.append((path, providers))

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name=name,
                    shape=["batch", 3, 8, 12] if name == "image" else ["batch", 2],
                )
                for name in ("image", "im_shape", "scale_factor")
            ]

        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in ("bbox", "bbox_num")]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=FakeSession,
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )

    runner = infer_cli.OnnxInferenceRunner(model_path, torch.device("cuda:1"))

    assert sessions == [
        (
            str(model_path),
            [
                (
                    "CUDAExecutionProvider",
                    {"device_id": 1, "use_tf32": 1},
                ),
                "CPUExecutionProvider",
            ],
        )
    ]
    assert runner.device == torch.device("cuda:1")
    assert runner.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_onnx_inference_runner_rejects_missing_cuda_provider(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")

    def unexpected_session(*args, **kwargs):
        raise AssertionError("session must not be created without the CUDA provider")

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=unexpected_session,
            get_available_providers=lambda: ["CPUExecutionProvider"],
        ),
    )

    with pytest.raises(RuntimeError, match="export-gpu"):
        infer_cli.OnnxInferenceRunner(model_path, torch.device("cuda"))


def test_onnx_inference_runner_rejects_session_cuda_fallback(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture")

    class FakeSession:
        def __init__(self, path, providers):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=FakeSession,
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )

    with pytest.raises(RuntimeError, match="fell back to CPU"):
        infer_cli.OnnxInferenceRunner(model_path, torch.device("cuda"))


def test_torchscript_inference_runner_maps_batch(tmp_path):
    class TensorOnlyModel(torch.nn.Module):
        def forward(self, image, im_shape, scale_factor):
            batch_size = image.shape[0]
            labels_and_scores = torch.zeros(
                (batch_size, 2), dtype=image.dtype, device=image.device
            )
            boxes = torch.cat((im_shape, scale_factor), dim=1)
            return (
                torch.cat((labels_and_scores, boxes), dim=1),
                torch.ones(batch_size, dtype=torch.int32, device=image.device),
            )

    model_path = tmp_path / "model.pt"
    example = (
        torch.zeros((1, 3, 8, 12)),
        torch.tensor([[8.0, 12.0]]),
        torch.ones((1, 2)),
    )
    torch.jit.save(
        torch.jit.trace(TensorOnlyModel(), example),
        str(model_path),
        _extra_files={
            TORCHSCRIPT_METADATA_FILE: json.dumps(
                {"schema_version": 1, "input_size": [8, 12]}
            )
        },
    )
    runner = infer_cli.TorchScriptInferenceRunner(model_path, torch.device("cpu"))
    batch = {
        "image": torch.zeros((2, 3, 8, 12)),
        "im_shape": torch.tensor([[8.0, 12.0], [8.0, 12.0]]),
        "scale_factor": torch.ones((2, 2)),
        "im_id": torch.tensor([[0], [1]]),
    }

    outputs = runner(batch)

    assert runner.eval() is runner
    assert outputs["bbox"].shape == (2, 6)
    assert outputs["bbox_num"].tolist() == [1, 1]
    with pytest.raises(RuntimeError, match="expects fixed spatial size 8x12"):
        runner({**batch, "image": torch.zeros((2, 3, 8, 11))})


def test_build_inference_runner_selects_exported_backend(monkeypatch):
    observed = []
    expected = object()
    monkeypatch.setattr(
        infer_cli,
        "OnnxInferenceRunner",
        lambda path, device: observed.append((path, device)) or expected,
    )
    args = SimpleNamespace(
        checkpoint=None,
        onnx_model="model.onnx",
        torchscript_model=None,
        use_ema=False,
    )

    result = infer_cli.build_inference_runner(
        AttrDict(),
        args,
        torch.device("cpu"),
    )

    assert result is expected
    assert observed == [("model.onnx", torch.device("cpu"))]


def test_main_keeps_onnx_cuda_preprocessing_on_cpu(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((8, 12, 3), dtype=np.uint8))
    output_directory = tmp_path / "results"
    observed = {}
    runner = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        infer_cli,
        "load_config",
        lambda path: AttrDict(eval_size=[8, 12]),
    )
    monkeypatch.setattr(infer_cli, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(
        infer_cli,
        "create_preprocessors",
        lambda cfg, image_size=None: (object(), object()),
    )

    def fake_build_runner(cfg, args, device):
        observed["runner_device"] = device
        return runner

    monkeypatch.setattr(infer_cli, "build_inference_runner", fake_build_runner)
    monkeypatch.setattr(
        infer_cli,
        "get_category_metadata",
        lambda cfg, annotation: ({0: 1}, {0: "person"}),
    )

    def fake_predict(
        model,
        image_paths,
        sample_transform,
        batch_transform,
        device,
        **kwargs,
    ):
        observed["preprocessing_device"] = device
        return [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "scores": torch.empty(0),
                "boxes": torch.empty((0, 4)),
            }
        ]

    monkeypatch.setattr(infer_cli, "predict_images", fake_predict)

    assert (
        infer_cli.main(
            [
                "--config",
                "model.yml",
                "--onnx-model",
                "model.onnx",
                "--infer-img",
                str(image_path),
                "--output-dir",
                str(output_directory),
                "--device",
                "cuda:1",
            ]
        )
        == 0
    )

    assert observed == {
        "runner_device": torch.device("cuda:1"),
        "preprocessing_device": torch.device("cpu"),
    }


def test_main_writes_visualization_and_machine_readable_results(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.zeros((20, 30, 3), dtype=np.uint8),
    )
    output_directory = tmp_path / "results"
    cfg = AttrDict(eval_size=[640, 640])
    detections = [
        {
            "labels": torch.tensor([0]),
            "scores": torch.tensor([0.75]),
            "boxes": torch.tensor([[2.0, 3.0, 12.0, 15.0]]),
        }
    ]

    monkeypatch.setattr(infer_cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(infer_cli, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(
        infer_cli,
        "create_preprocessors",
        lambda cfg, image_size=None: (object(), object()),
    )
    monkeypatch.setattr(
        infer_cli,
        "build_model",
        lambda cfg, checkpoint, device, use_ema=False: object(),
    )
    monkeypatch.setattr(
        infer_cli,
        "get_category_metadata",
        lambda cfg, annotation: ({0: 1}, {0: "person"}),
    )
    monkeypatch.setattr(
        infer_cli,
        "predict_images",
        lambda *args, **kwargs: detections,
    )

    assert (
        infer_cli.main(
            [
                "--config",
                "model.yml",
                "--checkpoint",
                "model.pth",
                "--infer-img",
                str(image_path),
                "--output-dir",
                str(output_directory),
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--threshold",
                "0.5",
                "--imgsz",
                "16",
                "--save-results",
            ]
        )
        == 0
    )

    assert cfg.eval_size == [16, 16]
    rendered = cv2.imread(str(output_directory / image_path.name))
    assert rendered is not None
    assert np.count_nonzero(rendered) > 0
    records = json.loads(
        (output_directory / "detections.json").read_text(encoding="utf-8")
    )
    assert records == [
        {
            "image_id": 0,
            "image": str(image_path),
            "category_id": 1,
            "category_name": "person",
            "bbox": [2.0, 3.0, 10.0, 12.0],
            "score": pytest.approx(0.75),
        }
    ]
