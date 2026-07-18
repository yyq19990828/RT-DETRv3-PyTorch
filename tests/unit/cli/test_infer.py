from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ppdet_pytorch.cli import infer as infer_cli
from ppdet_pytorch.core.workspace import AttrDict
from ppdet_pytorch.data.utils import default_collate_fn


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


def test_split_detections_rejects_inconsistent_output():
    with pytest.raises(RuntimeError, match="bbox_num"):
        infer_cli.split_detections(
            {
                "bbox": torch.zeros((2, 6)),
                "bbox_num": torch.tensor([1]),
            }
        )


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
