"""Run RT-DETRv3 inference with the repository's current data API."""

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from ppdet_pytorch import modeling as _modeling  # noqa: F401
from ppdet_pytorch.cli.eval import load_evaluation_weights
from ppdet_pytorch.core.workspace import create, load_config
from ppdet_pytorch.data.reader import BatchCompose, Compose
from ppdet_pytorch.data.source.category import get_categories
from ppdet_pytorch.utils.config import apply_overrides
from ppdet_pytorch.utils.logger import setup_logger


logger = setup_logger("infer")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_COLORS = (
    np.array(
        [
            [0.000, 0.447, 0.741],
            [0.850, 0.325, 0.098],
            [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556],
            [0.466, 0.674, 0.188],
            [0.301, 0.745, 0.933],
            [0.635, 0.078, 0.184],
            [1.000, 0.000, 0.000],
            [1.000, 0.500, 0.000],
            [0.000, 1.000, 0.000],
            [0.000, 0.000, 1.000],
            [0.667, 0.000, 1.000],
        ],
        dtype=np.float32,
    )
    * 255
)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="RT-DETRv3 inference")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)

    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument(
        "--infer-img",
        "--infer_img",
        dest="infer_img",
        help="Path to one image.",
    )
    image_group.add_argument(
        "--infer-dir",
        "--infer_dir",
        dest="infer_dir",
        help="Directory containing images (non-recursive).",
    )

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default="output/infer",
    )
    parser.add_argument(
        "--save-results",
        "--save_results",
        dest="save_results",
        action="store_true",
        help="Save thresholded detections to detections.json.",
    )
    parser.add_argument(
        "--threshold",
        "--draw-threshold",
        dest="threshold",
        type=float,
        default=0.3,
        help="Minimum score used for visualization and saved results.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Override the square Resize target in TestReader.",
    )
    parser.add_argument(
        "--anno-file",
        help="Optional annotation JSON/TXT used for category names.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights from a training checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("-o", "--override", nargs="*", default=[])
    return parser


def parse_args(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.imgsz is not None and args.imgsz < 1:
        parser.error("--imgsz must be at least 1")
    return args


def get_image_list(infer_dir=None, infer_img=None):
    """Return a deterministic image list for a single image or directory."""
    if infer_img is not None:
        image_path = Path(infer_img)
        if not image_path.is_file():
            raise FileNotFoundError("Inference image not found: {}".format(image_path))
        if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError("Unsupported image suffix: {}".format(image_path.suffix))
        return [image_path]

    image_directory = Path(infer_dir)
    if not image_directory.is_dir():
        raise NotADirectoryError(
            "Inference directory not found: {}".format(image_directory)
        )
    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError("No supported images found in {}".format(image_directory))
    return image_paths


def _test_transforms(cfg, image_size=None):
    if "TestReader" not in cfg or "sample_transforms" not in cfg.TestReader:
        raise ValueError("Config must define TestReader.sample_transforms")
    transforms = deepcopy(cfg.TestReader["sample_transforms"])
    if image_size is not None:
        for transform in transforms:
            if "Resize" in transform:
                transform["Resize"]["target_size"] = [image_size, image_size]
                break
        else:
            raise ValueError("--imgsz requires a Resize transform in TestReader")
    return transforms


def create_preprocessors(cfg, image_size=None):
    transforms = _test_transforms(cfg, image_size=image_size)
    num_classes = int(cfg.get("num_classes", 80))
    sample_transform = Compose(transforms, num_classes=num_classes)
    batch_transform = BatchCompose(
        cfg.TestReader.get("batch_transforms", []),
        num_classes=num_classes,
        collate_batch=True,
    )
    return sample_transform, batch_transform


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (np.ndarray, np.generic)):
        return torch.as_tensor(value, device=device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def prepare_image_batch(
    image_paths,
    image_ids,
    sample_transform,
    batch_transform,
    device,
):
    samples = []
    for image_path, image_id in zip(image_paths, image_ids):
        sample = {
            "im_file": str(image_path),
            "im_id": np.asarray([image_id], dtype=np.int64),
        }
        samples.append(sample_transform(sample))
    return _move_to_device(batch_transform(samples), device)


def split_detections(outputs, threshold=0.0):
    """Split the model's concatenated ``bbox`` output by ``bbox_num``."""
    if not isinstance(outputs, Mapping) or not {"bbox", "bbox_num"} <= set(outputs):
        raise RuntimeError("Model must return bbox and bbox_num")

    bboxes = torch.as_tensor(outputs["bbox"]).detach().cpu()
    bbox_num = torch.as_tensor(outputs["bbox_num"]).detach().cpu().tolist()
    if bboxes.ndim != 2 or bboxes.shape[1] != 6:
        raise RuntimeError("bbox must have shape [N, 6]")
    if sum(int(count) for count in bbox_num) != len(bboxes):
        raise RuntimeError("bbox_num does not match the number of bbox rows")

    detections = []
    start = 0
    for count in bbox_num:
        rows = bboxes[start : start + int(count)]
        start += int(count)
        keep = (rows[:, 0] >= 0) & (rows[:, 1] >= threshold)
        rows = rows[keep]
        detections.append(
            {
                "labels": rows[:, 0].to(torch.int64),
                "scores": rows[:, 1],
                "boxes": rows[:, 2:6],
            }
        )
    return detections


@torch.inference_mode()
def predict_images(
    model,
    image_paths,
    sample_transform,
    batch_transform,
    device,
    batch_size=1,
    threshold=0.3,
):
    model.eval()
    all_detections = []
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        image_ids = range(start, start + len(paths))
        batch = prepare_image_batch(
            paths,
            image_ids,
            sample_transform,
            batch_transform,
            device,
        )
        all_detections.extend(split_detections(model(batch), threshold))
    return all_detections


def _resolve_annotation_file(cfg, annotation_file=None):
    if annotation_file is not None:
        annotation_path = Path(annotation_file)
        if not annotation_path.is_file():
            raise FileNotFoundError(
                "Annotation file not found: {}".format(annotation_path)
            )
        return str(annotation_path)

    dataset = cfg.get("TestDataset", {})
    configured_path = dataset.get("anno_path")
    if not configured_path:
        return None
    annotation_path = Path(configured_path)
    if not annotation_path.is_absolute():
        annotation_path = Path(dataset.get("dataset_dir", "")) / annotation_path
    return str(annotation_path) if annotation_path.is_file() else None


def get_category_metadata(cfg, annotation_file=None):
    annotation_file = _resolve_annotation_file(cfg, annotation_file)
    clsid2catid, catid2name = get_categories(
        cfg.get("metric", "COCO"),
        anno_file=annotation_file,
    )
    class_names = {
        int(class_id): catid2name.get(category_id, str(category_id))
        for class_id, category_id in clsid2catid.items()
    }
    return clsid2catid, class_names


def visualize_detections(image, detections, class_names):
    visualized = image.copy()
    for box, score, label in zip(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        label_id = int(label)
        color = _COLORS[label_id % len(_COLORS)].astype(int).tolist()
        cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)
        text = "{}: {:.2f}".format(
            class_names.get(label_id, str(label_id)),
            float(score),
        )
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1 - 5, text_size[1] + 5)
        cv2.rectangle(
            visualized,
            (x1, text_y - text_size[1] - 5),
            (x1 + text_size[0], text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            visualized,
            text,
            (x1, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return visualized


def detections_to_records(image_paths, detections, clsid2catid, class_names):
    records = []
    for image_id, (image_path, image_detections) in enumerate(
        zip(image_paths, detections)
    ):
        for box, score, label in zip(
            image_detections["boxes"],
            image_detections["scores"],
            image_detections["labels"],
        ):
            label_id = int(label)
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            records.append(
                {
                    "image_id": image_id,
                    "image": str(image_path),
                    "category_id": int(clsid2catid[label_id]),
                    "category_name": class_names[label_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return records


def build_model(cfg, checkpoint_path, device, use_ema=False):
    architecture = cfg.get("architecture")
    if not architecture or architecture not in cfg:
        raise ValueError("Config must define an architecture block")
    model_config = dict(cfg[architecture])
    model_config["name"] = architecture
    model = create(model_config)
    model.load_meanstd(cfg.TestReader["sample_transforms"])
    load_evaluation_weights(model, checkpoint_path, use_ema=use_ema)
    model.to(device)
    model.eval()
    return model


def main(argv=None):
    args = parse_args(argv)
    image_paths = get_image_list(args.infer_dir, args.infer_img)

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")

    sample_transform, batch_transform = create_preprocessors(cfg, image_size=args.imgsz)
    model = build_model(
        cfg,
        args.checkpoint,
        device,
        use_ema=args.use_ema,
    )
    clsid2catid, class_names = get_category_metadata(cfg, args.anno_file)

    logger.info(
        "Running inference on %d image(s), batch_size=%d, device=%s",
        len(image_paths),
        args.batch_size,
        device,
    )
    detections = predict_images(
        model,
        image_paths,
        sample_transform,
        batch_transform,
        device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    for image_path, image_detections in zip(image_paths, detections):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(
                "Failed to decode image for output: {}".format(image_path)
            )
        output_path = output_directory / image_path.name
        if not cv2.imwrite(
            str(output_path),
            visualize_detections(image, image_detections, class_names),
        ):
            raise RuntimeError(
                "Failed to write inference image: {}".format(output_path)
            )
        logger.info(
            "Processed %s: %d detection(s) -> %s",
            image_path,
            len(image_detections["boxes"]),
            output_path,
        )

    if args.save_results:
        records = detections_to_records(
            image_paths,
            detections,
            clsid2catid,
            class_names,
        )
        results_path = output_directory / "detections.json"
        results_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d detection record(s) to %s", len(records), results_path)

    logger.info("Inference complete. Results saved to %s", output_directory)
    return 0


if __name__ == "__main__":
    main()
