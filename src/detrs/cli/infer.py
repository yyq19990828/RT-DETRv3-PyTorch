"""Run RT-DETRv3 inference with the repository's current data API."""

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from detrs import modeling as _modeling  # noqa: F401
from detrs.cli.eval import load_evaluation_weights
from detrs.core.workspace import create, load_config
from detrs.data.reader import BatchCompose, Compose
from detrs.data.source.category import get_categories
from detrs.deploy import TORCHSCRIPT_METADATA_FILE
from detrs.utils.cli import DetrsHelpFormatter
from detrs.utils.config import apply_overrides
from detrs.utils.console import get_console
from detrs.utils.logger import setup_logger

logger = setup_logger("infer")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_TENSOR_INPUT_NAMES = ("image", "im_shape", "scale_factor")
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
    parser = argparse.ArgumentParser(
        description="RT-DETRv3 inference",
        formatter_class=DetrsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint")
    model_group.add_argument(
        "--onnx-model",
        help="Run a tensor-only ONNX export with ONNX Runtime CPU or CUDA.",
    )
    model_group.add_argument(
        "--torchscript-model",
        help="Run a tensor-only traced TorchScript export on a PyTorch device.",
    )

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
        default=None,
    )
    parser.add_argument("-o", "--override", nargs="*", default=[])
    return parser


def parse_args(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    onnx_model = args.onnx_model is not None
    exported_model = args.onnx_model is not None or args.torchscript_model is not None
    if args.device is None:
        args.device = (
            "cpu" if onnx_model else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError):
        parser.error("--device must be a valid PyTorch device")
    if onnx_model and device.type not in ("cpu", "cuda"):
        parser.error("ONNX Infer supports only --device cpu or cuda[:id]")
    if exported_model and args.use_ema:
        parser.error("--use-ema is only valid with --checkpoint")
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


def configure_input_size(cfg, image_size=None):
    """Keep model caches aligned with a TestReader Resize override."""
    if image_size is not None:
        cfg.eval_size = [image_size, image_size]


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


def _tensor_inputs(batch):
    if not isinstance(batch, Mapping):
        raise RuntimeError("Inference batch must be a mapping")
    missing = [name for name in _TENSOR_INPUT_NAMES if name not in batch]
    if missing:
        raise RuntimeError("Inference batch is missing: {}".format(", ".join(missing)))
    return tuple(torch.as_tensor(batch[name]) for name in _TENSOR_INPUT_NAMES)


def _model_file(path, model_type):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError("{} model not found: {}".format(model_type, model_path))
    return model_path


def _fixed_spatial_shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return None
    height, width = shape[-2:]
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (height, width)
    ):
        return None
    return height, width


def _validate_spatial_shape(inputs, expected, model_type):
    image = inputs[0]
    if image.ndim != 4:
        raise RuntimeError(
            "{} image input must have shape [B, C, H, W]".format(model_type)
        )
    actual = (int(image.shape[-2]), int(image.shape[-1]))
    if expected is not None and actual != expected:
        raise RuntimeError(
            "{} model expects fixed spatial size {}x{}, got {}x{}".format(
                model_type,
                expected[0],
                expected[1],
                actual[0],
                actual[1],
            )
        )


def _torchscript_input_size(raw_metadata):
    if not raw_metadata:
        return None
    try:
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(raw_metadata)
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("TorchScript export metadata is invalid") from error
    if not isinstance(metadata, Mapping):
        raise RuntimeError("TorchScript export metadata must be an object")
    if metadata.get("schema_version") != 1:
        raise RuntimeError("Unsupported TorchScript export metadata schema")
    input_size = metadata.get("input_size")
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in input_size
        )
    ):
        raise RuntimeError("TorchScript export metadata input_size is invalid")
    return tuple(input_size)


class OnnxInferenceRunner:
    """Adapt one reusable ONNX Runtime session to the Infer batch contract."""

    def __init__(self, model_path, device):
        import onnxruntime as ort

        self.model_path = _model_file(model_path, "ONNX")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "ONNX Runtime CUDAExecutionProvider is unavailable; install "
                    "the GPU runtime with `uv sync --extra export-gpu` and verify "
                    "CUDA/cuDNN compatibility"
                )
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self.device.index or 0,
                        "use_tf32": 1,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.providers = tuple(self.session.get_providers())
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(
                "ONNX Runtime CUDA session fell back to CPU; verify the "
                "onnxruntime-gpu, CUDA, and cuDNN versions"
            )
        input_metadata = self.session.get_inputs()
        input_names = tuple(value.name for value in input_metadata)
        output_names = tuple(value.name for value in self.session.get_outputs())
        if set(input_names) != set(_TENSOR_INPUT_NAMES):
            raise RuntimeError(
                "ONNX inputs must be image, im_shape, and scale_factor; got {}".format(
                    input_names
                )
            )
        if set(output_names) != {"bbox", "bbox_num"}:
            raise RuntimeError(
                "ONNX outputs must be bbox and bbox_num; got {}".format(output_names)
            )
        self.input_names = input_names
        self.output_names = output_names
        image_metadata = next(
            value for value in input_metadata if value.name == "image"
        )
        self.input_size = _fixed_spatial_shape(image_metadata.shape)

    def eval(self):
        return self

    def __call__(self, batch):
        tensor_inputs = _tensor_inputs(batch)
        _validate_spatial_shape(tensor_inputs, self.input_size, "ONNX")
        inputs = dict(zip(_TENSOR_INPUT_NAMES, tensor_inputs))
        feed = {name: inputs[name].detach().cpu().numpy() for name in self.input_names}
        values = self.session.run(None, feed)
        outputs = dict(zip(self.output_names, values))
        return {
            "bbox": torch.from_numpy(np.asarray(outputs["bbox"])),
            "bbox_num": torch.from_numpy(np.asarray(outputs["bbox_num"])),
        }


class TorchScriptInferenceRunner:
    """Adapt one loaded tensor-only TorchScript module to the Infer batch contract."""

    def __init__(self, model_path, device):
        self.model_path = _model_file(model_path, "TorchScript")
        self.device = torch.device(device)
        extra_files = {TORCHSCRIPT_METADATA_FILE: b""}
        self.model = torch.jit.load(
            str(self.model_path),
            map_location=self.device,
            _extra_files=extra_files,
        ).eval()
        self.input_size = _torchscript_input_size(
            extra_files[TORCHSCRIPT_METADATA_FILE]
        )

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, batch):
        inputs = _tensor_inputs(batch)
        _validate_spatial_shape(inputs, self.input_size, "TorchScript")
        inputs = tuple(value.to(self.device) for value in inputs)
        outputs = self.model(*inputs)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError("TorchScript model must return bbox and bbox_num")
        return {"bbox": outputs[0], "bbox_num": outputs[1]}


def split_detections(outputs, threshold=0.0):
    """Split the model's concatenated ``bbox`` output by ``bbox_num``."""
    if not isinstance(outputs, Mapping) or not {"bbox", "bbox_num"} <= set(outputs):
        raise RuntimeError("Model must return bbox and bbox_num")

    bboxes = torch.as_tensor(outputs["bbox"]).detach().cpu()
    bbox_num_tensor = torch.as_tensor(outputs["bbox_num"]).detach().cpu()
    if bboxes.ndim != 2 or bboxes.shape[1] != 6:
        raise RuntimeError("bbox must have shape [N, 6]")
    if bbox_num_tensor.ndim != 1:
        raise RuntimeError("bbox_num must have shape [B]")
    if bbox_num_tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError("bbox_num must contain integer counts")
    bbox_num = [int(count) for count in bbox_num_tensor.tolist()]
    if any(count < 0 for count in bbox_num):
        raise RuntimeError("bbox_num counts must be non-negative")
    if sum(bbox_num) != len(bboxes):
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
        batch_detections = split_detections(model(batch), threshold)
        if len(batch_detections) != len(paths):
            raise RuntimeError("bbox_num length does not match inference batch size")
        all_detections.extend(batch_detections)
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
    if clsid2catid is None:
        raise ValueError("Keypoint category metadata is not supported by Infer CLI")
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
    if len(image_paths) != len(detections):
        raise RuntimeError("Detection groups do not match the number of input images")
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


def build_inference_runner(cfg, args, device):
    """Build the one model source selected by the Infer CLI."""
    if args.checkpoint is not None:
        return build_model(
            cfg,
            args.checkpoint,
            device,
            use_ema=args.use_ema,
        )
    if args.onnx_model is not None:
        return OnnxInferenceRunner(args.onnx_model, device)
    return TorchScriptInferenceRunner(args.torchscript_model, device)


def main(argv=None):
    args = parse_args(argv)
    image_paths = get_image_list(args.infer_dir, args.infer_img)

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    configure_input_size(cfg, args.imgsz)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")

    sample_transform, batch_transform = create_preprocessors(cfg, image_size=args.imgsz)
    model = build_inference_runner(cfg, args, device)
    clsid2catid, class_names = get_category_metadata(cfg, args.anno_file)
    backend = (
        "checkpoint"
        if args.checkpoint is not None
        else ("onnx" if args.onnx_model is not None else "torchscript")
    )

    provider_note = (
        ", providers={}".format(",".join(model.providers))
        if isinstance(model, OnnxInferenceRunner)
        else ""
    )
    console = get_console()
    console.print(
        "[bold]detrs infer[/bold] · backend=[cyan]{}[/cyan] · {} image(s) · "
        "batch_size={} · device=[cyan]{}[/cyan]{}".format(
            backend,
            len(image_paths),
            args.batch_size,
            device,
            provider_note,
        )
    )
    preprocessing_device = (
        torch.device("cpu") if args.onnx_model is not None else device
    )
    detections = predict_images(
        model,
        image_paths,
        sample_transform,
        batch_transform,
        preprocessing_device,
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
        console.print(
            "Processed [bold]{}[/bold]: [green]{}[/green] detection(s) -> {}".format(
                image_path, len(image_detections["boxes"]), output_path
            )
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
        console.print(
            "Saved {} detection record(s) to {}".format(len(records), results_path)
        )

    console.print(
        "[green]Inference complete.[/green] Results saved to {}".format(
            output_directory
        )
    )
    return 0


if __name__ == "__main__":
    main()
