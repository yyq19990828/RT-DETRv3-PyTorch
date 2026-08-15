"""Evaluate RT-DETRv3 checkpoints with the repository's current data API."""

import argparse
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager

import torch
from tqdm import tqdm

from detrs.core.workspace import load_config
from detrs.engine import Trainer
from detrs.metrics import COCOMetric, Metric, YOLOMetric
from detrs.utils.config import apply_overrides
from detrs.utils.logger import setup_logger

logger = setup_logger("eval")

_DERIVED_BUFFER_KEYS = {
    "aux_o2m_head.anchor_points",
    "aux_o2m_head.stride_tensor",
}
_COCO_METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="RT-DETRv3 COCO evaluation")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anno-file", "--anno_file", dest="anno_file")
    parser.add_argument("--image-dir", "--image_dir", dest="image_dir")
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=4
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        help="Keep COCO prediction files in this directory.",
    )
    parser.add_argument(
        "--use-ema",
        "--use_ema",
        dest="use_ema",
        action="store_true",
        help="Evaluate the EMA state stored in a training checkpoint.",
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
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def _get_ema_state_dict(checkpoint):
    if "ema" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain EMA weights")

    ema = checkpoint["ema"]
    if isinstance(ema, dict) and "module" in ema:
        state_dict = ema["module"]
        if (
            not isinstance(state_dict, dict)
            or not state_dict
            or not all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in state_dict.items()
            )
        ):
            raise RuntimeError("Upstream EMA module must be a tensor state dict")
        return state_dict
    if not isinstance(ema, dict) or "ema_state_dict" not in ema:
        return ema

    state_dict = ema["ema_state_dict"]
    step = int(ema.get("step", 0))
    decay_type = ema.get("ema_decay_type", "exponential")
    if step == 0 or decay_type == "exponential":
        return state_dict

    if "ema_black_list" not in ema:
        raise RuntimeError(
            "Cannot apply bias correction to this legacy EMA checkpoint: "
            "ema_black_list is missing"
        )
    correction = 1 - float(ema["current_decay"]) ** step
    black_list = set(ema["ema_black_list"])
    return {
        key: value if key in black_list else value / correction
        for key, value in state_dict.items()
    }


def load_evaluation_weights(model, checkpoint_path, use_ema=False):
    """Load a model checkpoint and reject unknown state-dict differences."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if use_ema:
        state_dict = _get_ema_state_dict(checkpoint)
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=False)
    unknown_missing = set(incompatible.missing_keys) - _DERIVED_BUFFER_KEYS
    unknown_unexpected = set(incompatible.unexpected_keys)
    if unknown_missing or unknown_unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the evaluation model: "
            "missing={}, unexpected={}".format(
                sorted(unknown_missing), sorted(unknown_unexpected)
            )
        )
    if incompatible.missing_keys:
        logger.info(
            "Regenerated derived buffers not stored in checkpoint: %s",
            sorted(incompatible.missing_keys),
        )
    logger.info(
        "Loaded %s weights from %s", "EMA" if use_ema else "model", checkpoint_path
    )


@torch.no_grad()
def evaluate(model, data_loader, metric, prepare_batch, device):
    model.eval()
    logger.info(
        "Starting evaluation on %d batches with %s",
        len(data_loader),
        device,
    )
    for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
        batch = prepare_batch(batch)
        outputs = model(batch)
        metric.update(batch, outputs)

    metric.accumulate()
    return metric.get_results()


def _configure_dataset(cfg, anno_file=None, image_dir=None):
    if str(cfg.get("metric", "COCO")).upper() == "YOLO":
        logger.info(
            "YOLO metric evaluates against the dataset labels; "
            "ignoring --anno-file/--image-dir overrides."
        )
        return
    if anno_file is None and image_dir is None:
        return

    dataset_dir = Path(cfg.EvalDataset.get("dataset_dir", ""))
    if anno_file is None:
        configured_anno = cfg.EvalDataset["anno_path"]
        if isinstance(configured_anno, (list, tuple)):
            raise ValueError(
                "EvalDataset anno_path must be a single annotation file; "
                "list-valued anno_path is only supported for training. "
                "Pass --anno-file explicitly to override."
            )
        anno_file = dataset_dir / configured_anno
    if image_dir is None:
        image_dir = dataset_dir / cfg.EvalDataset["image_dir"]

    cfg.EvalDataset["dataset_dir"] = "."
    cfg.EvalDataset["anno_path"] = str(Path(anno_file).resolve())
    cfg.EvalDataset["image_dir"] = str(Path(image_dir).resolve())


def _format_results(raw_results):
    formatted = {}
    for metric_type, stats in raw_results.items():
        formatted[metric_type] = {
            name: float(value) for name, value in zip(_COCO_METRIC_NAMES, stats)
        }
    return formatted


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    _configure_dataset(cfg, args.anno_file, args.image_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    cfg.EvalReader["batch_size"] = args.batch_size
    cfg.worker_num = args.num_workers
    cfg.device = device
    cfg.use_gpu = device.type == "cuda"
    cfg.use_ema = False

    output_context: ContextManager[str]
    if args.output_dir:
        output_directory = Path(args.output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_context = nullcontext(str(output_directory))
    else:
        output_context = tempfile.TemporaryDirectory(prefix="detrs-eval-")

    with output_context as evaluation_directory:
        cfg.save_dir = evaluation_directory
        trainer = Trainer(cfg, mode="eval")
        load_evaluation_weights(
            trainer.model,
            args.checkpoint,
            use_ema=args.use_ema,
        )

        metric: Metric
        if str(cfg.get("metric", "COCO")).upper() == "YOLO":
            metric = YOLOMetric(
                trainer.dataset,
                output_eval=evaluation_directory,
            )
        else:
            if isinstance(trainer.dataset.anno_path, (list, tuple)):
                raise ValueError(
                    "EvalDataset anno_path must be a single annotation file; "
                    "list-valued anno_path is only supported for training."
                )
            annotation_path = os.path.join(
                trainer.dataset.dataset_dir,
                trainer.dataset.anno_path,
            )
            metric = COCOMetric(
                annotation_path,
                output_eval=evaluation_directory,
            )
        raw_results = evaluate(
            trainer.model,
            trainer.loader,
            metric,
            trainer._prepare_batch,
            device,
        )
        if args.output_dir:
            logger.info("Kept evaluation outputs in %s", evaluation_directory)

    results = _format_results(raw_results)
    for metric_type, values in results.items():
        logger.info("%s metrics:", metric_type.upper())
        for name, value in values.items():
            logger.info("  %-5s: %.6f", name, value)
    return 0


if __name__ == "__main__":
    main()
