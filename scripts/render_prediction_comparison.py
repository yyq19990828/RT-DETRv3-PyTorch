#!/usr/bin/env python3
"""Render and summarize Paddle/PyTorch predictions with one visualizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Prediction:
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]
    score: float


COLORS = (
    (189, 114, 0),
    (25, 83, 217),
    (32, 177, 237),
    (142, 47, 126),
    (48, 172, 119),
    (238, 190, 77),
    (47, 20, 162),
    (0, 0, 255),
    (0, 128, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 170),
)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a common-renderer comparison from Paddle and PyTorch COCO JSON."
        )
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--paddle-results", required=True, type=Path)
    parser.add_argument("--pytorch-results", required=True, type=Path)
    parser.add_argument("--paddle-checkpoint", required=True, type=Path)
    parser.add_argument("--pytorch-checkpoint", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--render-threshold", type=float, default=0.5)
    parser.add_argument("--box-tolerance", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--model", default="RT-DETRv3-R18")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, logical_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": logical_name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_categories(path: Path) -> dict[int, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    categories = document.get("categories") if isinstance(document, dict) else None
    if not isinstance(categories, list):
        raise ValueError("annotation JSON must contain a categories list")
    return {
        int(category["id"]): str(category["name"])
        for category in categories
        if isinstance(category, dict) and "id" in category and "name" in category
    }


def _load_predictions(
    path: Path,
    categories: dict[int, str],
    threshold: float,
) -> list[Prediction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"prediction JSON must contain a list: {path}")
    predictions = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"prediction row must be an object: {path}")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"prediction bbox must contain four values: {path}")
        score = float(row["score"])
        if score < threshold:
            continue
        category_id = int(row["category_id"])
        x, y, width, height = (float(value) for value in bbox)
        predictions.append(
            Prediction(
                category_id=category_id,
                category_name=categories.get(category_id, str(category_id)),
                bbox=(x, y, width, height),
                score=score,
            )
        )
    return sorted(
        predictions,
        key=lambda item: (-item.score, item.category_id, item.bbox),
    )


def _xyxy(prediction: Prediction) -> tuple[float, float, float, float]:
    x, y, width, height = prediction.bbox
    return x, y, x + width, y + height


def _box_difference(left: Prediction, right: Prediction) -> float:
    return max(abs(a - b) for a, b in zip(_xyxy(left), _xyxy(right)))


def match_predictions(
    paddle_predictions: list[Prediction],
    pytorch_predictions: list[Prediction],
    box_tolerance: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    unused_pytorch = set(range(len(pytorch_predictions)))
    matches = []
    unmatched_paddle = []
    for paddle_index, paddle_prediction in enumerate(paddle_predictions):
        candidates = [
            pytorch_index
            for pytorch_index in unused_pytorch
            if pytorch_predictions[pytorch_index].category_id
            == paddle_prediction.category_id
        ]
        if not candidates:
            unmatched_paddle.append(paddle_index)
            continue
        pytorch_index = min(
            candidates,
            key=lambda index: (
                _box_difference(paddle_prediction, pytorch_predictions[index]),
                abs(paddle_prediction.score - pytorch_predictions[index].score),
                index,
            ),
        )
        pytorch_prediction = pytorch_predictions[pytorch_index]
        box_difference = _box_difference(paddle_prediction, pytorch_prediction)
        if box_difference > box_tolerance:
            unmatched_paddle.append(paddle_index)
            continue
        unused_pytorch.remove(pytorch_index)
        matches.append(
            {
                "paddle_index": paddle_index,
                "pytorch_index": pytorch_index,
                "category_id": paddle_prediction.category_id,
                "score_abs_diff": abs(
                    paddle_prediction.score - pytorch_prediction.score
                ),
                "box_linf_px": box_difference,
            }
        )
    return matches, unmatched_paddle, sorted(unused_pytorch)


def _prediction_record(prediction: Prediction) -> dict[str, Any]:
    return {
        "category_id": prediction.category_id,
        "category_name": prediction.category_name,
        "bbox_xywh": list(prediction.bbox),
        "score": prediction.score,
    }


def _draw_predictions(
    image: np.ndarray,
    predictions: list[Prediction],
    threshold: float,
) -> np.ndarray:
    rendered = image.copy()
    for prediction in reversed(predictions):
        if prediction.score < threshold:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in _xyxy(prediction)]
        color = COLORS[prediction.category_id % len(COLORS)]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction.category_name} {prediction.score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        text_y = max(y1, text_height + 6)
        cv2.rectangle(
            rendered,
            (x1, text_y - text_height - 6),
            (x1 + text_width + 4, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _render_comparison(
    image: np.ndarray,
    paddle_predictions: list[Prediction],
    pytorch_predictions: list[Prediction],
    *,
    render_threshold: float,
    score_threshold: float,
    matches: list[dict[str, Any]],
) -> np.ndarray:
    panels = [
        _draw_predictions(image, paddle_predictions, render_threshold),
        _draw_predictions(image, pytorch_predictions, render_threshold),
    ]
    height, width = image.shape[:2]
    title_height = 44
    footer_height = 42
    canvas = np.full(
        (height + title_height + footer_height, width * 2, 3), 24, dtype=np.uint8
    )
    titles = ("Paddle original (.pdparams)", "PyTorch converted (.pth)")
    for index, (title, panel) in enumerate(zip(titles, panels)):
        start_x = index * width
        canvas[title_height : title_height + height, start_x : start_x + width] = panel
        cv2.putText(
            canvas,
            title,
            (start_x + 16, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    max_score_diff = max(
        (float(match["score_abs_diff"]) for match in matches), default=0.0
    )
    max_box_diff = max((float(match["box_linf_px"]) for match in matches), default=0.0)
    footer = (
        f"render score >= {render_threshold:.2f} | matched {len(matches)} at score "
        f">= {score_threshold:.2f} | max score diff {max_score_diff:.2e} | "
        f"max box diff {max_box_diff:.6f}px"
    )
    cv2.putText(
        canvas,
        footer,
        (16, title_height + height + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argument_parser().parse_args(argv)
    if not 0.0 <= args.score_threshold <= args.render_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= score <= render <= 1")
    if args.box_tolerance < 0:
        raise ValueError("--box-tolerance must be non-negative")

    categories = _load_categories(args.annotations)
    paddle_predictions = _load_predictions(
        args.paddle_results, categories, args.score_threshold
    )
    pytorch_predictions = _load_predictions(
        args.pytorch_results, categories, args.score_threshold
    )
    matches, unmatched_paddle, unmatched_pytorch = match_predictions(
        paddle_predictions, pytorch_predictions, args.box_tolerance
    )
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"cannot decode image: {args.image}")

    comparison = _render_comparison(
        image,
        paddle_predictions,
        pytorch_predictions,
        render_threshold=args.render_threshold,
        score_threshold=args.score_threshold,
        matches=matches,
    )
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_image), comparison):
        raise RuntimeError(f"failed to write image: {args.output_image}")

    max_score_diff = max(
        (float(match["score_abs_diff"]) for match in matches), default=0.0
    )
    max_box_diff = max((float(match["box_linf_px"]) for match in matches), default=0.0)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "model": args.model,
            "device": "CPU",
            "dtype": "FP32",
            "input_size": [args.input_size, args.input_size],
            "score_threshold": args.score_threshold,
            "render_threshold": args.render_threshold,
            "box_match_tolerance_px": args.box_tolerance,
            "renderer": "scripts/render_prediction_comparison.py",
        },
        "inputs": {
            "image": _file_record(args.image, f"COCO val2017/{args.image.name}"),
            "annotations": _file_record(
                args.annotations, "COCO annotations/instances_val2017.json"
            ),
            "paddle_checkpoint": _file_record(
                args.paddle_checkpoint,
                f"pretrained_models/paddle/{args.paddle_checkpoint.name}",
            ),
            "pytorch_checkpoint": _file_record(
                args.pytorch_checkpoint,
                f"pretrained_models/pytorch/{args.pytorch_checkpoint.name}",
            ),
        },
        "comparison": {
            "paddle_prediction_count": len(paddle_predictions),
            "pytorch_prediction_count": len(pytorch_predictions),
            "matched_count": len(matches),
            "unmatched_paddle_indices": unmatched_paddle,
            "unmatched_pytorch_indices": unmatched_pytorch,
            "max_score_abs_diff": max_score_diff,
            "max_box_linf_px": max_box_diff,
            "matches": matches,
        },
        "predictions": {
            "paddle": [_prediction_record(item) for item in paddle_predictions],
            "pytorch": [_prediction_record(item) for item in pytorch_predictions],
        },
        "comparison_image": _file_record(
            args.output_image,
            "docs/archive/rtdetrv3-v0.1.0/reports/assets/"
            f"{args.output_image.name}",
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    printable_comparison = dict(summary["comparison"])
    printable_comparison.pop("matches")
    print(json.dumps(printable_comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
