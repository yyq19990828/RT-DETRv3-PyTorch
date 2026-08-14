#!/usr/bin/env python3
"""Run the manifest-bound model-family validation matrix."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from validation_common import (
    APPROVE,
    BLOCKED,
    DEFAULT_PLAN,
    EXIT_CODES,
    FAIL,
    DriverArgumentParser,
    normalized_plan_identity,
    parse_csv,
    publish_json,
    require_sha256,
    result_document,
    run_main,
)

FAMILIES = {
    "rtdetrv3": ("r18", "r34", "r50"),
    "dfine": ("n", "s", "m", "l", "x"),
    "deim-dfine": ("n", "s", "m", "l", "x"),
    "deim-rtdetrv2": ("s", "m", "m-star", "l", "x"),
    "rtdetrv4": ("s", "m", "l", "x"),
    "deimv2": ("x", "l", "m", "s", "n", "pico", "femto", "atto"),
}
PHASES = (
    "verify",
    "checkpoint-parity",
    "train-resume",
    "coco",
    "eval",
    "infer",
    "export",
    "teacher",
    "teacher-preflight",
)
FAMILY_PHASES = {
    "deim-dfine": (
        "verify",
        "checkpoint-parity",
        "train-resume",
        "eval",
        "infer",
        "export",
        "coco",
    ),
    "deim-rtdetrv2": (
        "verify",
        "checkpoint-parity",
        "train-resume",
        "eval",
        "infer",
        "export",
        "coco",
    ),
    "dfine": (
        "verify",
        "checkpoint-parity",
        "train-resume",
        "eval",
        "infer",
        "export",
        "coco",
    ),
    "rtdetrv3": ("verify", "eval", "infer", "export"),
    "rtdetrv4": PHASES,
    # deimv2: the infer phase's four-image upstream parity helper is a
    # follow-up; inference is covered by coco/export phases and the model
    # documentation's val2017 evidence.
    "deimv2": (
        "verify",
        "checkpoint-parity",
        "train-resume",
        "eval",
        "coco",
        "export",
    ),
}
NEGATIVES = (
    "missing-teacher",
    "bad-checksum",
    "wrong-size",
    "wrong-family",
    "missing-stage1",
)
DFINE_RUNTIME_PHASES = {"verify", "train-resume", "eval", "infer", "coco", "export"}
DEIM_DFINE_RUNTIME_PHASES = set(FAMILY_PHASES["deim-dfine"])
DEIM_RTDETRV2_RUNTIME_PHASES = set(FAMILY_PHASES["deim-rtdetrv2"])
RTDETRV4_RUNTIME_PHASES = set(FAMILY_PHASES["rtdetrv4"])
DFINE_PARITY_IMAGES = {
    "000000000139.jpg": "ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d",
    "000000000285.jpg": "f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d",
    "000000000632.jpg": "a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550",
    "000000000724.jpg": "5c0e559c75d3969c8e3e297b61f61063f78045c9d4802b526ba616361f3823fd",
}
COCO_VAL2017_ANNOTATION_SHA256 = (
    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
)


def _family_config(family, variant):
    if family == "dfine":
        return f"configs/dfine/dfine_hgnetv2_{variant}_coco.yml"
    if family == "deim-dfine":
        return f"configs/deim/dfine/deim_hgnetv2_{variant}_coco.yml"
    if family == "rtdetrv4":
        return f"configs/rtdetrv4/rtdetrv4_hgnetv2_{variant}_coco.yml"
    if family == "deimv2":
        branch = "dinov3" if variant in {"x", "l", "m", "s"} else "hgnetv2"
        return f"configs/deimv2/deimv2_{branch}_{variant}_coco.yml"
    names = {
        "s": "r18vd_120e",
        "m": "r34vd_120e",
        "m-star": "r50vd_m_60e",
        "l": "r50vd_60e",
        "x": "r101vd_60e",
    }
    return f"configs/deim/rtdetrv2/deim_{names[variant]}_coco.yml"


def create_parser() -> argparse.ArgumentParser:
    parser = DriverArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--phase", default="verify")
    parser.add_argument("--negative")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--dinov3-weights", type=Path)
    parser.add_argument("--dinov3-sha256")
    parser.add_argument("--installed-prefix", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Validate and emit the request matrix without constructing models.",
    )
    return parser


def _selected_variants(family: str, selection: str) -> list[str]:
    available = FAMILIES[family]
    if selection == "all":
        return list(available)
    if selection == "smallest":
        return [available[0]]
    selected = parse_csv(selection, "variants")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            "unknown variants for {}: {}".format(family, ",".join(unknown))
        )
    return selected


def _validate_installed_prefix(prefix: Path) -> None:
    prefix = prefix.resolve()
    if not (prefix / "pyvenv.cfg").is_file():
        raise ValueError(
            "installed prefix is not a virtual environment: {}".format(prefix)
        )
    candidates = sorted((prefix / "lib").glob("python*/site-packages"))
    if not candidates:
        raise ValueError("installed prefix has no site-packages: {}".format(prefix))
    spec = importlib.machinery.PathFinder.find_spec(
        "detrs", [str(path) for path in candidates]
    )
    if spec is None or spec.origin is None:
        raise ValueError("installed prefix does not contain detrs: {}".format(prefix))
    origin = Path(spec.origin).resolve()
    if not any(
        origin == path.resolve() or path.resolve() in origin.parents
        for path in candidates
    ):
        raise ValueError(
            "installed mode resolved an import outside site-packages: {}".format(origin)
        )


def _preflight_assets(args: argparse.Namespace, phases: list[str]) -> list[str]:
    missing = []
    if set(phases) & {
        "checkpoint-parity",
        "train-resume",
        "coco",
        "eval",
        "infer",
        "export",
    }:
        if args.checkpoint_root is None or not args.checkpoint_root.is_dir():
            missing.append("checkpoint_root")
    if set(phases) & {"train-resume", "coco"}:
        required = ("train2017", "val2017", "annotations")
        if args.coco_root is None:
            missing.append("coco_root")
        else:
            for name in required:
                if not (args.coco_root / name).exists():
                    missing.append("coco_root/{}".format(name))
    if set(phases) & {"teacher", "teacher-preflight"}:
        if args.dinov3_repo is None or not args.dinov3_repo.is_dir():
            missing.append("dinov3_repo")
        if args.dinov3_weights is None or not args.dinov3_weights.is_file():
            missing.append("dinov3_weights")
        if not args.dinov3_sha256:
            missing.append("dinov3_sha256")
        else:
            require_sha256(args.dinov3_sha256, "DINOv3 weights")
    return sorted(set(missing))


def _dfine_runtime_preflight(args, phases, variants):
    from dfine_checkpoint_parity import (
        DEFAULT_MANIFEST,
        load_manifest,
        preflight_artifact,
        sha256_file,
        verify_upstream_checkout,
    )

    manifest = load_manifest(args.manifest or DEFAULT_MANIFEST)
    for variant in variants:
        preflight_artifact(args.checkpoint_root, variant, manifest)
    upstream_root = None
    image_paths = []
    if "infer" in phases:
        upstream_value = os.environ.get("DFINE_UPSTREAM_ROOT")
        if not upstream_value or not Path(upstream_value).is_dir():
            raise FileNotFoundError("DFINE_UPSTREAM_ROOT is required for infer parity")
        upstream_root = Path(upstream_value)
        verify_upstream_checkout(upstream_root)
        for filename, expected_sha in DFINE_PARITY_IMAGES.items():
            path = args.coco_root / "val2017" / filename
            if not path.is_file():
                raise FileNotFoundError("missing D-FINE parity image: {}".format(path))
            if sha256_file(path) != expected_sha:
                raise ValueError(
                    "D-FINE parity image SHA-256 mismatch: {}".format(path)
                )
            image_paths.append(path)
    if "coco" in phases:
        annotation = args.coco_root / "annotations/instances_val2017.json"
        if not annotation.is_file():
            raise FileNotFoundError(
                "missing COCO val2017 annotation: {}".format(annotation)
            )
        if sha256_file(annotation) != COCO_VAL2017_ANNOTATION_SHA256:
            raise ValueError("COCO val2017 annotation SHA-256 mismatch")
        annotations = json.loads(annotation.read_text(encoding="utf-8"))
        expected_images = {item["file_name"] for item in annotations["images"]}
        actual_images = {
            path.name for path in (args.coco_root / "val2017").glob("*.jpg")
        }
        if len(expected_images) != 5000 or actual_images != expected_images:
            raise ValueError(
                "COCO val2017 image set mismatch: missing={}, unexpected={}".format(
                    len(expected_images - actual_images),
                    len(actual_images - expected_images),
                )
            )
    return manifest, upstream_root, image_paths


def _deim_runtime_preflight(args, phases, variants, family):
    parity_module = {
        "deim-dfine": "deim_dfine_checkpoint_parity",
        "deim-rtdetrv2": "deim_rtdetrv2_checkpoint_parity",
        "rtdetrv4": "rtdetrv4_checkpoint_parity",
        "deimv2": "deimv2_checkpoint_parity",
    }[family]
    module = importlib.import_module(parity_module)
    DEFAULT_MANIFEST = module.DEFAULT_MANIFEST
    load_manifest = module.load_manifest
    preflight_artifact = module.preflight_artifact
    sha256_file = module.sha256_file
    verify_upstream_checkout = module.verify_upstream_checkout

    manifest = load_manifest(args.manifest or DEFAULT_MANIFEST)
    for variant in variants:
        preflight_artifact(args.checkpoint_root, variant, manifest)
    if family == "deim-rtdetrv2" and "train-resume" in phases:
        pretrained_value = os.environ.get("DEIM_RTDETRV2_PRETRAINED_ROOT")
        if not pretrained_value or not Path(pretrained_value).is_dir():
            raise FileNotFoundError(
                "DEIM_RTDETRV2_PRETRAINED_ROOT is required for train-resume"
            )
        pretrained_root = Path(pretrained_value)
        for entry in manifest["pretrained_backbones"].values():
            path = pretrained_root / entry["filename"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing PResNet pretrained checkpoint: {path}"
                )
            if path.stat().st_size != entry["source_size_bytes"]:
                raise ValueError(f"PResNet pretrained checkpoint size mismatch: {path}")
            if sha256_file(path) != entry["source_sha256"]:
                raise ValueError(
                    f"PResNet pretrained checkpoint SHA-256 mismatch: {path}"
                )
    upstream_root = None
    image_paths = []
    if set(phases) & {"checkpoint-parity", "infer"}:
        if family == "rtdetrv4":
            upstream_name = "RTDETRV4_UPSTREAM_ROOT"
        elif family == "deimv2":
            upstream_name = "DEIMV2_UPSTREAM_ROOT"
        else:
            upstream_name = "DEIM_UPSTREAM_ROOT"
        upstream_value = os.environ.get(upstream_name)
        if not upstream_value or not Path(upstream_value).is_dir():
            raise FileNotFoundError(
                "{} is required for {} parity".format(upstream_name, family)
            )
        upstream_root = Path(upstream_value)
        verify_upstream_checkout(upstream_root)
    if "infer" in phases:
        for filename, expected_sha in DFINE_PARITY_IMAGES.items():
            path = args.coco_root / "val2017" / filename
            if not path.is_file():
                raise FileNotFoundError(
                    "missing {} parity image: {}".format(family, path)
                )
            if sha256_file(path) != expected_sha:
                raise ValueError(
                    "{} parity image SHA-256 mismatch: {}".format(family, path)
                )
            image_paths.append(path)
    if "coco" in phases:
        annotation = args.coco_root / "annotations/instances_val2017.json"
        if not annotation.is_file():
            raise FileNotFoundError(
                "missing COCO val2017 annotation: {}".format(annotation)
            )
        if sha256_file(annotation) != COCO_VAL2017_ANNOTATION_SHA256:
            raise ValueError("COCO val2017 annotation SHA-256 mismatch")
        annotations = json.loads(annotation.read_text(encoding="utf-8"))
        expected_images = {item["file_name"] for item in annotations["images"]}
        actual_images = {
            path.name for path in (args.coco_root / "val2017").glob("*.jpg")
        }
        if len(expected_images) != 5000 or actual_images != expected_images:
            raise ValueError("COCO val2017 image set mismatch")
    return manifest, upstream_root, image_paths


def _run_dfine_train_resume():
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_dfine_runtime.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
    if completed.returncode:
        raise RuntimeError("D-FINE reduced train/resume integration tests failed")
    return {
        "checks": [
            {
                "claim": "reduced-runtime-and-epoch-boundary-resume",
                "convergence_claim": False,
                "device": "cpu",
                "dtype": "float32",
                "name": "dfine-runtime-pytest",
                "status": APPROVE,
            }
        ],
        "name": "train-resume",
        "status": APPROVE,
    }


def _run_deim_dfine_train_resume():
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_deim_dfine_runtime.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
    if completed.returncode:
        raise RuntimeError("DEIM-D-FINE reduced train/resume tests failed")
    return {
        "checks": [
            {
                "claim": "reduced-runtime-and-epoch-boundary-resume",
                "convergence_claim": False,
                "device": "cpu",
                "dtype": "float32",
                "name": "deim-dfine-runtime-pytest",
                "status": APPROVE,
            }
        ],
        "name": "train-resume",
        "status": APPROVE,
    }


def _run_deim_rtdetrv2_train_resume():
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_deim_rtdetrv2_runtime.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
    if completed.returncode:
        raise RuntimeError("DEIM-RT-DETRv2 reduced train/resume tests failed")
    return {
        "checks": [
            {
                "claim": "reduced-runtime-and-epoch-boundary-resume",
                "convergence_claim": False,
                "device": "cpu",
                "dtype": "float32",
                "name": "deim-rtdetrv2-runtime-pytest",
                "status": APPROVE,
            }
        ],
        "name": "train-resume",
        "status": APPROVE,
    }


def _run_rtdetrv4_train_resume(args):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_rtdetrv4_runtime.py",
        "-q",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "DINOV3_REPO": str(args.dinov3_repo),
            "DINOV3_WEIGHTS": str(args.dinov3_weights),
            "DINOV3_WEIGHTS_SHA256": args.dinov3_sha256,
        }
    )
    completed = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], env=environment
    )
    if completed.returncode:
        raise RuntimeError("RT-DETRv4 real-teacher train/resume tests failed")
    return {
        "checks": [
            {
                "claim": "real-teacher-reduced-update-and-gam-resume",
                "convergence_claim": False,
                "device": "cpu",
                "dtype": "float32",
                "name": "rtdetrv4-runtime-pytest",
                "status": APPROVE,
            }
        ],
        "name": "train-resume",
        "status": APPROVE,
    }


def _run_deim_eval(variant, checkpoint, family="deim-dfine"):
    import torch

    from detrs.cli.infer import build_model
    from detrs.core.workspace import load_config

    config_path = _family_config(family, variant)
    config = load_config(config_path)
    model = build_model(
        config,
        checkpoint,
        torch.device("cpu"),
        use_ema=family == "rtdetrv4",
    ).eval()
    image = torch.rand(1, 3, 640, 640, generator=torch.Generator().manual_seed(0))
    batch = {
        "image": image,
        "im_shape": torch.full((1, 2), 640.0),
        "scale_factor": torch.ones(1, 2),
    }
    with torch.inference_mode():
        prediction = model(batch)
    if set(prediction) != {"bbox", "bbox_num"}:
        raise AssertionError(f"{family} eager prediction contract mismatch")
    expected_count = int(config.DETRPostProcess.get("num_top_queries", 100))
    if prediction["bbox_num"].tolist() != [expected_count]:
        raise AssertionError(f"{family} eager prediction count mismatch")
    if not torch.isfinite(prediction["bbox"]).all():
        raise AssertionError(f"{family} eager prediction is non-finite")
    return {
        "checks": [
            {
                "bbox_count": expected_count,
                "device": "cpu",
                "dtype": "float32",
                "name": "eager-fixed-640",
                "status": APPROVE,
            }
        ],
        "name": "eval",
        "status": APPROVE,
    }


def _run_dfine_infer(
    variant,
    checkpoint,
    upstream_root,
    image_paths,
    evidence_dir,
    *,
    family="dfine",
):
    from dfine_checkpoint_parity import sha256_file, validate_real_images
    from PIL import Image

    if family == "deim-dfine":
        from deim_dfine_checkpoint_parity import validate_real_images
    elif family == "deim-rtdetrv2":
        from deim_rtdetrv2_checkpoint_parity import validate_real_images
    elif family == "rtdetrv4":
        from rtdetrv4_checkpoint_parity import validate_real_images

    output_dir = evidence_dir / f"{family}-infer" / variant
    fixture_dir = evidence_dir / f"{family}-four-images" / variant
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        shutil.copyfile(image_path, fixture_dir / image_path.name)
    command = [
        sys.executable,
        "-m",
        "detrs.cli.infer",
        "-c",
        _family_config(family, variant),
        "--checkpoint",
        str(checkpoint),
        "--infer-dir",
        str(fixture_dir),
        "--output-dir",
        str(output_dir),
        "--save-results",
        "--threshold",
        "0.3",
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]
    if family == "rtdetrv4":
        command.append("--use-ema")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    rendered = sorted(output_dir.glob("*.jpg"))
    records = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    if len(rendered) != 4 or not records:
        raise AssertionError("D-FINE infer must emit four renders and non-empty JSON")
    input_sizes = {path.name: Image.open(path).size for path in image_paths}
    if any(Image.open(path).size != input_sizes[path.name] for path in rendered):
        raise AssertionError("D-FINE render dimensions do not match source images")
    for record in records:
        if set(record) != {
            "image_id",
            "image",
            "category_id",
            "category_name",
            "bbox",
            "score",
        }:
            raise AssertionError("D-FINE inference JSON fields are invalid")
        values = [record["score"], *record["bbox"]]
        if (
            len(record["bbox"]) != 4
            or record["score"] < 0.3
            or not all(math.isfinite(value) for value in values)
        ):
            raise AssertionError("D-FINE inference JSON contains invalid values")
    parity = validate_real_images(
        variant,
        checkpoint.parent,
        upstream_root,
        image_paths,
    )
    return {
        "checks": [
            {
                "detection_count": len(records),
                "name": "eager-json-render",
                "render_count": len(rendered),
                "render_sha256": {path.name: sha256_file(path) for path in rendered},
                "status": APPROVE,
            },
            {
                "images": parity["images"],
                "input": parity["input"],
                "name": "four-image-raw-output-parity",
                "outputs": parity["checks"],
                "status": APPROVE,
            },
        ],
        "name": "infer",
        "status": APPROVE,
    }


def _run_dfine_coco(
    variant,
    checkpoint,
    coco_root,
    expected_ap,
    evidence_dir,
    *,
    family="dfine",
):
    import pycocotools
    import torch
    from dfine_checkpoint_parity import sha256_file
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    output_dir = evidence_dir / f"{family}-coco" / variant
    if output_dir.exists():
        shutil.rmtree(output_dir)
    annotation = coco_root / "annotations/instances_val2017.json"
    device = "cuda" if family == "rtdetrv4" and torch.cuda.is_available() else "cpu"
    batch_size = "16" if device == "cuda" else "4"
    command = [
        sys.executable,
        "-m",
        "detrs.cli.eval",
        "-c",
        _family_config(family, variant),
        "--checkpoint",
        str(checkpoint),
        "--anno-file",
        str(annotation),
        "--image-dir",
        str(coco_root / "val2017"),
        "--batch-size",
        batch_size,
        "--num-workers",
        "4",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    if family == "rtdetrv4":
        command.append("--use-ema")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    prediction = output_dir / "bbox.json"
    records = json.loads(prediction.read_text(encoding="utf-8"))
    coco = COCO(str(annotation))
    evaluation = COCOeval(coco, coco.loadRes(str(prediction)), "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    observed_ap = float(evaluation.stats[0])
    error = abs(observed_ap - expected_ap)
    if error > 0.001:
        raise AssertionError(
            "D-FINE {} COCO AP mismatch: observed={}, expected={}".format(
                variant, observed_ap, expected_ap
            )
        )
    return {
        "checks": [
            {
                "absolute_error": error,
                "annotation_sha256": COCO_VAL2017_ANNOTATION_SHA256,
                "image_count": 5000,
                "name": "coco-val2017",
                "observed_bbox_ap": observed_ap,
                "official_bbox_ap": expected_ap,
                "checkpoint_sha256": sha256_file(checkpoint),
                "device": device,
                "dtype": "float32",
                "prediction_count": len(records),
                "prediction_sha256": sha256_file(prediction),
                "python_version": sys.version.split()[0],
                "pycocotools_version": getattr(pycocotools, "__version__", "unknown"),
                "pytorch_version": torch.__version__,
                "status": APPROVE,
                "tolerance": 0.001,
            }
        ],
        "name": "coco",
        "status": APPROVE,
    }


def _run_dfine_export(variant, checkpoint, evidence_dir, *, family="dfine"):
    import onnx
    import torch

    from detrs.cli.infer import build_model
    from detrs.core.workspace import load_config
    from detrs.deploy import (
        DetectionExportAdapter,
        export_onnx,
        export_torchscript,
        make_example_inputs,
        run_onnx,
        run_torchscript,
        validate_detection_outputs,
    )

    output_dir = evidence_dir / f"{family}-export" / variant
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    config_path = _family_config(family, variant)
    cfg = load_config(config_path)
    # DEIMv2 tiny variants export at their own fixed input sizes.
    export_size = (
        (320, 320)
        if family == "deimv2" and variant == "atto"
        else ((416, 416) if family == "deimv2" and variant == "femto" else (640, 640))
    )
    cfg.eval_size = list(export_size)
    cfg.eval_spatial_size = list(export_size)
    model = build_model(
        cfg,
        checkpoint,
        torch.device("cpu"),
        use_ema=family == "rtdetrv4",
    ).eval()
    adapter = DetectionExportAdapter(model).eval()
    output_tolerances = (
        {
            "score_atol": 4e-4 if variant == "x" else 2e-5,
            "box_atol": 0.1,
        }
        if family == "deim-rtdetrv2"
        else {}
    )
    deploy_tolerances = (
        {"score_atol": 3e-5 if variant == "x" else 2e-5, "box_atol": 0.02}
        if family == "deim-rtdetrv2"
        else {}
    )
    single_inputs = make_example_inputs(1, *export_size)
    inputs_by_batch = {
        1: single_inputs,
        4: tuple(value.expand(4, *value.shape[1:]).clone() for value in single_inputs),
    }
    with torch.inference_mode():
        before_deploy = adapter(*inputs_by_batch[1])
    if not hasattr(model, "deploy"):
        raise AssertionError("D-FINE model does not expose deploy conversion")
    if model.deploy() is not model or model.deploy() is not model:
        raise AssertionError("D-FINE deploy conversion must be idempotent")
    with torch.inference_mode():
        references = {
            batch: adapter(*inputs) for batch, inputs in inputs_by_batch.items()
        }
    deploy_metrics = validate_detection_outputs(
        before_deploy, references[1], **deploy_tolerances
    )

    onnx_path = output_dir / "model.onnx"
    torchscript_path = output_dir / "model.torchscript.pt"
    try:
        export_onnx(adapter, inputs_by_batch[1], onnx_path, opset_version=17)
        export_torchscript(adapter, inputs_by_batch[1], torchscript_path)
        graph = onnx.load(str(onnx_path))
        if [(item.domain, item.version) for item in graph.opset_import] != [("", 17)]:
            raise AssertionError("D-FINE ONNX graph must use exact opset 17")
        input_shapes = {
            value.name: [
                dimension.dim_param or dimension.dim_value
                for dimension in value.type.tensor_type.shape.dim
            ]
            for value in graph.graph.input
        }
        if input_shapes != {
            "image": ["batch", 3, 640, 640],
            "im_shape": ["batch", 2],
            "scale_factor": ["batch", 2],
        }:
            raise AssertionError(
                "D-FINE ONNX input shape contract mismatch: {}".format(input_shapes)
            )
        graph_names = [node.name for node in graph.graph.node] + [
            item.name for item in graph.graph.initializer
        ]
        graph_residues = (
            "criterion",
            "denoising",
            "dn_",
            "aux_outputs",
            "teacher",
            "distill",
            "feature_projector",
        )
        if any(
            token in name.lower() for name in graph_names for token in graph_residues
        ):
            raise AssertionError("D-FINE ONNX graph contains training-only residue")
        scripted = torch.jit.load(str(torchscript_path), map_location="cpu")
        scripted_names = [name for name, _ in scripted.named_parameters()] + [
            name for name, _ in scripted.named_buffers()
        ]
        scripted_graph = str(scripted.inlined_graph).lower()
        parameter_residues = (
            ("criterion", "dn_", "aux_outputs")
            if family == "deim-rtdetrv2"
            else graph_residues
        )
        if any(
            token in name.lower()
            for name in scripted_names
            for token in parameter_residues
        ) or any(token in scripted_graph for token in graph_residues):
            raise AssertionError("D-FINE TorchScript contains training-only residue")

        checks = [
            {
                "format": "deploy-eager",
                "name": "deploy-conversion-parity",
                "status": APPROVE,
                **deploy_metrics,
            }
        ]
        for export_format, runner, path in (
            ("onnx", run_onnx, onnx_path),
            ("torchscript", run_torchscript, torchscript_path),
        ):
            for batch in (1, 4):
                tolerances = (
                    output_tolerances
                    if export_format == "onnx"
                    else {
                        "score_atol": 0.0,
                        "box_atol": 0.0,
                    }
                )
                metrics = validate_detection_outputs(
                    references[batch],
                    runner(path, inputs_by_batch[batch]),
                    **tolerances,
                )
                checks.append(
                    {
                        "batch_size": batch,
                        "format": export_format,
                        "name": "{}-batch-{}-parity".format(export_format, batch),
                        "status": APPROVE,
                        **metrics,
                    }
                )
        return {
            "checks": checks,
            "graph": {
                "dynamic_batch": True,
                "fixed_spatial_size": [640, 640],
                "opset": 17,
                "training_residue": False,
            },
            "name": "export",
            "status": APPROVE,
        }
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _run_dinov3_teacher_preflight(args):
    import torch

    from detrs.modeling.teachers.dinov3 import DINOv3TeacherModel

    weights = args.dinov3_weights.resolve()
    teacher = DINOv3TeacherModel(
        dinov3_repo_path=str(args.dinov3_repo.resolve()),
        dinov3_weights_path=str(weights),
        weights_filename=weights.name,
        weights_size_bytes=weights.stat().st_size,
        weights_sha256=args.dinov3_sha256,
    )
    images = torch.linspace(0.0, 1.0, steps=3 * 64 * 64).reshape(1, 3, 64, 64)
    features = teacher(images)
    if features.shape != (1, 768, 2, 2):
        raise ValueError(
            "DINOv3 teacher feature boundary mismatch: {}".format(tuple(features.shape))
        )
    if features.requires_grad or not torch.isfinite(features).all():
        raise ValueError("DINOv3 teacher features must be finite and detached")
    if teacher.training or teacher.model.training:
        raise ValueError("DINOv3 teacher must remain in eval mode")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("DINOv3 teacher parameters must be frozen")
    return {
        "checks": [
            {
                "embed_dim": 768,
                "feature_shape": list(features.shape),
                "name": "pinned-hub-feature-boundary",
                "patch_size": 16,
                "status": APPROVE,
            },
            {
                "filename": weights.name,
                "name": "authorized-weight-identity",
                "sha256": args.dinov3_sha256,
                "size_bytes": weights.stat().st_size,
                "status": APPROVE,
            },
            {
                "detached": True,
                "eval": True,
                "name": "frozen-eval-no-grad",
                "status": APPROVE,
            },
        ],
        "name": "teacher-preflight",
        "status": APPROVE,
    }


def _single_family_command(args, family, phases, output):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--family",
        family,
        "--variants",
        args.variants,
        "--phase",
        ",".join(phases),
        "--plan",
        str(args.plan),
        "--evidence-dir",
        str(args.evidence_dir),
        "--output",
        str(output),
    ]
    for option, value in (
        ("--checkpoint-root", args.checkpoint_root),
        ("--coco-root", args.coco_root),
        ("--dinov3-repo", args.dinov3_repo),
        ("--dinov3-weights", args.dinov3_weights),
        ("--dinov3-sha256", args.dinov3_sha256),
        ("--installed-prefix", args.installed_prefix),
        ("--manifest", args.manifest),
    ):
        if value is not None:
            command.extend((option, str(value)))
    return command


def _invoke_single_family(args, family, phases, output):
    completed = subprocess.run(
        _single_family_command(args, family, phases, output),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    return result_document(
        normalized_plan_identity(args.plan),
        family_results=[
            {
                "family": family,
                "phases": [],
                "reason": (completed.stdout + completed.stderr).strip(),
                "status": FAIL,
                "variant": variant,
            }
            for variant in _selected_variants(family, args.variants)
        ],
        negatives=[],
        status=FAIL,
    )


def _expect_artifact_preflight_failure(
    checkpoint_root, family, variant, mutation, *, source_family=None
):
    modules = {
        "dfine": "dfine_checkpoint_parity",
        "deim-dfine": "deim_dfine_checkpoint_parity",
        "deim-rtdetrv2": "deim_rtdetrv2_checkpoint_parity",
        "rtdetrv4": "rtdetrv4_checkpoint_parity",
    }
    module = importlib.import_module(modules[family])
    manifest = module.load_manifest(module.DEFAULT_MANIFEST)
    entry = manifest["models"][variant]
    source = checkpoint_root / entry["filename"]
    with tempfile.TemporaryDirectory(prefix="model-family-negative-") as directory:
        mutated_root = Path(directory)
        target = mutated_root / entry["filename"]
        if source_family is not None:
            source_module = importlib.import_module(modules[source_family])
            source_manifest = source_module.load_manifest(
                source_module.DEFAULT_MANIFEST
            )
            source_variant = FAMILIES[source_family][0]
            source_entry = source_manifest["models"][source_variant]
            source = checkpoint_root / source_entry["filename"]
        shutil.copyfile(source, target)
        if mutation == "bad-checksum":
            with target.open("r+b") as checkpoint:
                first = checkpoint.read(1)
                if not first:
                    raise ValueError("cannot checksum-mutate an empty checkpoint")
                checkpoint.seek(0)
                checkpoint.write(bytes([first[0] ^ 0xFF]))
        elif mutation == "wrong-size":
            target.write_bytes(b"wrong-size")
        try:
            module.preflight_artifact(mutated_root, variant, manifest)
        except (FileNotFoundError, ValueError):
            return
    raise AssertionError(f"{mutation} was not rejected by artifact preflight")


def _run_negatives(args, selected):
    results = []
    targets = {
        "missing-teacher": ("rtdetrv4", FAMILIES["rtdetrv4"][0]),
        "bad-checksum": ("dfine", FAMILIES["dfine"][0]),
        "wrong-size": ("dfine", FAMILIES["dfine"][0]),
        "wrong-family": ("dfine", FAMILIES["dfine"][0]),
        "missing-stage1": ("dfine", FAMILIES["dfine"][0]),
    }
    for mutation in selected:
        family, variant = targets[mutation]
        try:
            if mutation == "missing-teacher":
                negative_args = argparse.Namespace(**vars(args))
                negative_args.dinov3_repo = None
                negative_args.dinov3_weights = None
                missing = _preflight_assets(negative_args, ["teacher"])
                if not {"dinov3_repo", "dinov3_weights"} <= set(missing):
                    raise AssertionError("missing teacher assets passed preflight")
            elif mutation == "missing-stage1":
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/integration/test_dfine_runtime.py::test_rejects_missing_stage1_without_state_mutation",
                        "-q",
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                )
                if completed.returncode:
                    raise RuntimeError("missing-stage1 rejection test failed")
            else:
                _expect_artifact_preflight_failure(
                    args.checkpoint_root,
                    family,
                    variant,
                    mutation,
                    source_family="deim-dfine" if mutation == "wrong-family" else None,
                )
            result = {
                "family": family,
                "mutation": mutation,
                "mutation_boundary": "preflight",
                "state_mutated": False,
                "status": APPROVE,
                "variant": variant,
            }
        except Exception as error:
            result = {
                "family": family,
                "mutation": mutation,
                "reason": str(error),
                "status": FAIL,
                "variant": variant,
            }
        results.append(result)
    return results


def _run_multi_family(args, families, phases, negatives, identity):
    family_results = []
    status = APPROVE
    with tempfile.TemporaryDirectory(prefix="model-family-receipts-") as directory:
        for family in families:
            supported_phases = [
                phase for phase in phases if phase in FAMILY_PHASES[family]
            ]
            receipt = _invoke_single_family(
                args,
                family,
                supported_phases,
                Path(directory) / f"{family}.json",
            )
            family_results.extend(receipt["family_results"])
            if receipt["status"] != APPROVE:
                status = FAIL
    negative_results = _run_negatives(args, negatives)
    if any(result["status"] != APPROVE for result in negative_results):
        status = FAIL
    document = result_document(
        identity,
        family_results=family_results,
        negatives=negative_results,
        status=status,
    )
    document["command"] = [sys.executable, *sys.argv]
    document["git_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = args.output or args.evidence_dir / "final-F3-user-qa.json"
    publish_json(output, document)
    print(json.dumps(document, allow_nan=False, sort_keys=True))
    return EXIT_CODES[status]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    families = parse_csv(args.family, "family")
    unknown_families = sorted(set(families) - set(FAMILIES))
    if unknown_families:
        raise ValueError("unknown families: {}".format(",".join(unknown_families)))
    phases = parse_csv(args.phase, "phase")
    if phases == ["all"]:
        if len(families) == 1:
            phases = list(FAMILY_PHASES[families[0]])
        else:
            phases = list(PHASES)
    unknown_phases = sorted(set(phases) - set(PHASES))
    if unknown_phases:
        raise ValueError("unknown phases: {}".format(",".join(unknown_phases)))
    negatives = parse_csv(args.negative, "negative") if args.negative else []
    unknown_negatives = sorted(set(negatives) - set(NEGATIVES))
    if unknown_negatives:
        raise ValueError("unknown negatives: {}".format(",".join(unknown_negatives)))
    if args.installed_prefix:
        _validate_installed_prefix(args.installed_prefix)
    identity = normalized_plan_identity(args.plan)
    matrix = [
        (family, variant)
        for family in families
        for variant in _selected_variants(family, args.variants)
    ]
    if not args.contract_only and len(families) > 1:
        return _run_multi_family(args, families, phases, negatives, identity)
    if not args.contract_only:
        dfine_runtime = families == ["dfine"] and set(phases) <= DFINE_RUNTIME_PHASES
        deim_dfine_runtime = families == ["deim-dfine"] and set(phases) <= (
            DEIM_DFINE_RUNTIME_PHASES
        )
        deim_rtdetrv2_runtime = families == ["deim-rtdetrv2"] and set(phases) <= (
            DEIM_RTDETRV2_RUNTIME_PHASES
        )
        rtdetrv4_teacher = (
            families == ["rtdetrv4"]
            and bool(phases)
            and set(phases) <= {"teacher", "teacher-preflight"}
        )
        rtdetrv4_runtime = (
            families == ["rtdetrv4"]
            and set(phases) <= RTDETRV4_RUNTIME_PHASES
            and not rtdetrv4_teacher
        )
        if (
            dfine_runtime
            or deim_dfine_runtime
            or deim_rtdetrv2_runtime
            or rtdetrv4_runtime
        ):
            missing = []
            if args.checkpoint_root is None or not args.checkpoint_root.is_dir():
                missing.append("checkpoint_root")
            if set(phases) & {"infer", "coco"} and (
                args.coco_root is None or not args.coco_root.is_dir()
            ):
                missing.append("coco_root")
            if rtdetrv4_runtime and set(phases) & {
                "train-resume",
                "teacher",
                "teacher-preflight",
            }:
                if args.dinov3_repo is None or not args.dinov3_repo.is_dir():
                    missing.append("dinov3_repo")
                if args.dinov3_weights is None or not args.dinov3_weights.is_file():
                    missing.append("dinov3_weights")
                if not args.dinov3_sha256:
                    missing.append("dinov3_sha256")
        else:
            missing = _preflight_assets(args, phases)
        if families == ["dfine"] and phases == ["checkpoint-parity"]:
            upstream_value = os.environ.get("DFINE_UPSTREAM_ROOT")
            if not upstream_value or not Path(upstream_value).is_dir():
                missing.append("DFINE_UPSTREAM_ROOT")
            if missing:
                print("BLOCKED: missing required assets: {}".format(", ".join(missing)))
                return EXIT_CODES[BLOCKED]
            from dfine_checkpoint_parity import DEFAULT_MANIFEST, validate_variant

            manifest = args.manifest or DEFAULT_MANIFEST
            family_results = []
            status = APPROVE
            for _, variant in matrix:
                try:
                    result = validate_variant(
                        variant,
                        args.checkpoint_root,
                        Path(upstream_value),
                        manifest,
                    )
                    phase = {
                        "checks": result.pop("checks"),
                        "name": "checkpoint-parity",
                        "result": result,
                        "status": APPROVE,
                    }
                except Exception as error:
                    status = FAIL
                    phase = {
                        "checks": [],
                        "name": "checkpoint-parity",
                        "reason": str(error),
                        "status": FAIL,
                    }
                family_results.append(
                    {
                        "family": "dfine",
                        "phases": [phase],
                        "status": phase["status"],
                        "variant": variant,
                    }
                )
            document = result_document(
                identity, family_results=family_results, negatives=[], status=status
            )
            document["command"] = [sys.executable, *sys.argv]
            document["git_revision"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            output = args.output or args.evidence_dir / "task-11-rtdetrv4-merge.json"
            publish_json(output, document)
            print(json.dumps(document, allow_nan=False, sort_keys=True))
            return EXIT_CODES[status]
        if missing:
            print("BLOCKED: missing required assets: {}".format(", ".join(missing)))
            return EXIT_CODES[BLOCKED]
        if rtdetrv4_teacher:
            try:
                phase = _run_dinov3_teacher_preflight(args)
                status = APPROVE
            except Exception as error:
                status = FAIL
                phase = {
                    "checks": [],
                    "name": "teacher-preflight",
                    "reason": str(error),
                    "status": FAIL,
                }
            family_results = [
                {
                    "family": "rtdetrv4",
                    "phases": [phase],
                    "status": status,
                    "variant": variant,
                }
                for _, variant in matrix
            ]
            document = result_document(
                identity, family_results=family_results, negatives=[], status=status
            )
            document["command"] = [sys.executable, *sys.argv]
            document["git_revision"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            output = args.output or args.evidence_dir / "task-18-rtdetrv4-merge.json"
            publish_json(output, document)
            print(json.dumps(document, allow_nan=False, sort_keys=True))
            return EXIT_CODES[status]
        if deim_dfine_runtime or deim_rtdetrv2_runtime or rtdetrv4_runtime:
            family = families[0]
            variants = [variant for _, variant in matrix]
            try:
                manifest, upstream_root, image_paths = _deim_runtime_preflight(
                    args, phases, variants, family
                )
            except FileNotFoundError as error:
                print("BLOCKED: {}".format(error))
                return EXIT_CODES[BLOCKED]
            except ValueError as error:
                print("FAIL: {}".format(error))
                return EXIT_CODES[FAIL]

            train_phase = (
                (
                    _run_deim_dfine_train_resume()
                    if family == "deim-dfine"
                    else (
                        _run_rtdetrv4_train_resume(args)
                        if family == "rtdetrv4"
                        else _run_deim_rtdetrv2_train_resume()
                    )
                )
                if "train-resume" in phases
                else None
            )
            family_results = []
            status = APPROVE
            for variant in variants:
                variant_phases = []
                entry = manifest["models"][variant]
                checkpoint = args.checkpoint_root / entry["filename"]
                for phase_name in phases:
                    try:
                        if phase_name == "train-resume":
                            phase = dict(train_phase)
                        elif phase_name == "verify":
                            parity = importlib.import_module(
                                {
                                    "deim-dfine": "deim_dfine_checkpoint_parity",
                                    "deim-rtdetrv2": "deim_rtdetrv2_checkpoint_parity",
                                    "rtdetrv4": "rtdetrv4_checkpoint_parity",
                                }[family]
                            )

                            artifact, state = parity.preflight_artifact(
                                args.checkpoint_root, variant, manifest
                            )
                            parity.build_local_model(variant, state)
                            result = {
                                "artifact": {
                                    "filename": artifact.name,
                                    "sha256": entry["source_sha256"],
                                    "size_bytes": artifact.stat().st_size,
                                },
                                "state_tensor_count": len(state),
                                "status": APPROVE,
                                "variant": variant,
                            }
                            phase = {
                                "checks": [
                                    {
                                        "name": "manifest-state-preflight",
                                        "status": APPROVE,
                                    }
                                ],
                                "name": phase_name,
                                "result": result,
                                "status": APPROVE,
                            }
                        elif phase_name == "checkpoint-parity":
                            parity = importlib.import_module(
                                {
                                    "deim-dfine": "deim_dfine_checkpoint_parity",
                                    "deim-rtdetrv2": "deim_rtdetrv2_checkpoint_parity",
                                    "rtdetrv4": "rtdetrv4_checkpoint_parity",
                                }[family]
                            )

                            result = parity.validate_variant(
                                variant,
                                args.checkpoint_root,
                                upstream_root,
                                args.manifest or parity.DEFAULT_MANIFEST,
                            )
                            phase = {
                                "checks": result.pop("checks"),
                                "name": phase_name,
                                "result": result,
                                "status": APPROVE,
                            }
                        elif phase_name == "eval":
                            phase = _run_deim_eval(variant, checkpoint, family)
                        elif phase_name == "infer":
                            phase = _run_dfine_infer(
                                variant,
                                checkpoint,
                                upstream_root,
                                image_paths,
                                args.evidence_dir,
                                family=family,
                            )
                        elif phase_name == "export":
                            phase = _run_dfine_export(
                                variant,
                                checkpoint,
                                args.evidence_dir,
                                family=family,
                            )
                        elif phase_name in {"teacher", "teacher-preflight"}:
                            phase = _run_dinov3_teacher_preflight(args)
                            phase["name"] = phase_name
                        else:
                            phase = _run_dfine_coco(
                                variant,
                                checkpoint,
                                args.coco_root,
                                float(entry["official_bbox_ap"]),
                                args.evidence_dir,
                                family=family,
                            )
                    except Exception as error:
                        status = FAIL
                        phase = {
                            "checks": [],
                            "name": phase_name,
                            "reason": str(error),
                            "status": FAIL,
                        }
                    variant_phases.append(phase)
                variant_status = (
                    APPROVE
                    if all(phase["status"] == APPROVE for phase in variant_phases)
                    else FAIL
                )
                family_results.append(
                    {
                        "family": family,
                        "phases": variant_phases,
                        "status": variant_status,
                        "variant": variant,
                    }
                )
            document = result_document(
                identity, family_results=family_results, negatives=[], status=status
            )
            document["command"] = [sys.executable, *sys.argv]
            document["git_revision"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            task = {"deim-dfine": 16, "deim-rtdetrv2": 17, "rtdetrv4": 20}[family]
            output = (
                args.output or args.evidence_dir / f"task-{task}-rtdetrv4-merge.json"
            )
            publish_json(output, document)
            print(json.dumps(document, allow_nan=False, sort_keys=True))
            return EXIT_CODES[status]
        if dfine_runtime:
            variants = [variant for _, variant in matrix]
            try:
                manifest, upstream_root, image_paths = _dfine_runtime_preflight(
                    args, phases, variants
                )
            except FileNotFoundError as error:
                print("BLOCKED: {}".format(error))
                return EXIT_CODES[BLOCKED]
            except ValueError as error:
                print("FAIL: {}".format(error))
                return EXIT_CODES[FAIL]

            train_phase = (
                _run_dfine_train_resume() if "train-resume" in phases else None
            )
            family_results = []
            status = APPROVE
            for variant in variants:
                variant_phases = []
                entry = manifest["models"][variant]
                checkpoint = args.checkpoint_root / entry["filename"]
                for phase_name in phases:
                    try:
                        if phase_name == "train-resume":
                            phase = dict(train_phase)
                        elif phase_name == "verify":
                            from dfine_checkpoint_parity import (
                                build_local_model,
                                preflight_artifact,
                            )

                            artifact, state = preflight_artifact(
                                args.checkpoint_root, variant, manifest
                            )
                            build_local_model(variant, state)
                            phase = {
                                "checks": [
                                    {
                                        "name": "manifest-state-preflight",
                                        "status": APPROVE,
                                    }
                                ],
                                "name": phase_name,
                                "result": {
                                    "artifact": artifact.name,
                                    "state_tensor_count": len(state),
                                },
                                "status": APPROVE,
                            }
                        elif phase_name == "eval":
                            phase = _run_deim_eval(variant, checkpoint, "dfine")
                        elif phase_name == "infer":
                            phase = _run_dfine_infer(
                                variant,
                                checkpoint,
                                upstream_root,
                                image_paths,
                                args.evidence_dir,
                            )
                        elif phase_name == "export":
                            phase = _run_dfine_export(
                                variant, checkpoint, args.evidence_dir
                            )
                        else:
                            phase = _run_dfine_coco(
                                variant,
                                checkpoint,
                                args.coco_root,
                                float(entry["official_bbox_ap"]),
                                args.evidence_dir,
                            )
                    except Exception as error:
                        status = FAIL
                        phase = {
                            "checks": [],
                            "name": phase_name,
                            "reason": str(error),
                            "status": FAIL,
                        }
                    variant_phases.append(phase)
                variant_status = (
                    APPROVE
                    if all(phase["status"] == APPROVE for phase in variant_phases)
                    else FAIL
                )
                family_results.append(
                    {
                        "family": "dfine",
                        "phases": variant_phases,
                        "status": variant_status,
                        "variant": variant,
                    }
                )
            document = result_document(
                identity, family_results=family_results, negatives=[], status=status
            )
            document["command"] = [sys.executable, *sys.argv]
            document["git_revision"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            task = "13" if phases == ["export"] else "12"
            output = (
                args.output
                or args.evidence_dir / "task-{}-rtdetrv4-merge.json".format(task)
            )
            publish_json(output, document)
            print(json.dumps(document, allow_nan=False, sort_keys=True))
            return EXIT_CODES[status]
        raise ValueError(
            "model-family execution plugins are not implemented; use --contract-only only for driver contract tests"
        )

    family_results = [
        {
            "family": family,
            "phases": [
                {"checks": [], "name": phase, "status": "CONTRACT_ONLY"}
                for phase in phases
            ],
            "status": "CONTRACT_ONLY",
            "variant": variant,
        }
        for family, variant in matrix
    ]
    negative_results = [
        {
            "family": family,
            "mutation": negative,
            "status": "CONTRACT_ONLY",
            "variant": variant,
        }
        for family, variant in matrix
        for negative in negatives
    ]
    document = result_document(
        identity,
        family_results=family_results,
        negatives=negative_results,
        status="CONTRACT_ONLY",
    )
    output = args.output or args.evidence_dir / "model-family-contract.json"
    publish_json(output, document)
    print(json.dumps(document, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main))
