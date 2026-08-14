#!/usr/bin/env python3
"""Compare named tensors from pinned upstream and local PyTorch runners."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    result_document,
    run_main,
)

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6
SURFACES = ("state", "activation", "output", "eager", "onnx", "torchscript")


def compare_named_tensors(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> list[dict[str, Any]]:
    """Compare tensors in stable name order with actionable diagnostics."""
    import torch

    names = sorted(set(reference) | set(candidate))
    results = []
    for name in names:
        if name not in reference or name not in candidate:
            results.append(
                {
                    "name": name,
                    "reason": "missing from {}".format(
                        "reference" if name not in reference else "candidate"
                    ),
                    "status": FAIL,
                }
            )
            continue
        left = reference[name].detach().cpu()
        right = candidate[name].detach().cpu()
        if left.shape != right.shape:
            results.append(
                {
                    "candidate_shape": list(right.shape),
                    "name": name,
                    "reason": "shape mismatch",
                    "reference_shape": list(left.shape),
                    "status": FAIL,
                }
            )
            continue
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            results.append(
                {"name": name, "reason": "non-finite tensor", "status": FAIL}
            )
            continue
        difference = torch.abs(left.to(torch.float64) - right.to(torch.float64))
        if difference.numel():
            flat_index = int(torch.argmax(difference).item())
            max_abs = float(difference.flatten()[flat_index].item())
            denominator = torch.abs(left.to(torch.float64)).clamp_min(atol)
            max_rel = float(torch.max(difference / denominator).item())
        else:
            flat_index = 0
            max_abs = 0.0
            max_rel = 0.0
        passed = bool(torch.allclose(left, right, rtol=rtol, atol=atol))
        results.append(
            {
                "atol": atol,
                "candidate_dtype": str(right.dtype),
                "max_abs_error": max_abs,
                "max_error_flat_index": flat_index,
                "max_rel_error": max_rel,
                "name": name,
                "reference_dtype": str(left.dtype),
                "rtol": rtol,
                "shape": list(left.shape),
                "status": APPROVE if passed else FAIL,
            }
        )
    return results


def _load_tensor_document(path: Path) -> dict[str, Any]:
    import torch

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document:
        raise ValueError("tensor document must be a non-empty mapping: {}".format(path))
    tensors = {}
    for name, value in document.items():
        if not isinstance(name, str):
            raise ValueError("tensor names must be strings")
        tensors[name] = torch.as_tensor(value)
    return tensors


def validate_baseline(
    baseline: Mapping[str, Any], families: Sequence[str], identity: str
) -> None:
    if baseline.get("plan_identity") != identity:
        raise ValueError("baseline plan identity does not match the current plan")
    if baseline.get("status") != APPROVE:
        raise ValueError("baseline is not approved")
    recorded = {
        item.get("family"): item
        for item in baseline.get("family_results", [])
        if isinstance(item, dict)
    }
    if not set(families).issubset(recorded):
        raise ValueError("baseline does not contain every requested family")
    if any(recorded[family].get("status") != APPROVE for family in families):
        raise ValueError("baseline family is not approved")


def reject_submodule_diff(diff: str) -> None:
    if diff.strip():
        raise ValueError("read-only Paddle submodule has modifications")


def _publish_result(path: Path, document: dict[str, Any]) -> None:
    document["command"] = [sys.executable, *sys.argv]
    document["git_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    publish_json(path, document)


def create_parser() -> argparse.ArgumentParser:
    parser = DriverArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variants", default="smallest")
    parser.add_argument("--surfaces", default="state,activation,output")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--upstream-revision")
    parser.add_argument("--expected-upstream-revision")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    families = parse_csv(args.family, "family")
    surfaces = parse_csv(args.surfaces, "surfaces")
    unknown = sorted(set(surfaces) - set(SURFACES))
    if unknown:
        raise ValueError("unknown surfaces: {}".format(",".join(unknown)))
    parse_csv(args.variants, "variants")
    identity = normalized_plan_identity(args.plan)
    if (args.reference is None) != (args.candidate is None):
        raise ValueError("--reference and --candidate must be provided together")
    if args.expected_upstream_revision:
        if args.upstream_revision != args.expected_upstream_revision:
            raise ValueError(
                "upstream revision mismatch: expected {}, got {}".format(
                    args.expected_upstream_revision, args.upstream_revision
                )
            )

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        validate_baseline(baseline, families, identity)

    if args.reference is None:
        if not args.baseline:
            print("BLOCKED: no upstream/local runner inputs were provided")
            return EXIT_CODES[BLOCKED]
        family_results = [
            {
                "family": family,
                "status": APPROVE,
                "surfaces": [
                    {"name": surface, "status": APPROVE} for surface in surfaces
                ],
                "variant_selection": args.variants,
            }
            for family in families
        ]
        _publish_result(
            args.output,
            result_document(identity, family_results=family_results, status=APPROVE),
        )
        print(APPROVE)
        return EXIT_CODES[APPROVE]

    reference = _load_tensor_document(args.reference)
    candidate = _load_tensor_document(args.candidate)
    tensors = compare_named_tensors(reference, candidate)
    status = APPROVE if all(item["status"] == APPROVE for item in tensors) else FAIL
    family_results = [
        {
            "family": family,
            "surfaces": [
                {"name": surface, "status": status, "tensors": tensors}
                for surface in surfaces
            ],
            "status": status,
            "variant_selection": args.variants,
        }
        for family in families
    ]
    _publish_result(
        args.output,
        result_document(identity, family_results=family_results, status=status),
    )
    if status == FAIL:
        failed = next(item for item in tensors if item["status"] == FAIL)
        print(
            "FAIL: tensor {} max_abs_error={}".format(
                failed["name"], failed.get("max_abs_error", "n/a")
            )
        )
    else:
        print(APPROVE)
    return EXIT_CODES[status]


if __name__ == "__main__":
    raise SystemExit(run_main(main))
