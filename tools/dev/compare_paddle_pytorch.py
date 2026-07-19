"""
T067: Compare Paddle and PyTorch module interfaces

This script compares the public interfaces of dataset and engine modules
between Paddle and PyTorch implementations to identify:
- Implemented features
- Missing features
- Different APIs
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Set

# Allow running this development script without installing either codebase.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "third-party" / "RT-DETRv3-paddle"))


def get_public_methods(cls) -> Set[str]:
    """Get all public methods of a class (excluding private and magic methods)"""
    methods = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(cls, name)
            # Include methods and functions, but exclude classes and modules
            if (
                callable(attr)
                and not inspect.isclass(attr)
                and not inspect.ismodule(attr)
            ):
                methods.add(name)
        except (AttributeError, TypeError):
            pass
    return methods


def get_public_attributes(cls) -> Set[str]:
    """Get all public attributes/properties of a class"""
    return {
        name
        for name in dir(cls)
        if not name.startswith("_") and not callable(getattr(cls, name, None))
    }


def get_method_signature(cls, method_name: str) -> str:
    """Get the signature of a method"""
    try:
        method = getattr(cls, method_name)
        sig = inspect.signature(method)
        return str(sig)
    except Exception as e:
        return f"<error: {e}>"


def compare_classes(paddle_cls, pytorch_cls, class_name: str) -> Dict:
    """Compare two classes and return comparison results"""
    paddle_methods = get_public_methods(paddle_cls) if paddle_cls else set()
    pytorch_methods = get_public_methods(pytorch_cls) if pytorch_cls else set()

    result = {
        "class_name": class_name,
        "paddle_exists": paddle_cls is not None,
        "pytorch_exists": pytorch_cls is not None,
        "common_methods": sorted(paddle_methods & pytorch_methods),
        "paddle_only_methods": sorted(paddle_methods - pytorch_methods),
        "pytorch_only_methods": sorted(pytorch_methods - paddle_methods),
        "signature_diffs": [],
    }

    # Compare signatures of common methods
    for method in result["common_methods"]:
        paddle_sig = get_method_signature(paddle_cls, method) if paddle_cls else ""
        pytorch_sig = get_method_signature(pytorch_cls, method) if pytorch_cls else ""
        if paddle_sig != pytorch_sig:
            result["signature_diffs"].append(
                {"method": method, "paddle": paddle_sig, "pytorch": pytorch_sig}
            )

    return result


def safe_import_class(module_path: str, class_name: str):
    """Safely import a class, return None if fails"""
    try:
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, attr_name = parts
        else:
            module_name = module_path
            attr_name = class_name

        module = importlib.import_module(module_name)
        return getattr(module, attr_name, None) or getattr(module, class_name, None)
    except Exception as e:
        print(f"⚠️  Failed to import {module_path}.{class_name}: {e}")
        return None


def compare_datasets():
    """Compare dataset modules"""
    print("\n" + "=" * 70)
    print("Dataset Module Comparison")
    print("=" * 70)

    datasets_to_compare = [
        ("COCODataSet", "ppdet.data.source.coco", "ppdet_pytorch.data.source.coco"),
        (
            "DetDataset",
            "ppdet.data.source.dataset",
            "ppdet_pytorch.data.source.dataset",
        ),
    ]

    results = []
    for class_name, paddle_module, pytorch_module in datasets_to_compare:
        print(f"\n--- {class_name} ---")
        paddle_cls = safe_import_class(paddle_module, class_name)
        pytorch_cls = safe_import_class(pytorch_module, class_name)

        result = compare_classes(paddle_cls, pytorch_cls, class_name)
        results.append(result)

        print_comparison_result(result)

    return results


def compare_engines():
    """Compare engine modules"""
    print("\n" + "=" * 70)
    print("Engine Module Comparison")
    print("=" * 70)

    engines_to_compare = [
        ("Trainer", "ppdet.engine.trainer", "ppdet_pytorch.engine.trainer"),
        ("Checkpointer", "ppdet.engine.callbacks", "ppdet_pytorch.engine.callbacks"),
    ]

    results = []
    for class_name, paddle_module, pytorch_module in engines_to_compare:
        print(f"\n--- {class_name} ---")
        paddle_cls = safe_import_class(paddle_module, class_name)
        pytorch_cls = safe_import_class(pytorch_module, class_name)

        result = compare_classes(paddle_cls, pytorch_cls, class_name)
        results.append(result)

        print_comparison_result(result)

    return results


def compare_metrics():
    """Compare metrics modules"""
    print("\n" + "=" * 70)
    print("Metrics Module Comparison")
    print("=" * 70)

    metrics_to_compare = [
        ("COCOMetric", "ppdet.metrics.coco_utils", "ppdet_pytorch.metrics.coco_utils"),
    ]

    results = []
    for class_name, paddle_module, pytorch_module in metrics_to_compare:
        print(f"\n--- {class_name} ---")
        paddle_cls = safe_import_class(paddle_module, class_name)
        pytorch_cls = safe_import_class(pytorch_module, class_name)

        result = compare_classes(paddle_cls, pytorch_cls, class_name)
        results.append(result)

        print_comparison_result(result)

    return results


def print_comparison_result(result: Dict):
    """Print comparison result in a readable format"""
    if not result["paddle_exists"]:
        print(f"⚠️  {result['class_name']} - Paddle class not found")
    if not result["pytorch_exists"]:
        print(f"⚠️  {result['class_name']} - PyTorch class not found")

    if not (result["paddle_exists"] and result["pytorch_exists"]):
        return

    print(
        f"✅ Common methods ({len(result['common_methods'])}): {', '.join(result['common_methods'][:5])}"
        + (", ..." if len(result["common_methods"]) > 5 else "")
    )

    if result["paddle_only_methods"]:
        print(
            f"⚠️  Paddle-only methods ({len(result['paddle_only_methods'])}): {', '.join(result['paddle_only_methods'][:5])}"
            + (", ..." if len(result["paddle_only_methods"]) > 5 else "")
        )

    if result["pytorch_only_methods"]:
        print(
            f"ℹ️  PyTorch-only methods ({len(result['pytorch_only_methods'])}): {', '.join(result['pytorch_only_methods'][:5])}"
            + (", ..." if len(result["pytorch_only_methods"]) > 5 else "")
        )

    if result["signature_diffs"]:
        print(f"⚠️  Signature differences ({len(result['signature_diffs'])})")
        for diff in result["signature_diffs"][:3]:
            print(f"    {diff['method']}:")
            print(f"      Paddle:  {diff['paddle']}")
            print(f"      PyTorch: {diff['pytorch']}")


def generate_summary(dataset_results, engine_results, metrics_results):
    """Generate summary statistics"""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_results = dataset_results + engine_results + metrics_results
    total_classes = len(all_results)
    implemented = sum(1 for r in all_results if r["pytorch_exists"])
    fully_compatible = sum(
        1 for r in all_results if r["pytorch_exists"] and not r["paddle_only_methods"]
    )

    print(f"\nTotal classes compared: {total_classes}")
    print(
        f"PyTorch implemented: {implemented}/{total_classes} ({implemented / total_classes * 100:.1f}%)"
    )
    print(
        f"Fully compatible (no missing methods): {fully_compatible}/{total_classes} ({fully_compatible / total_classes * 100:.1f}%)"
    )

    print("\n⚠️  Missing features to implement:")
    for result in all_results:
        if result["paddle_only_methods"]:
            print(f"\n  {result['class_name']}:")
            for method in result["paddle_only_methods"]:
                print(f"    - {method}")

    return {
        "total_classes": total_classes,
        "implemented": implemented,
        "fully_compatible": fully_compatible,
        "all_results": all_results,
    }


def main():
    """Run all comparisons"""
    print("=" * 70)
    print("T067: Paddle vs PyTorch Module Interface Comparison")
    print("=" * 70)

    # Compare each module type
    dataset_results = compare_datasets()
    engine_results = compare_engines()
    metrics_results = compare_metrics()

    # Generate summary
    summary = generate_summary(dataset_results, engine_results, metrics_results)

    print("\n" + "=" * 70)
    print("✅ Comparison complete! See results above.")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
