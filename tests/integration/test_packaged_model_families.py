import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from detrs.cli import export as export_cli

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
FAMILIES = ("rtdetrv3", "dfine", "deim-dfine", "deim-rtdetrv2", "rtdetrv4", "deimv2")
NEW_FAMILY_CONFIGS = (
    *(f"configs/dfine/dfine_hgnetv2_{variant}_coco.yml" for variant in "nsmlx"),
    *(f"configs/deim/dfine/deim_hgnetv2_{variant}_coco.yml" for variant in "nsmlx"),
    "configs/deim/rtdetrv2/deim_r18vd_120e_coco.yml",
    "configs/deim/rtdetrv2/deim_r34vd_120e_coco.yml",
    "configs/deim/rtdetrv2/deim_r50vd_m_60e_coco.yml",
    "configs/deim/rtdetrv2/deim_r50vd_60e_coco.yml",
    "configs/deim/rtdetrv2/deim_r101vd_60e_coco.yml",
    *(f"configs/rtdetrv4/rtdetrv4_hgnetv2_{variant}_coco.yml" for variant in "smlx"),
    *(f"configs/deimv2/deimv2_dinov3_{variant}_coco.yml" for variant in "smlx"),
    *(
        f"configs/deimv2/deimv2_hgnetv2_{variant}_coco.yml"
        for variant in ("n", "pico", "femto", "atto")
    ),
)
MANIFESTS = (
    "rtdetrv3_coco.yml",
    "dfine_coco.yml",
    "deim_dfine_coco.yml",
    "deim_rtdetrv2_coco.yml",
    "rtdetrv4_coco.yml",
    "deimv2_coco.yml",
)


def _single_wheel(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"expected exactly one wheel, found {len(paths)}")
    return paths[0]


def _require_files(names, required):
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"packaged files are missing: {sorted(missing)}")


def _reject_paddle_imports(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        if any(name == "paddle" or name.startswith("paddle.") for name in modules):
            raise ValueError("core smoke imported Paddle")


def test_installed_wheel_lists_and_loads_all_model_families(tmp_path):
    wheels = sorted(DIST.glob("*.whl"))
    if not wheels:
        pytest.skip("run `uv build` before the packaged-wheel integration test")
    wheel = _single_wheel(wheels)
    target = tmp_path / "site-packages"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(target),
            "--no-deps",
            str(wheel),
        ],
        check=True,
        cwd=tmp_path,
    )

    with ZipFile(wheel) as archive:
        names = archive.namelist()
    required = {
        *(f"detrs/{path}" for path in NEW_FAMILY_CONFIGS),
        *(f"detrs/configs/checkpoints/{name}" for name in MANIFESTS),
    }
    _require_files(names, required)

    script = """
import builtins
import json
from pathlib import Path

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "paddle" or name.startswith("paddle."):
        raise AssertionError("core wheel attempted to import Paddle")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import detrs
from detrs.cli.models import default_manifest_path, load_artifacts
from detrs.core.workspace import load_config

target = Path({target!r}).resolve()
assert str(Path(detrs.__file__).resolve()).startswith(str(target))
families = {families!r}
configs = {configs!r}
result = {{family: list(load_artifacts(default_manifest_path(family))) for family in families}}
for path in configs:
    load_config(path)
assert "paddle" not in __import__("sys").modules
print(json.dumps(result, sort_keys=True))
""".format(target=str(target), families=FAMILIES, configs=NEW_FAMILY_CONFIGS)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    records = json.loads(completed.stdout)
    assert set(records) == set(FAMILIES)
    assert len(records["dfine"]) == 5
    assert len(records["deim-dfine"]) == 5
    assert len(records["deim-rtdetrv2"]) == 5
    assert len(records["rtdetrv4"]) == 4


def test_rejects_missing_config():
    with pytest.raises(ValueError, match="missing"):
        _require_files({"detrs/configs/present.yml"}, {"missing.yml"})


def test_rejects_paddle_core_import():
    with pytest.raises(ValueError, match="Paddle"):
        _reject_paddle_imports("import paddle\n")


def test_rejects_lowered_threshold():
    from scripts import check_coverage

    assert check_coverage.FULL_PACKAGE_MINIMUM >= 50.5
    assert check_coverage.DIRECT_MAINTAINED_MINIMUM >= 90.0


def test_rejects_opset16():
    default = inspect.signature(export_cli.main).parameters.get("opset_version")
    assert default is None
    parser = export_cli.create_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["config.yml", "model.pth", "--opset-version", "16"])


def test_rejects_multiple_wheels(tmp_path):
    wheels = [tmp_path / "one.whl", tmp_path / "two.whl"]
    with pytest.raises(ValueError, match="exactly one wheel"):
        _single_wheel(wheels)
