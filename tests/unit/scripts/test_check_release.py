import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/check_release.py"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_repository_release_metadata_and_manifest_are_valid(monkeypatch):
    script = load_script(monkeypatch)

    summary = script.validate_repository(require_models=False)

    assert summary["manifest_entries"] == 4
    assert summary["distribution_artifacts"] == 4
    assert summary["checked_model_files"] >= 0


@pytest.mark.parametrize("name", ["../payload", "/absolute/payload"])
def test_archive_validation_rejects_unsafe_paths(monkeypatch, name):
    script = load_script(monkeypatch)

    with pytest.raises(ValueError):
        script._validate_archive_names([name], "fixture")
