import hashlib
import importlib.util
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.upstream_parity import run_driver, tensor_document

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / ".omo/plans/rtdetrv4-merge.md"
DRIVERS = (
    "compare_upstream_pytorch.py",
    "validate_model_family.py",
    "audit_plan_evidence.py",
    "audit_model_family_graphs.py",
)


@pytest.fixture
def validation_driver(monkeypatch):
    path = ROOT / "tools/dev/validate_model_family.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("validate_model_family_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compare_driver(monkeypatch):
    path = ROOT / "tools/dev/compare_upstream_pytorch.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("compare_upstream_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_contract_help():
    for driver in DRIVERS:
        result = run_driver(ROOT, driver, ["--help"])
        assert result.returncode == 0, (driver, result.stdout, result.stderr)


def test_normalized_plan_identity_ignores_only_task_checkboxes(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_bytes(b"- [ ] 1. task\n  - [ ] nested\n- [x] F2. final\n")
    second.write_bytes(b"- [x] 1. task\n  - [ ] nested\n- [ ] F2. final\n")
    expected = hashlib.sha256(
        b"- [ ] 1. task\n  - [ ] nested\n- [ ] F2. final\n"
    ).hexdigest()

    output_a = tmp_path / "a.json"
    output_b = tmp_path / "b.json"
    arguments = [
        "--family",
        "rtdetrv3",
        "--reference",
        str(tensor_document(tmp_path / "reference.json", {"x": [1.0]})),
        "--candidate",
        str(tensor_document(tmp_path / "candidate.json", {"x": [1.0]})),
    ]
    result_a = run_driver(
        ROOT,
        "compare_upstream_pytorch.py",
        [*arguments, "--plan", str(first), "--output", str(output_a)],
    )
    result_b = run_driver(
        ROOT,
        "compare_upstream_pytorch.py",
        [*arguments, "--plan", str(second), "--output", str(output_b)],
    )

    assert result_a.returncode == result_b.returncode == 0
    assert json.loads(output_a.read_text())["plan_identity"] == expected
    assert output_a.read_bytes() == output_b.read_bytes()


def test_plan_identity_changes_for_any_other_byte(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("- [ ] 1. task\n", encoding="utf-8")
    second.write_text("- [ ] 1. changed\n", encoding="utf-8")
    reference = tensor_document(tmp_path / "reference.json", {"x": [1.0]})
    candidate = tensor_document(tmp_path / "candidate.json", {"x": [1.0]})
    identities = []
    for index, plan in enumerate((first, second)):
        output = tmp_path / "{}.json".format(index)
        result = run_driver(
            ROOT,
            "compare_upstream_pytorch.py",
            [
                "--family",
                "rtdetrv3",
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--plan",
                str(plan),
                "--output",
                str(output),
            ],
        )
        assert result.returncode == 0
        identities.append(json.loads(output.read_text())["plan_identity"])
    assert identities[0] != identities[1]


def test_compare_rejects_wrong_upstream_sha_before_output(tmp_path):
    output = tmp_path / "result.json"
    result = run_driver(
        ROOT,
        "compare_upstream_pytorch.py",
        [
            "--family",
            "dfine",
            "--upstream-revision",
            "wrong",
            "--expected-upstream-revision",
            "267a6da6d04c8ad52e54120692896515b9e55981",
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 1
    assert "upstream revision mismatch" in result.stdout
    assert not output.exists()


def test_scope_audit_rejects_submodule_diff(compare_driver):
    with pytest.raises(ValueError, match="submodule has modifications"):
        compare_driver.reject_submodule_diff("Submodule third-party changed\n")


def test_parity_rejects_v3_baseline_mismatch(tmp_path):
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "result.json"
    baseline.write_text(
        json.dumps(
            {
                "family_results": [{"family": "rtdetrv3", "status": "APPROVE"}],
                "plan_identity": "0" * 64,
                "status": "APPROVE",
            }
        ),
        encoding="utf-8",
    )

    result = run_driver(
        ROOT,
        "compare_upstream_pytorch.py",
        [
            "--baseline",
            str(baseline),
            "--family",
            "rtdetrv3",
            "--surfaces",
            "eager,onnx,torchscript",
            "--output",
            str(output),
        ],
    )

    assert result.returncode == 1
    assert "baseline plan identity" in result.stdout
    assert not output.exists()


def test_compare_reports_perturbed_tensor_and_maximum_error(tmp_path):
    output = tmp_path / "result.json"
    result = run_driver(
        ROOT,
        "compare_upstream_pytorch.py",
        [
            "--family",
            "dfine",
            "--reference",
            str(tensor_document(tmp_path / "reference.json", {"aifi.f5": [0.0]})),
            "--candidate",
            str(tensor_document(tmp_path / "candidate.json", {"aifi.f5": [0.1]})),
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 1
    assert "aifi.f5" in result.stdout
    assert "max_abs_error" in result.stdout
    document = json.loads(output.read_text())
    assert document["status"] == "FAIL"


def test_validate_parses_final_multi_family_invocation_deterministically(tmp_path):
    output = tmp_path / "result.json"
    arguments = [
        "--family",
        "dfine,deim-dfine,deim-rtdetrv2,rtdetrv4",
        "--variants",
        "smallest",
        "--phase",
        "verify,train-resume,eval,infer,export,teacher",
        "--negative",
        "missing-teacher,bad-checksum,wrong-size,wrong-family,missing-stage1",
        "--evidence-dir",
        str(tmp_path),
        "--output",
        str(output),
        "--contract-only",
    ]
    first = run_driver(ROOT, "validate_model_family.py", arguments)
    first_bytes = output.read_bytes()
    second = run_driver(ROOT, "validate_model_family.py", arguments)

    assert first.returncode == second.returncode == 0
    assert output.read_bytes() == first_bytes
    document = json.loads(first_bytes)
    assert [item["variant"] for item in document["family_results"]] == [
        "n",
        "n",
        "s",
        "s",
    ]
    assert document["status"] == "CONTRACT_ONLY"


def test_validate_multi_family_dispatches_only_supported_phases(
    tmp_path, monkeypatch, validation_driver
):
    dispatched = []

    def invoke(args, family, phases, output):
        del args, output
        dispatched.append((family, phases))
        return {
            "family_results": [
                {
                    "family": family,
                    "phases": [
                        {"checks": [], "name": phase, "status": "APPROVE"}
                        for phase in phases
                    ],
                    "status": "APPROVE",
                    "variant": validation_driver.FAMILIES[family][0],
                }
            ],
            "status": "APPROVE",
        }

    monkeypatch.setattr(validation_driver, "_invoke_single_family", invoke)
    monkeypatch.setattr(validation_driver, "_run_negatives", lambda args, items: [])
    monkeypatch.setattr(
        validation_driver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40),
    )
    output = tmp_path / "aggregate.json"
    result = validation_driver.main(
        [
            "--family",
            "dfine,deim-dfine,deim-rtdetrv2,rtdetrv4",
            "--variants",
            "smallest",
            "--phase",
            "verify,train-resume,eval,infer,export,teacher",
            "--plan",
            str(PLAN),
            "--evidence-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert [family for family, _ in dispatched] == [
        "dfine",
        "deim-dfine",
        "deim-rtdetrv2",
        "rtdetrv4",
    ]
    assert all("teacher" not in phases for _, phases in dispatched[:-1])
    assert dispatched[-1][1][-1] == "teacher"
    document = json.loads(output.read_text())
    assert document["schema_version"] == 1
    assert document["status"] == "APPROVE"


def test_validate_multi_family_aggregates_child_failure(
    tmp_path, monkeypatch, validation_driver
):
    def invoke(args, family, phases, output):
        del args, phases, output
        status = "FAIL" if family == "deim-dfine" else "APPROVE"
        return {
            "family_results": [
                {
                    "family": family,
                    "phases": [],
                    "status": status,
                    "variant": validation_driver.FAMILIES[family][0],
                }
            ],
            "status": status,
        }

    monkeypatch.setattr(validation_driver, "_invoke_single_family", invoke)
    monkeypatch.setattr(validation_driver, "_run_negatives", lambda args, items: [])
    monkeypatch.setattr(
        validation_driver.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40),
    )
    output = tmp_path / "aggregate.json"
    result = validation_driver.main(
        [
            "--family",
            "dfine,deim-dfine",
            "--variants",
            "smallest",
            "--phase",
            "verify",
            "--plan",
            str(PLAN),
            "--evidence-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 1
    document = json.loads(output.read_text())
    assert document["status"] == "FAIL"
    assert [item["status"] for item in document["family_results"]] == [
        "APPROVE",
        "FAIL",
    ]


def test_validate_executes_all_negative_preflights(
    tmp_path, monkeypatch, validation_driver
):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "dfine.pth").write_bytes(b"original")
    (checkpoint_root / "deim.pth").write_bytes(b"foreign-data")
    calls = []

    def preflight(root, variant, manifest):
        calls.append(
            (variant, (root / manifest["models"][variant]["filename"]).read_bytes())
        )
        path = root / manifest["models"][variant]["filename"]
        if path.stat().st_size != 8:
            raise ValueError("size mismatch")
        if path.read_bytes() != b"original":
            raise ValueError("checksum mismatch")
        raise AssertionError("negative unexpectedly matched the official artifact")

    modules = {
        "dfine_checkpoint_parity": SimpleNamespace(
            DEFAULT_MANIFEST=tmp_path / "dfine.yml",
            load_manifest=lambda path: {"models": {"n": {"filename": "dfine.pth"}}},
            preflight_artifact=preflight,
        ),
        "deim_dfine_checkpoint_parity": SimpleNamespace(
            DEFAULT_MANIFEST=tmp_path / "deim.yml",
            load_manifest=lambda path: {"models": {"n": {"filename": "deim.pth"}}},
        ),
    }
    monkeypatch.setattr(
        validation_driver.importlib,
        "import_module",
        lambda name: modules[name],
    )

    pytest_commands = []

    def run(command, **kwargs):
        del kwargs
        pytest_commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation_driver.subprocess, "run", run)
    args = Namespace(
        checkpoint_root=checkpoint_root,
        coco_root=None,
        dinov3_repo=tmp_path,
        dinov3_weights=checkpoint_root / "dfine.pth",
        dinov3_sha256="0" * 64,
    )
    results = validation_driver._run_negatives(args, list(validation_driver.NEGATIVES))

    assert [item["mutation"] for item in results] == list(validation_driver.NEGATIVES)
    assert all(item["status"] == "APPROVE" for item in results)
    assert all(item["state_mutated"] is False for item in results)
    assert len(calls) == 3
    assert len(pytest_commands) == 1
    assert "test_rejects_missing_stage1_without_state_mutation" in pytest_commands[0][3]
    assert (checkpoint_root / "dfine.pth").read_bytes() == b"original"


def test_validate_rejects_missing_assets_before_creating_evidence(tmp_path):
    evidence = tmp_path / "missing" / "evidence"
    result = run_driver(
        ROOT,
        "validate_model_family.py",
        [
            "--family",
            "dfine",
            "--phase",
            "checkpoint-parity",
            "--evidence-dir",
            str(evidence),
        ],
    )
    assert result.returncode == 2
    assert "checkpoint_root" in result.stdout
    assert not evidence.exists()


def test_deim_rtdetrv2_validate_contract_and_missing_asset_boundary(tmp_path):
    output = tmp_path / "contract.json"
    arguments = [
        "--family",
        "deim-rtdetrv2",
        "--variants",
        "all",
        "--phase",
        "all",
        "--evidence-dir",
        str(tmp_path),
        "--output",
        str(output),
        "--contract-only",
    ]
    result = run_driver(ROOT, "validate_model_family.py", arguments)
    assert result.returncode == 0
    document = json.loads(output.read_text())
    assert [item["variant"] for item in document["family_results"]] == [
        "s",
        "m",
        "m-star",
        "l",
        "x",
    ]
    assert document["status"] == "CONTRACT_ONLY"

    blocked = run_driver(
        ROOT,
        "validate_model_family.py",
        [
            "--family",
            "deim-rtdetrv2",
            "--phase",
            "verify",
            "--evidence-dir",
            str(tmp_path / "blocked"),
        ],
    )
    assert blocked.returncode == 2
    assert "checkpoint_root" in blocked.stdout


def test_rtdetrv4_teacher_preflight_does_not_require_student_checkpoint(tmp_path):
    result = run_driver(
        ROOT,
        "validate_model_family.py",
        [
            "--family",
            "rtdetrv4",
            "--phase",
            "teacher-preflight",
            "--evidence-dir",
            str(tmp_path),
        ],
    )

    assert result.returncode == 2
    assert "dinov3_repo" in result.stdout
    assert "checkpoint_root" not in result.stdout


def test_dfine_upstream_preflight_rejects_tracked_modifications(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    (upstream / "tracked.py").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=upstream, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=upstream,
        check=True,
    )
    (upstream / "tracked.py").write_text("dirty\n", encoding="utf-8")

    path = ROOT / "tools/dev/dfine_checkpoint_parity.py"
    spec = importlib.util.spec_from_file_location("dfine_parity_dirty", path)
    assert spec is not None and spec.loader is not None
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    parity.PINNED_SHA = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="modifications"):
        parity.verify_upstream_checkout(upstream)

    subprocess.run(["git", "checkout", "--", "tracked.py"], cwd=upstream, check=True)
    (upstream / "untracked.py").write_text("shadow\n", encoding="utf-8")
    with pytest.raises(ValueError, match="modifications"):
        parity.verify_upstream_checkout(upstream)


def test_validate_rejects_source_import_in_installed_mode(tmp_path):
    prefix = tmp_path / "venv"
    (prefix / "lib/python3.12/site-packages").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    result = run_driver(
        ROOT,
        "validate_model_family.py",
        [
            "--family",
            "rtdetrv3",
            "--evidence-dir",
            str(evidence),
            "--installed-prefix",
            str(prefix),
            "--contract-only",
        ],
    )
    assert result.returncode == 1
    assert "does not contain ppdet_pytorch" in result.stdout
    assert not evidence.exists()


def test_validate_accepts_package_inside_installed_prefix(tmp_path):
    prefix = tmp_path / "venv"
    site_packages = prefix / "lib/python3.12/site-packages"
    package = site_packages / "ppdet_pytorch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    output = tmp_path / "result.json"
    result = run_driver(
        ROOT,
        "validate_model_family.py",
        [
            "--family",
            "rtdetrv3",
            "--evidence-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--installed-prefix",
            str(prefix),
            "--contract-only",
        ],
    )
    assert result.returncode == 0
    assert json.loads(output.read_text())["status"] == "CONTRACT_ONLY"


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ({"nodes": ["dinov3/teacher_encoder_output"]}, "training-residue"),
        ({"opset": 16}, "opset"),
        ({"paddle_imports": ["ppdet_pytorch/model.py:1"]}, "core-dependencies"),
        ({"tolerances": {"activation_atol": 0.1}}, "tolerance-contract"),
    ],
)
def test_graph_audit_rejects_training_node_opset16_paddle_import_tolerance_change(
    tmp_path, fixture, message
):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    output = tmp_path / "result.json"
    result = run_driver(
        ROOT,
        "audit_model_family_graphs.py",
        [
            "--all",
            "--fixture",
            str(fixture_path),
            "--evidence-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 1
    checks = json.loads(output.read_text())["family_results"][0]["checks"]
    assert next(item for item in checks if item["name"] == message)["status"] == "FAIL"


def test_rejects_wrong_plan_identity_fixture(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    receipt = {
        "command": ["pytest"],
        "git_revision": "1" * 40,
        "plan_identity": "0" * 64,
        "schema_version": 1,
        "status": "APPROVE",
    }
    (attempt / "task-1-rtdetrv4-merge.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    output = tmp_path / "audit.md"
    result = run_driver(
        ROOT,
        "audit_plan_evidence.py",
        [
            "--plan",
            str(PLAN),
            "--attempt-dir",
            str(attempt),
            "--require-tasks",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 1
    assert "stale plan identity" in result.stdout
    assert not output.exists()


def test_audit_plan_accepts_tasks_and_finals(tmp_path):
    common = {
        "command": ["pytest"],
        "git_revision": "1" * 40,
        "plan_identity": None,
        "schema_version": 1,
        "status": "APPROVE",
    }
    normalized = PLAN.read_bytes()
    import re

    normalized = re.sub(
        rb"(?m)^- \[[ xX]\] ((?:[1-9][0-9]*|F[1-9][0-9]*)\.)",
        rb"- [ ] \1",
        normalized,
    )
    common["plan_identity"] = hashlib.sha256(normalized).hexdigest()
    for name in (
        "task-1-rtdetrv4-merge.json",
        "task-2-rtdetrv4-merge.json",
        "final-F2-quality.json",
    ):
        (tmp_path / name).write_text(json.dumps(common), encoding="utf-8")
    output = tmp_path / "audit.md"
    result = run_driver(
        ROOT,
        "audit_plan_evidence.py",
        [
            "--attempt-dir",
            str(tmp_path),
            "--require-tasks",
            "1-2",
            "--require-finals",
            "F2",
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 0
    assert output.read_text(encoding="utf-8").endswith("APPROVE\n")
