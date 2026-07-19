import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/check_coverage.py"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("check_coverage", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "resolve_pytest", lambda: str(ROOT / ".venv/bin/pytest")
    )
    return module


def test_build_command_runs_non_paddle_package_coverage(monkeypatch, tmp_path):
    script = load_script(monkeypatch)
    report_path = tmp_path / "coverage.json"

    command = script.build_command(report_path)

    assert command[0].endswith("pytest")
    assert command[command.index("-m") + 1] == "not paddle"
    assert "--cov=ppdet_pytorch" in command
    assert f"--cov-report=json:{report_path}" in command
    assert "no:cacheprovider" in command


def test_summarize_files_keeps_full_and_direct_scopes_separate(monkeypatch):
    script = load_script(monkeypatch)
    files = {
        "src/ppdet_pytorch/cli/train.py": {
            "summary": {"num_statements": 80, "covered_lines": 60}
        },
        "src/ppdet_pytorch/data/dataset.py": {
            "summary": {"num_statements": 120, "covered_lines": 20}
        },
    }

    full, direct = script.summarize_files(files)

    assert (full.statements, full.covered) == (200, 80)
    assert (direct.statements, direct.covered) == (80, 60)


def test_threshold_failures_report_each_scope(monkeypatch):
    script = load_script(monkeypatch)
    below = script.CoverageSummary(statements=100, covered=40)
    above = script.CoverageSummary(statements=100, covered=70)

    assert len(script.threshold_failures(below, below)) == 2
    assert script.threshold_failures(above, above) == []


def test_main_uses_temporary_outputs_and_reports_success(monkeypatch, capsys):
    script = load_script(monkeypatch)
    observed_environment = {}

    def fake_run(command, **kwargs):
        observed_environment.update(kwargs["env"])
        report_argument = next(
            item for item in command if item.startswith("--cov-report=json:")
        )
        report_path = Path(report_argument.split(":", maxsplit=1)[1])
        report_path.write_text(
            '{"files": {'
            '"src/ppdet_pytorch/cli/train.py": {'
            '"summary": {"num_statements": 100, "covered_lines": 70}}, '
            '"src/ppdet_pytorch/data/dataset.py": {'
            '"summary": {"num_statements": 100, "covered_lines": 20}}}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    assert script.main() == 0
    assert "full package:" in capsys.readouterr().out
    assert "COVERAGE_FILE" in observed_environment
    assert not Path(observed_environment["COVERAGE_FILE"]).parent.exists()


def test_main_preserves_pytest_failure_code(monkeypatch):
    script = load_script(monkeypatch)
    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, returncode=3),
    )

    assert script.main() == 3
