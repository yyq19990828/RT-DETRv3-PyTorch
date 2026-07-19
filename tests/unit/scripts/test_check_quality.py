import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts/check_quality.py"


def load_script(monkeypatch, patch_tools=True):
    spec = importlib.util.spec_from_file_location("check_quality", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if patch_tools:
        monkeypatch.setattr(
            module,
            "resolve_tool",
            lambda name: str(ROOT / ".venv/bin" / name),
        )
    return module


def test_build_commands_use_ruff_for_format_and_lint(monkeypatch):
    script = load_script(monkeypatch)
    commands = script.build_commands(fix=False)

    assert commands[0][1:3] == ["format", "--check"]
    assert commands[0][-1] == "."
    assert commands[1][1] == "check"
    assert commands[1][-1] == "."
    assert commands[2][0].endswith("mypy")
    assert "tests/legacy" not in commands[0]


def test_fix_commands_format_before_safe_lint_fixes(monkeypatch):
    script = load_script(monkeypatch)
    commands = script.build_commands(fix=True)

    assert commands[0][1] == "format"
    assert "--check" not in commands[0]
    assert commands[1][1:3] == ["check", "--fix"]


def test_resolve_tool_prefers_current_environment(tmp_path, monkeypatch):
    script = load_script(monkeypatch, patch_tools=False)
    environment_bin = tmp_path / "bin"
    environment_bin.mkdir()
    python = environment_bin / "python"
    ruff = environment_bin / "ruff"
    python.write_bytes(b"")
    ruff.write_bytes(b"")
    monkeypatch.setattr(script.sys, "executable", str(python))

    assert script.resolve_tool("ruff") == str(ruff)


def test_main_stops_after_first_failed_command(monkeypatch):
    script = load_script(monkeypatch)
    commands = [["ruff", "format"], ["ruff", "check"]]
    observed = []

    monkeypatch.setattr(script, "build_commands", lambda fix: commands)

    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, returncode=2)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    assert script.main([]) == 2
    assert observed == [commands[0]]
