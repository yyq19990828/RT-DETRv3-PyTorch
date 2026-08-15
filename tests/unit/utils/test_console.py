"""Tests for the shared rich console."""

from rich.console import Console

from detrs.utils.console import get_console


def test_get_console_returns_singleton():
    assert get_console() is get_console()


def test_piped_output_has_no_ansi_escapes(capsys):
    console = Console()

    console.print("[green]ok[/green] plain text")

    captured = capsys.readouterr()
    assert "ok" in captured.out
    assert "\x1b[" not in captured.out
