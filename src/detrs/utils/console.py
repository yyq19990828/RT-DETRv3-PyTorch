"""Shared rich console for CLI and training presentation output."""

from __future__ import annotations

from rich.console import Console

__all__ = ["get_console"]

_console: Console | None = None


def get_console() -> Console:
    """Return the shared rich console singleton.

    Terminal detection and ``NO_COLOR`` follow rich defaults: piped or CI
    output stays plain text without ANSI escapes.
    """
    global _console
    if _console is None:
        _console = Console()
    return _console
