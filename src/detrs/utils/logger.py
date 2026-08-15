"""Logging utilities for RT-DETRv3 PyTorch implementation

Adapted from PaddlePaddle's ppdet.utils.logger to maintain compatibility
while using PyTorch's distributed training APIs.
"""

import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
from rich.logging import RichHandler

from detrs.utils.console import get_console

__all__ = ["setup_logger"]

logger_initialized = []


@lru_cache(maxsize=1)
def _path_roots() -> Tuple[Path, ...]:
    """Roots that log paths are displayed relative to.

    Source-checkout files render as ``src/detrs/...``; site-packages entries
    keep third-party frames recognizable (e.g. ``pycocotools/coco.py``). The
    venv usually lives inside the repository, so the most specific matching
    root must win (see ``_display_path``).
    """
    roots = []
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            roots.append(parent)
            break
    for entry in sys.path:
        if entry and Path(entry).name in {"site-packages", "dist-packages"}:
            roots.append(Path(entry))
    return tuple(roots)


@lru_cache(maxsize=512)
def _display_path(pathname: str) -> str:
    resolved = Path(pathname).resolve()
    shortest: Optional[Path] = None
    for root in _path_roots():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if shortest is None or len(relative.parts) < len(shortest.parts):
            shortest = relative
    return shortest.as_posix() if shortest is not None else os.path.basename(pathname)


class ProjectPathRichHandler(RichHandler):
    """RichHandler whose file:line column shows project-relative paths."""

    # Mirrors RichHandler.render with os.path.basename(record.pathname)
    # replaced by _display_path; the terminal link keeps the absolute path.
    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Any,
        message_renderable: Any,
    ) -> Any:
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        return self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=_display_path(record.pathname),
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )


def setup_logger(
    name: str = "rtdetrv3",
    output: Optional[str] = None,
    log_ranks: Union[str, int, List[int]] = "0",
) -> logging.Logger:
    """
    Initialize logger and set its verbosity level to INFO.

    Args:
        name: The root module name of this logger (default: "rtdetrv3")
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        log_ranks: The ids of gpu to log which are separated by "," when more than 1, "0" by default.
            Only processes with rank in log_ranks will output to console.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Formatter matching PaddlePaddle format
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # Parse log_ranks parameter
    if isinstance(log_ranks, str):
        log_ranks = [int(i) for i in log_ranks.split(",")]
    elif isinstance(log_ranks, int):
        log_ranks = [log_ranks]

    # Logger modules are commonly imported before init_process_group(). Use
    # torchrun's global RANK in that window so non-zero ranks do not attach a
    # rank-0 console handler by mistake.
    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank()
    else:
        try:
            local_rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            local_rank = 0

    # Console logging: only specified ranks. RichHandler renders through the
    # shared console so log lines coordinate with live progress bars; piped or
    # CI output automatically degrades to plain text without ANSI escapes.
    if local_rank in log_ranks:
        ch = ProjectPathRichHandler(
            console=get_console(),
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            log_time_format="[%m/%d %H:%M:%S]",
        )
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    elif output is None:
        # Suppress logging.lastResort on non-log ranks. Without an explicit
        # handler, WARNING messages would still leak to stderr unformatted.
        logger.addHandler(logging.NullHandler())

    # File logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        # Add rank suffix for non-master processes
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        fh = logging.FileHandler(filename, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger_initialized.append(name)
    return logger
