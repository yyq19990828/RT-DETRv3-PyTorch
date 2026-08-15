"""Relay third-party print output through the structured logger."""

import logging
import sys
from types import TracebackType
from typing import IO, Optional, Type

__all__ = ["relay_prints"]


class _LineRelay:
    """File-like stream that forwards each completed line to a logger.

    Lines are relayed the moment they are printed (not buffered until the
    context manager exits), so long-running third-party calls such as
    ``COCOeval.evaluate()`` keep reporting progress in real time. Writes
    arriving while a line is already being relayed go straight to the real
    stdout: the logging handlers themselves write to stdout, and feeding
    their output back into the relay would recurse.
    """

    def __init__(self, logger: logging.Logger, fallback: IO[str]) -> None:
        self._logger = logger
        self._fallback = fallback
        self._partial = ""
        self._relaying = False

    def write(self, text: str) -> int:
        if self._relaying:
            return self._fallback.write(text)
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            line = line.rstrip()
            if line:
                self._emit(line)
        return len(text)

    def flush(self) -> None:
        self._fallback.flush()

    def drain(self) -> None:
        """Log a trailing line that was never terminated with a newline."""
        if self._partial.strip():
            self._emit(self._partial.rstrip())
        self._partial = ""

    def _emit(self, line: str) -> None:
        self._relaying = True
        try:
            # stacklevel=3 reports the print() caller (the third-party
            # source file) instead of this module's write/_emit frames.
            self._logger.info(line, stacklevel=3)
        finally:
            self._relaying = False


class relay_prints:
    """Route print() output inside the block through ``logger``.

    pycocotools reports loading and evaluation progress with bare print()
    calls. Wrapping those calls relays the lines to the logger so they get
    the same rich console rendering, file:line origin, and log-file output
    as our own messages.

    Example::

        with relay_prints(logger):
            coco_dt = coco_gt.loadRes(jsonfile)
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._fallback: Optional[IO[str]] = None
        self._relay: Optional[_LineRelay] = None

    def __enter__(self) -> _LineRelay:
        self._fallback = sys.stdout
        self._relay = _LineRelay(self._logger, self._fallback)
        sys.stdout = self._relay
        return self._relay

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        assert self._fallback is not None and self._relay is not None
        sys.stdout = self._fallback
        self._relay.drain()
