import logging
import sys

import pytest

from detrs.utils.stdio import relay_prints


class _RecordCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture()
def captured_logger():
    logger = logging.getLogger("detrs.test.stdio")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    collector = _RecordCollector()
    logger.addHandler(collector)
    yield logger, collector
    logging.getLogger("detrs.test.stdio").removeHandler(collector)


def test_relay_prints_logs_printed_lines(captured_logger):
    logger, collector = captured_logger

    with relay_prints(logger):
        print("loading annotations into memory...")
        print("index created!")

    assert [r.getMessage() for r in collector.records] == [
        "loading annotations into memory...",
        "index created!",
    ]


def test_relay_prints_emits_lines_while_block_is_open(captured_logger):
    logger, collector = captured_logger

    with relay_prints(logger):
        print("running per image evaluation...")
        assert collector.records[-1].getMessage() == "running per image evaluation..."


def test_relay_prints_reports_print_caller_location(captured_logger):
    logger, collector = captured_logger

    with relay_prints(logger):
        print("locate me")

    record = collector.records[0]
    # stacklevel must point at the print() call, not at detrs/utils/stdio.py
    assert record.pathname == __file__
    with open(__file__) as source:
        assert source.readlines()[record.lineno - 1].strip() == 'print("locate me")'


def test_relay_prints_restores_stdout_and_logs_after_exception(captured_logger):
    logger, collector = captured_logger
    original_stdout = sys.stdout

    with pytest.raises(RuntimeError):
        with relay_prints(logger):
            print("before failure")
            raise RuntimeError("boom")

    assert sys.stdout is original_stdout
    assert [r.getMessage() for r in collector.records] == ["before failure"]


def test_relay_prints_skips_blank_lines(captured_logger):
    logger, collector = captured_logger

    with relay_prints(logger):
        print()
        print("   ")
        print("kept")

    assert [r.getMessage() for r in collector.records] == ["kept"]


def test_relay_prints_does_not_recurse_when_logger_writes_to_stdout(
    captured_logger, capsys
):
    logger, _ = captured_logger
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    try:
        with relay_prints(logger):
            logger.info("own message")
    finally:
        logger.removeHandler(stdout_handler)

    assert "own message" in capsys.readouterr().out
