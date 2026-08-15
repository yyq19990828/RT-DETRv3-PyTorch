import logging
from pathlib import Path

from detrs.utils import logger as logger_module


def test_setup_logger_honors_torchrun_rank_before_process_group(monkeypatch):
    name = "rtdetrv3.test.nonzero_rank"
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(logger_module.dist, "is_initialized", lambda: False)

    created_logger = logger_module.setup_logger(name)

    assert len(created_logger.handlers) == 1
    assert isinstance(created_logger.handlers[0], logging.NullHandler)

    logger_module.logger_initialized.remove(name)
    logging.Logger.manager.loggerDict.pop(name, None)


def test_display_path_is_relative_to_the_repository():
    display = logger_module._display_path(str(Path(__file__).resolve()))

    assert display == "tests/unit/test_logger.py"


def test_display_path_strips_site_packages_prefix():
    import pycocotools.cocoeval

    display = logger_module._display_path(pycocotools.cocoeval.__file__)

    assert display == "pycocotools/cocoeval.py"


def test_display_path_falls_back_to_the_basename(tmp_path):
    outside = tmp_path / "somewhere" / "module.py"

    assert logger_module._display_path(str(outside)) == "module.py"


def test_console_handler_renders_project_relative_location(capsys):
    name = "rtdetrv3.test.relative_path"
    logger = logger_module.setup_logger(name)

    logger.info("locate me")

    assert "tests/unit/test_logger.py" in capsys.readouterr().out

    logger_module.logger_initialized.remove(name)
    logging.Logger.manager.loggerDict.pop(name, None)
