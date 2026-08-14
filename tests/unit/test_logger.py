import logging

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
