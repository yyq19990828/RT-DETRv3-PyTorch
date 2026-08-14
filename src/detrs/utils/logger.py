"""Logging utilities for RT-DETRv3 PyTorch implementation

Adapted from PaddlePaddle's ppdet.utils.logger to maintain compatibility
while using PyTorch's distributed training APIs.
"""

import logging
import os
import sys
from typing import List, Optional, Union

import torch.distributed as dist

__all__ = ["setup_logger"]

logger_initialized = []


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

    # Console logging: only specified ranks
    if local_rank in log_ranks:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
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
