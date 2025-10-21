"""
RT-DETRv3 PyTorch Training Engine

Contains training and evaluation components:
- trainer: Training loop and management
- evaluator: Model evaluation on validation sets
- callbacks: Training callbacks for logging and checkpointing
"""

from .trainer import Trainer
from .callbacks import (
    Callback, ComposeCallback, LogPrinter, Checkpointer,
    LearningRateLogger, BestModelSaver
)

__all__ = [
    'Trainer',
    'Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer',
    'LearningRateLogger', 'BestModelSaver'
]
