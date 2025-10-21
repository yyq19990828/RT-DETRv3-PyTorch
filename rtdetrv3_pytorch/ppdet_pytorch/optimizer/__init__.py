"""
RT-DETRv3 PyTorch Optimizer Module

Contains optimizer and learning rate scheduler builders, and EMA.
"""

from .ema import ModelEMA

__all__ = ['ModelEMA']
