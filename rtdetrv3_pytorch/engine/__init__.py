"""
Engine module for RT-DETRv3

Contains training, evaluation, and inference utilities.
"""

from .evaluator import COCOEvaluator, build_coco_evaluator
from .optimizer import (
    build_optimizer,
    build_lr_scheduler,
    clip_gradients,
    LinearWarmupScheduler,
    MultiStepLRWithWarmup
)
from .trainer import Trainer

__all__ = [
    'COCOEvaluator',
    'build_coco_evaluator',
    'build_optimizer',
    'build_lr_scheduler',
    'clip_gradients',
    'LinearWarmupScheduler',
    'MultiStepLRWithWarmup',
    'Trainer'
]
