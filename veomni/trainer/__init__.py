"""
Trainer module for distributed model training.

This module provides the base trainer class and related utilities for
training large-scale models with various parallelism strategies.

Architecture:
    BaseTrainer (abstract base)
        └── Trainer (omni-modality training implementation)

    BaseCallback (callback base class)
        └── TrainerCallback (general training metrics)

Usage:
    # Basic training
    from veomni.trainer import Trainer
    trainer = Trainer(args)
    trainer.fit()

    # Add custom callbacks after construction
    from veomni.trainer.callbacks import BaseCallback

    trainer = Trainer(args)
    trainer.add_callback(MyCallback())
    trainer.fit()
"""

from . import callbacks
from .base import BaseTrainer
from .trainer import Trainer


__all__ = [
    "BaseTrainer",
    "Trainer",
    "callbacks",
]
