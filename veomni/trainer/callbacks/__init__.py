"""
Trainer Callbacks module.

Provides callback system for customizing trainer behavior at various stages of training.
"""

from .base import BaseCallback
from .text_callback import TrainerCallback


__all__ = [
    "BaseCallback",
    "TrainerCallback",
]
