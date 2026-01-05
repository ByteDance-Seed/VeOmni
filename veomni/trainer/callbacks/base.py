"""
Base Callback class for Trainer.

This module provides the BaseCallback base class which defines all available
hook points during training. Subclasses can override specific methods to customize
behavior at different stages of training.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseCallback(ABC):
    """
    Base class for all trainer callbacks.

    A callback is a set of functions that get called at various stages of training.
    Subclasses can override any of the following methods:

    evaluate: evaluation callback
    on_log: logging callback
    """

    @abstractmethod
    def evaluate(self, trainer, **kwargs) -> None:
        """for evaluation callback."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def on_log(self, trainer, **kwargs) -> Dict[str, Any]:
        """for metrics callback."""
        raise NotImplementedError("Subclasses must implement this method.")
