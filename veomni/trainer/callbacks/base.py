from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
    from ..base import BaseTrainer


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0


class Callback(ABC):
    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer

    def on_step_begin(self, state: TrainerState, micro_batches: List[List[Dict[str, Any]]] = None, **kwargs) -> None:
        pass

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        pass

    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class CallbackHandler:
    def __init__(self, callbacks: list[Callback]):
        self.callbacks = callbacks

    def call(self, event: str, state: TrainerState, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is not None:
                fn(state, **kwargs)

    def add(self, callback: Callback):
        self.callbacks.append(callback)
