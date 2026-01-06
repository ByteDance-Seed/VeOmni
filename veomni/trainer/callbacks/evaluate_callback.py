from typing import TYPE_CHECKING

from veomni.trainer.callbacks.base import TrainerState

from .base import Callback


if TYPE_CHECKING:
    from ..base import Arguments


class EvaluateCallback(Callback):
    def on_epoch_end(self, state: TrainerState, **kwargs):
        args: "Arguments" = self.trainer.args
        if args.train.eval_epochs and (state.epoch + 1) % args.train.eval_epochs == 0:
            self._evaluate(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args
        if args.train.eval_steps and (state.global_step + 1) % args.train.eval_steps == 0:
            self._evaluate(state)

    def _evaluate(self, state: TrainerState):
        # TODO: implement evaluate
        pass
