"""Text Trainer Callback for general text training metrics."""

from typing import TYPE_CHECKING, Any

from ...utils.logging import get_logger
from .base import BaseCallback


logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..base import BaseTrainer


class TrainerCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def evaluate(
        self,
        trainer: "BaseTrainer",
        **kwargs,
    ) -> Any:
        if hasattr(trainer, "eval_dataloader"):
            logger.info_rank0(
                f" on_evaluate at step {trainer.global_step}, eval_dataloader={len(trainer.eval_dataloader)}"
            )
            return None

    def on_log(
        self,
        trainer: "BaseTrainer",
        **kwargs,
    ) -> Any:
        return {}
