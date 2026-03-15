import json
import os
from collections import defaultdict
from typing import Any, Dict

import torch

from veomni.arguments import parse_args
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments


os.environ["NCCL_DEBUG"] = "OFF"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def process_dummy_example(
    example: dict,
    **kwargs,
):
    example = {key: torch.tensor(v) for key, v in example.items()}
    return [example]


class LogDictSaveCallback(Callback):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.log_dict = defaultdict(list)

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        self.log_dict["loss"].append(loss)
        for key, value in loss_dict.items():
            self.log_dict[key].append(value)
        self.log_dict["grad_norm"].append(grad_norm)

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.args.train.global_rank == 0:
            output_dir = self.trainer.args.train.checkpoint.output_dir
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "log_dict.json"), "w") as f:
                json.dump(self.log_dict, f, indent=4)


class TestDiTTrainer(DiTTrainer):
    """DiTTrainer subclass for SP-alignment testing.

    Uses ``WanT2VDataset`` to provide deterministic, pre-embedded batches
    directly to the model (no condition model required).  All SP ranks receive
    identical inputs because the data is generated from a fixed seed.
    """

    def __init__(self, args: VeOmniDiTArguments):
        args.train.training_task = "offline_training"
        super().__init__(args)
        self.base._log_callback = LogDictSaveCallback(self.base)

    # ------------------------------------------------------------------
    # No condition model needed – data arrives in model-ready format.
    # ------------------------------------------------------------------
    def _build_condition_model(self, condition_model_type: str) -> None:
        self.condition_model = None

    def _freeze_model_module(self) -> None:
        self.base.lora = False

    def _build_model_assets(self) -> None:
        self.base.model_assets = [self.base.model.config]

    def _build_data_transform(self) -> None:
        self.base.data_transform = process_dummy_example

    # ------------------------------------------------------------------
    # Skip condition model processing – batch is already model-ready.
    # ------------------------------------------------------------------
    def forward_backward_step(self, micro_batch: Dict[str, Any]) -> tuple:
        micro_batch = self.preforward(micro_batch)
        with self.base.model_fwd_context:
            outputs = self.base.model(**micro_batch)

        loss, loss_dict = self.postforward(outputs, micro_batch)

        with self.base.model_bwd_context:
            loss.backward()

        return loss, loss_dict

    def on_train_end(self):
        super().on_train_end()
        self.base._log_callback.on_train_end(self.base.state)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        super().on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.base._log_callback.on_step_end(
            self.base.state, loss=loss, loss_dict=loss_dict or {}, grad_norm=grad_norm or 0.0
        )


if __name__ == "__main__":
    args: VeOmniDiTArguments = parse_args(VeOmniDiTArguments)
    trainer = TestDiTTrainer(args)
    trainer.train()
