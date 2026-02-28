import json
import os
from collections import defaultdict
from typing import Dict

import torch

from veomni.arguments import parse_args
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


os.environ["NCCL_DEBUG"] = "OFF"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VEOMNI_USE_LIGER_KERNEL"] = "0"


def process_dummy_example(
    example: dict,
    **kwargs,
):
    example = {key: torch.tensor(v) for key, v in example.items()}
    return [example]


class TestTextTrainer(TextTrainer):
    def __init__(self, args: VeOmniArguments):
        super().__init__(args)
        self.base.logdictsave_callback = LogDictSaveCallback(self.base)

    def _build_model_assets(self):
        self.base.model_assets = []

    def _build_data_transform(self):
        self.base.data_transform = process_dummy_example

    def on_train_end(self):
        super().on_train_end()
        self.base.logdictsave_callback.on_train_end(self.base.state)

    def on_step_end(self, **kwargs):
        super().on_step_end(**kwargs)
        self.base.logdictsave_callback.on_step_end(self.base.state, **kwargs)


class LogDictSaveCallback(Callback):
    def __init__(self, trainer: TestTextTrainer) -> None:
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
            output_dir = self.trainer.args.train.output_dir
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "log_dict.json"), "w") as f:
                json.dump(self.log_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = TestTextTrainer(args)
    trainer.train()
