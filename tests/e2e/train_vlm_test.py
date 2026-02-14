import json
import os
from collections import defaultdict
from typing import Dict

import torch

from veomni.arguments import parse_args
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.trainer.vlm_trainer import Arguments, VLMTrainer


os.environ["NCCL_DEBUG"] = "OFF"


def process_dummy_example(
    example: dict,
    **kwargs,
):
    example = {key: torch.tensor(v) for key, v in example.items()}
    return [example]


class TestVLMTrainer(VLMTrainer):
    def _init_callbacks(self):
        super()._init_callbacks()
        self.callbacks.add(LogDictSaveCallback(self))

    def build_model_assets(self):
        return []

    def build_data_transform(self):
        return process_dummy_example


class LogDictSaveCallback(Callback):
    def __init__(self, trainer: TestVLMTrainer) -> None:
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
    args = parse_args(Arguments)
    trainer = TestVLMTrainer(args)
    trainer.fit()
