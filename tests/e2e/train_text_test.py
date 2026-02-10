import os

import torch

from veomni.arguments import parse_args
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


os.environ["NCCL_DEBUG"] = "OFF"


def process_dummy_example(
    example: dict,
    **kwargs,
):
    example = {key: torch.tensor(v) for key, v in example.items()}
    return [example]


class TestTextTrainer(TextTrainer):
    def build_model_assets(self):
        return []

    def build_data_transform(self):
        return process_dummy_example


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = TestTextTrainer(args)
    trainer.fit()
