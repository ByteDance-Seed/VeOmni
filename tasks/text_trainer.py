"""
Example usage of BaseTrainer for creating custom trainers.

This file demonstrates how to create a custom trainer by inheriting from BaseTrainer.
"""

from dataclasses import dataclass, field

from veomni.trainer import Trainer
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args


class TextTrainer(Trainer):
    """
    Example custom trainer that extends BaseTrainer.

    This trainer customizes the post_dataloading_process method to add
    custom data preprocessing.
    """

    def post_init(self) -> None:
        """
        Custom post-initialization setup.

        This is called after all base trainer components are initialized.
        """
        super().post_init()
        # Add custom initialization logic here
        # For example: load custom checkpoints, set up custom metrics, etc.
        print("Custom trainer initialized!")


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = TextTrainer(args)
    trainer.fit()
