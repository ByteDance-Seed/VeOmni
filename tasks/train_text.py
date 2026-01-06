from dataclasses import dataclass, field

from veomni.trainer import TextTrainer
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = TextTrainer(args)
    trainer.fit()
