from veomni.arguments import parse_args
from veomni.trainer.base_rl_trainer import BaseRLTrainer
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


class TextRLTrainer(BaseRLTrainer, TextTrainer):
    pass


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = TextRLTrainer(args)
    trainer.fit()
