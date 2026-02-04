from veomni.arguments import parse_args
from veomni.trainer.text_trainer import TextTrainer, VeOmniArguments


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = TextTrainer(args)
    trainer.fit()
