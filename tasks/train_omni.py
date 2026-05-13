from veomni.arguments import parse_args
from veomni.trainer.omni_trainer import OmniTrainer, VeOmniOmniArguments


if __name__ == "__main__":
    args = parse_args(VeOmniOmniArguments)
    trainer = OmniTrainer(args)
    trainer.train()
