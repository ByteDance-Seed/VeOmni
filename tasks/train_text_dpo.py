from veomni.arguments import VeOmniDPOArguments, parse_args
from veomni.trainer.dpo_trainer import TextDPOTrainer


if __name__ == "__main__":
    args = parse_args(VeOmniDPOArguments)
    trainer = TextDPOTrainer(args)
    trainer.train()
