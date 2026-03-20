from veomni.arguments import parse_args
from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments


if __name__ == "__main__":
    args = parse_args(VeOmniDiTArguments)
    trainer = DiTTrainer(args)
    trainer.train()
