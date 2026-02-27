from veomni.arguments import parse_args
from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer


if __name__ == "__main__":
    args = parse_args(VeOmniVLMArguments)
    trainer = VLMTrainer(args)
    trainer.fit()
