from veomni.arguments import parse_args
from veomni.trainer.diffusion_gemma_trainer import DiffusionGemmaTrainer, VeOmniArguments


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = DiffusionGemmaTrainer(args)
    trainer.train()
