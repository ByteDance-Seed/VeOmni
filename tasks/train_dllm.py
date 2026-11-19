from veomni.arguments import parse_args
from veomni.trainer.dllm_trainer import DllmTrainer, VeOmniArguments


if __name__ == "__main__":
    args = parse_args(VeOmniArguments)
    trainer = DllmTrainer(args)
    trainer.train()
