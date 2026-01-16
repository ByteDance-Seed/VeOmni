from veomni.arguments import parse_args
from veomni.trainer import VLMRLTrainer
from veomni.trainer.vlm_rl_trainer import Arguments


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMRLTrainer(args)
    trainer.fit()
