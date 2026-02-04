from veomni.arguments import parse_args
from veomni.trainer.vlm_rl_trainer import Arguments, VLMRLTrainer


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMRLTrainer(args)
    trainer.fit()
