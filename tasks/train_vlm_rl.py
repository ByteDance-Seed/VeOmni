from veomni.trainer import VLMRLTrainer
from veomni.trainer.vlm_rl_trainer import Arguments
from veomni.utils.arguments import parse_args


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMRLTrainer(args)
    trainer.fit()
