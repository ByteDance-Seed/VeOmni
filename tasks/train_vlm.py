from veomni.arguments import parse_args
from veomni.trainer.vlm_trainer import Arguments, VLMTrainer


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMTrainer(args)
    trainer.fit()
