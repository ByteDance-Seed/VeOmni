from veomni.arguments import parse_args
from veomni.trainer import VLMTrainer
from veomni.trainer.vlm_trainer import Arguments


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMTrainer(args)
    trainer.fit()
