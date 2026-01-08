from veomni.trainer import VLMTrainer
from veomni.trainer.vlm_trainer import Arguments
from veomni.utils.arguments import parse_args


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = VLMTrainer(args)
    trainer.fit()
