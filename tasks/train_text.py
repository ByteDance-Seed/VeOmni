from veomni.trainer import TextTrainer
from veomni.trainer.text_trainer import Arguments
from veomni.utils.arguments import parse_args


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = TextTrainer(args)
    trainer.fit()
