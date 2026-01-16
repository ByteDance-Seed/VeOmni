from veomni.arguments import parse_args
from veomni.trainer import TextTrainer
from veomni.trainer.text_trainer import Arguments


if __name__ == "__main__":
    args = parse_args(Arguments)
    trainer = TextTrainer(args)
    trainer.fit()
