from veomni.arguments import OmniArguments, parse_omni_args
from veomni.trainer.omni import OmniTrainer


if __name__ == "__main__":
    args = parse_omni_args(OmniArguments, preload_path_fields=("model.modules",))
    trainer = OmniTrainer(args)
    trainer.train()
