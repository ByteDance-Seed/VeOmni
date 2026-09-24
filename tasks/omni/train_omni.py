from veomni.arguments.omni_arguments_types import OmniArguments
from veomni.arguments.omni_parser import parse_omni_args
from veomni.trainer.omni import OmniTrainer


if __name__ == "__main__":
    args = parse_omni_args(OmniArguments, preload_path_fields=("model.model_config.modules",))
    trainer = OmniTrainer(args)
    trainer.train()
