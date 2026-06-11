from veomni.arguments import OmniArguments, parse_omni_args
from veomni.trainer.omni_inferencer import OmniInferencer


def main() -> None:
    args = parse_omni_args(
        OmniArguments,
        preload_path_fields=("model.modules", "infer.modules"),
    )
    inferencer = OmniInferencer(args)
    inferencer.generate()


if __name__ == "__main__":
    main()
