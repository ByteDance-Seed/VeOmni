from veomni.arguments import parse_args
from veomni.trainer.omni_inferencer import OmniInferenceArguments, OmniInferencer


def main() -> None:
    args = parse_args(OmniInferenceArguments)
    inferencer = OmniInferencer(args)
    inferencer.generate()


if __name__ == "__main__":
    main()
