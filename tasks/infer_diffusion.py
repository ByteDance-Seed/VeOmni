import json
import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from tqdm import tqdm

from veomni.data.multimodal.image_utils import fetch_images
from veomni.dit_trainer import DiTBaseGenerator, DiTTrainerRegistry
from veomni.utils import helper
from veomni.utils.arguments import InferArguments, ModelArguments, parse_args, save_args


logger = helper.create_logger(__name__)


def read_raw_data(data_path: str, negative_prompts_path: str):
    with open(negative_prompts_path, encoding="utf-8") as f:
        negative_text = f.readline().strip()
    raw_data = []

    assert data_path.endswith(".json"), f"Not support reading data path: {data_path}"
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            raw_data.append(
                {
                    "prompt": [item["user_prompt"]],
                    "image": fetch_images(item["image_file"]),  # convert to image
                    "negative_prompts": negative_text,
                }
            )
    return raw_data


@dataclass
class MyModelArguments(ModelArguments):
    lora_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for lora."},
    )
    generator_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for generator."},
    )
    trainer_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for trainer."},
    )


@dataclass
class MyInferArguments(InferArguments):
    data_path: str = field(
        default="",
        metadata={"help": "Path to the generate data."},
    )
    negative_prompts_path: str = field(
        default="",
        metadata={"help": "Path to the negative prompts."},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "Output directory."},
    )
    enable_full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full determinism."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for reproducibility."},
    )
    lora_scale: float = field(
        default=1.0,
        metadata={"help": "LoRA scale"},
    )
    lora_model_path: str = field(
        default=None,
        metadata={"help": "LoRA model path"},
    )


@dataclass
class Arguments:
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    infer: "MyInferArguments" = field(default_factory=MyInferArguments)


def main():
    args = parse_args(Arguments)
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.infer.seed, args.infer.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    helper.enable_third_party_logging()
    save_args(args, args.infer.output_dir)

    raw_data_list = read_raw_data(args.infer.data_path, args.infer.negative_prompts_path)

    logger.info_rank0("Prepare generator")

    if args.model.lora_config and args.infer.lora_model_path:
        lora_config = args.model.lora_config
        lora_config.update(
            lora_scale=args.infer.lora_scale,
            lora_adapter=args.infer.lora_model_path,
        )
    else:
        lora_config = None
    generator: DiTBaseGenerator = DiTTrainerRegistry.create(
        model_path=args.infer.model_path,
        lora_config=lora_config,
        condition_model_path=args.model.trainer_config["condition_model_path"],
        condition_model_cfg=args.model.trainer_config.get("condition_model_cfg", {}),
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        **args.model.generator_config,
    )

    processor = generator.processor

    os.makedirs(args.infer.output_dir, exist_ok=True)
    for i, raw_data in enumerate(tqdm(raw_data_list)):
        processed_data = processor.preprocess_infer(raw_data)
        outputs = generator.forward(processed_data)
        sample_output_dir = os.path.join(args.infer.output_dir, f"sample_{i}")
        processor.save_outputs(outputs, sample_output_dir)


if __name__ == "__main__":
    main()
