import json
import os


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import torch
from tqdm import tqdm

from veomni.data.data_collator import DataCollator
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.video_utils import save_video_tensors_to_file
from veomni.dit_trainer import DiTBaseGenerator, DiTTrainerRegistry
from veomni.utils import helper
from veomni.utils.arguments import InferArguments, ModelArguments, parse_args, save_args


logger = helper.create_logger(__name__)


def read_raw_data(data_path: str, negative_prompts_path: str):
    with open(negative_prompts_path, encoding="utf-8") as f:
        negative_text = f.readline().strip()
    raw_data = []

    if data_path.endswith(".jsonl"):
        with open(data_path, encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                data["negative_prompts"] = negative_text
                raw_data.append(
                    {
                        "prompt": data["prompt"],
                        "image": fetch_images([data["image_bytes"].encode("latin-1")])[0],  # convert to image
                        "negative_prompts": negative_text,
                    }
                )
    elif data_path.endswith(".json"):
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                raw_data.append(
                    {
                        "prompt": [item["user_prompt"]],
                        "image": fetch_images([item["image_file"]])[0],  # convert to image
                        "negative_prompts": negative_text,
                    }
                )
    else:
        raise NotImplementedError(f"Not support reading data path: {data_path}")

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


@dataclass
class DiTDataCollator(DataCollator):
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            batch[key] = torch.cat(batch[key], dim=0)

        return batch


def main():
    args = parse_args(Arguments)
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    helper.set_seed(args.infer.seed, args.infer.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    helper.enable_third_party_logging()
    save_args(args, args.infer.output_dir)

    raw_data_list = read_raw_data(args.infer.data_path, args.infer.negative_prompts_path)

    logger.info_rank0("Prepare generator")

    args.model.lora_config.update(
        lora_scale=args.infer.lora_scale,
        lora_model_path=args.infer.lora_model_path,
    )
    generator: DiTBaseGenerator = DiTTrainerRegistry.create(
        model_path=args.infer.model_path,
        build_foundation_model_func=None,
        lora_config=args.model.lora_config,
        condition_model_path=args.model.trainer_config["condition_model_path"],
        condition_model_cfg=args.model.trainer_config["condition_model_cfg"],
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        **args.model.generator_config,
    )

    for i, raw_data in enumerate(tqdm(raw_data_list)):
        videos = generator.forward(raw_data)

        os.makedirs("output", exist_ok=True)
        for j, video in enumerate(videos):
            save_video_tensors_to_file(video.cpu().numpy(), f"{args.infer.output_dir}/pred_{i}_{j}.mp4")


if __name__ == "__main__":
    main()
