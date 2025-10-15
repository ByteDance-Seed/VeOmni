import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
from tqdm import tqdm

from veomni.data.data_collator import DataCollator
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.video_utils import save_video_tensors_to_file
from veomni.dit_trainer import DiTBaseGenerator, DiTTrainerRegistry
from veomni.models import build_foundation_model
from veomni.utils import helper
from veomni.utils.arguments import ModelArguments, parse_args, save_args


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
class MyDataArguments:
    generate_path: str = field(
        default="",
        metadata={"help": "Path to the generate data."},
    )
    negative_prompts_path: str = field(
        default="",
        metadata={"help": "Path to the negative prompts."},
    )


@dataclass
class MyGenerateArguments:
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


@dataclass
class Arguments:
    model: "MyModelArguments" = field(default_factory=MyModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    generate: "MyGenerateArguments" = field(default_factory=MyGenerateArguments)


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
    helper.set_seed(args.generate.seed, args.generate.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    helper.enable_third_party_logging()
    save_args(args, args.generate.output_dir)

    raw_data_list = read_raw_data(args.data.generate_path, args.data.negative_prompts_path)

    logger.info_rank0("Prepare generator")
    build_foundation_model_func = partial(
        build_foundation_model,
        torch_dtype="bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        force_use_huggingface=args.model.force_use_huggingface,
    )

    generator: DiTBaseGenerator = DiTTrainerRegistry.create(
        model_path=args.model.model_path,
        build_foundation_model_func=build_foundation_model_func,
        lora_config=args.model.lora_config,
        condition_model_path=args.model.trainer_config["condition_model_path"],
        condition_model_cfg=args.model.trainer_config["condition_model_cfg"],
        **args.model.generator_config,
    )

    for i, raw_data in enumerate(tqdm(raw_data_list)):
        videos = generator.forward(raw_data)

        os.makedirs("output", exist_ok=True)
        for j, video in enumerate(videos):
            save_video_tensors_to_file(video.cpu().numpy(), f"{args.generate.output_dir}/pred_{i}_{j}.mp4")


if __name__ == "__main__":
    main()
