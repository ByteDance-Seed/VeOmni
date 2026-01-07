from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
from transformers import ProcessorMixin

from ..data import build_multimodal_chat_template
from ..data.chat_template import ChatTemplate
from ..data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ..data.multimodal import conv_preprocess
from ..data.multimodal.image_utils import fetch_images
from ..data.multimodal.video_utils import fetch_videos
from ..models import build_processor
from ..utils import helper
from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments
from .base import BaseTrainer


logger = helper.create_logger(__name__)
MAX_PIXELS = 768 * 28 * 28


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )


@dataclass
class MyDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def process_sample_qwen2_5_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example with qwen2_5_vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    )  # source_name if use multisource_dataset
    conversations = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    if "images" in sample:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num
    if "videos" in sample:
        videos, _ = fetch_videos(sample["videos"], **kwargs)
        video_inputs = processor.image_processor(images=None, videos=videos, return_tensors="pt")
        video_grid_thw = video_inputs["video_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]  # (dim, 1, seq_length)
    # Squeezed to (dim, seq_len) for later collator processing
    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]


def process_sample_qwen3_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example with qwen3_vl's pre-processor.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source"]
    )  # source_name if use multisource_dataset
    conversations = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    if "images" in sample:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num
    if "videos" in sample:
        videos, _ = fetch_videos(sample["videos"], **kwargs)
        video_inputs = processor.video_processor(images=None, videos=videos, return_tensors="pt")
        video_grid_thw = video_inputs["video_grid_thw"]
        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]  # (dim, 1, seq_length)
    # Squeezed to (dim, seq_len) for later collator processing
    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]


class VLMTrainer(BaseTrainer):
    def build_model_assets(self):
        self.processor = build_processor(self.args.model.tokenizer_path, max_pixels=MAX_PIXELS)
        self.chat_template = build_multimodal_chat_template(self.args.data.chat_template, self.processor.tokenizer)
        self.model_assets = [self.chat_template]

    def freeze_module(self):
        args: Arguments = self.args
        self.fsdp_kwargs = {}
        if args.train.freeze_vit:
            self.model.visual.requires_grad_(False)
            if args.train.data_parallel_mode == "fsdp1":
                self.fsdp_kwargs["use_orig_params"] = True

    def build_param_groups(self):
        args: Arguments = self.args
        vit_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "visual" in name:
                    vit_params.append(param)
                else:
                    other_params.append(param)

        return [{"params": vit_params, "lr": args.train.vit_lr}, {"params": other_params, "lr": args.train.lr}]

    def build_data_transform(self):
        args: Arguments = self.args
        position_id_func = self.model.get_position_id_func()
        if self.model_config.model_type in ("qwen3_vl", "qwen3_vl_moe"):
            process_function = process_sample_qwen3_vl
        elif self.model_config.model_type == "qwen2_5_vl":
            process_function = process_sample_qwen2_5_vl
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_config.model_type}.")
        data_transform = partial(
            process_function,
            processor=self.processor,
            chat_template=self.chat_template,
            position_id_func=position_id_func,
            **args.data.mm_configs,
        )
        return data_transform
