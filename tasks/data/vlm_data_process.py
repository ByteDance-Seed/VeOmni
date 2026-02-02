"""
Sample transformation module for Vision-Language Models (VLMs).

This module provides process_sample functions for different VLM variants,
extracted from training scripts for better extensibility and reusability.

Functions:
    process_sample_qwen2_5_vl: Process samples for Qwen2.5-VL models
    process_sample_qwen3_vl: Process samples for Qwen3-VL models
"""

from typing import TYPE_CHECKING, Any, Callable, Dict

import torch

from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.multimodal import conv_preprocess
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.video_utils import fetch_videos, fetch_videos_metadata


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate


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
        kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
    )  # source_name if use multisource_dataset
    conversations = sample["conversations"] if ("conversations" in sample and sample["conversations"]) else sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    if "images" in sample and sample["images"]:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num
    if "videos" in sample and sample["videos"]:
        videos, _ = fetch_videos(sample["videos"], **kwargs)
        video_inputs = processor.video_processor(videos=videos, return_tensors="pt")
        video_grid_thw = video_inputs["video_grid_thw"]
        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }
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
    Processes a multimodal example using the Qwen3-VL pre-processor.
    """

    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
    )  # 'source_name' is used if using a multisource dataset
    conversations = sample["conversations"] if ("conversations" in sample and sample["conversations"]) else sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None

    tokenized_example = {}
    if "images" in sample and sample["images"]:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num
        tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    if "videos" in sample and sample["videos"]:
        videos, metadata, _, _ = fetch_videos_metadata(sample["videos"], **kwargs)
        # Process videos without resizing or sampling frames initially
        video_inputs = processor.video_processor(
            videos=videos, video_metadata=metadata, return_tensors="pt", return_metadata=True
        )
        video_grid_thw = video_inputs["video_grid_thw"]
        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

        # Extract metadata for use in the chat template
        video_metadata = video_inputs.pop("video_metadata")

        # Uses Qwen3-VL chat template encoding with video metadata
        tokenized_example = chat_template.encode_messages(
            conversations, token_num_inputs, video_metadata=video_metadata
        )

    if not tokenized_example:
        tokenized_example = chat_template.encode_messages(conversations)

    # Ensure all values are tensors
    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }

    # Generate 3D position IDs and squeeze for the collator
    input_ids = tokenized_example["input_ids"]
    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]  # Returns (dim, 1, seq_length)

    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]
