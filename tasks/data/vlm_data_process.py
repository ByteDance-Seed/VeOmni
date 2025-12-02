"""
Sample transformation module for Vision-Language Models (VLMs).

This module provides process_sample functions for different VLM variants,
extracted from training scripts for better extensibility and reusability.

Functions:
    prepare_fa_kwargs_from_position_ids: Prepare flash attention kwargs for varlen attention
    process_sample_qwen2_5_vl: Process samples for Qwen2.5-VL models
    process_sample_qwen3_vl: Process samples for Qwen3-VL models
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Dict

import torch

from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.multimodal import conv_preprocess
from veomni.data.multimodal.image_utils import fetch_images
from veomni.data.multimodal.video_utils import fetch_videos


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate


def prepare_fa_kwargs_from_position_ids(position_ids: torch.Tensor):
    """
    Prepare-compute flash attention kwargs from position_ids for varlen flash attention.

    Qwen2.5-VL and Qwen3-VL note:
    - The model uses 3-D position ids (temporal, height, width) for vision tokens.
    Here we rely ONLY on the temporal channel semantics: within a single sequence,
    temporal ids are nondecreasing; when a new sequence begins, the temporal id
    resets and strictly drops (e.g., ... 100, 100, 101 | 0, 0, 0, 50, 50, ...).
    - Vision frames at the start of a sequence often share the same temporal id (many
    leading 0s). We DO NOT detect starts by "id == 0". Instead, we detect starts by
    a strict drop between adjacent tokens: pos[i] > pos[i+1] ⇒ i+1 is a new seq head.
    This works even if a sequence begins with many zeros.
    - Assumption: Each concatenated sequence has at least two items (text/image/video),
    so that a reset (drop) or a proper length can be inferred.
    """
    # Flatten to 1-D over the token dimension; the upstream caller must ensure that
    # this tensor corresponds to the temporal ids (or an equivalent 1-D monotone id).
    position_ids = position_ids.flatten()

    # Find boundaries where the temporal id strictly drops. Each drop marks the start
    # index of a NEW sequence within the concatenated batch stream.
    # Example: [0,0,0,50,50,100,100, 0,0,50] → drop at the transition 100→0
    seq_starts = torch.where(position_ids[:-1] > position_ids[1:])[0] + 1

    # Build cu_seq_lens (cumulative sequence lengths): always start at 0 and end at N.
    # We insert all detected start indices in order. This matches FlashAttention's varlen format.
    cu_seq_lens = torch.cat(
        (
            torch.tensor([0], device=position_ids.device, dtype=torch.int32),
            seq_starts.to(torch.int32),
            torch.tensor([position_ids.size(0)], device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = cu_seq_lens.diff().max().item()  # use cu_seq_lens to infer max_length, convert to int

    return {
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


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
    record_process_time = kwargs.get("record_process_time", False)
    if record_process_time:
        start_time = time.time()

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

    if record_process_time:
        process_time = time.time() - start_time
        tokenized_example["process_sample_time_sec"] = process_time

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
    record_process_time = kwargs.get("record_process_time", False)
    if record_process_time:
        start_time = time.time()

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

    if record_process_time:
        process_time = time.time() - start_time
        tokenized_example["process_sample_time_sec"] = process_time

    return [tokenized_example]
