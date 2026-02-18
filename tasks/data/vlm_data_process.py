"""
Sample transformation module for Vision-Language Models (VLMs).

This module provides process_sample functions for different VLM variants,
extracted from training scripts for better extensibility and reusability.

Functions:
    process_sample_qwen2_5_vl: Process samples for Qwen2.5-VL models
    process_sample_qwen3_vl: Process samples for Qwen3-VL models
    get_omni_token_ids: Resolve image/video/audio pad token IDs from processor vocab
    process_sample_qwen_omni: Process samples for Qwen2.5-Omni and Qwen3-Omni-MoE models
"""

from typing import TYPE_CHECKING, Any, Callable, Dict

import torch

from veomni.data.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.multimodal import conv_preprocess
from veomni.data.multimodal.audio_utils import fetch_audios
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


QWEN_OMNI_SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def get_omni_token_ids(processor: "ProcessorMixin") -> tuple[int, int, int]:
    """
    Resolve (image_token_id, video_token_id, audio_token_id) by reading from the processor's
    tokenizer vocab. Supports both Qwen2.5-Omni and Qwen3-Omni-MoE:
      Qwen2.5-Omni:   image=151655 (<|IMAGE|>),     video=151656 (<|VIDEO|>),     audio=151646 (<|AUDIO|>)
      Qwen3-Omni-MoE: image=151655 (<|image_pad|>), video=151656 (<|video_pad|>), audio=151675 (<|audio_pad|>)
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    # Qwen2.5-Omni uses <|IMAGE|>/<|VIDEO|>/<|AUDIO|>; https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/tokenizer_config.json
    # Qwen3-Omni uses <|image_pad|>/<|video_pad|>/<|audio_pad|>; https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json
    image_token_id = vocab.get("<|image_pad|>", vocab.get("<|IMAGE|>"))
    video_token_id = vocab.get("<|video_pad|>", vocab.get("<|VIDEO|>"))
    audio_token_id = vocab.get("<|audio_pad|>", vocab.get("<|AUDIO|>"))
    if image_token_id is None:
        raise ValueError("Cannot find image token (<|image_pad|> or <|IMAGE|>) in tokenizer vocab.")
    if video_token_id is None:
        raise ValueError("Cannot find video token (<|video_pad|> or <|VIDEO|>) in tokenizer vocab.")
    if audio_token_id is None:
        raise ValueError("Cannot find audio token (<|audio_pad|> or <|AUDIO|>) in tokenizer vocab.")
    return image_token_id, video_token_id, audio_token_id


def _process_sample_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    **kwargs,
) -> list[Dict[str, Any]]:
    """
    Shared implementation for Omni model sample processing (Qwen2.5-Omni and Qwen3-Omni-MoE).
    Token IDs for image/video/audio placeholders are passed explicitly to support
    different tokenizer vocabs across model versions.
    """
    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
    )  # source_name if use multisource_dataset
    conversations = (
        sample["conversations"] if ("conversations" in sample and len(sample["conversations"]) > 0) else sample
    )
    conversations = conv_preprocess(source, conversations, **kwargs)
    input_conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": QWEN_OMNI_SYSTEM_MESSAGE,
                },
            ],
        },
    ]
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        tmp_conv = {
            "role": conversation[0],
            "content": contents,
        }
        input_conversations.append(tmp_conv)
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    images = sample.get("images", [])
    if images:
        images = fetch_images(images, **kwargs)
    else:
        images = []

    videos = sample.get("videos", [])
    if videos:
        videos, video_audios = fetch_videos(videos, **kwargs)
    else:
        videos, video_audios = [], []

    audios = sample.get("audios", [])
    if audios:
        audio_audios = fetch_audios(audios, **kwargs)
    else:
        audio_audios = []

    video_audios_iter = iter(video_audios)
    audio_audios_iter = iter(audio_audios)
    audios = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(next(video_audios_iter))
            elif content["type"] == "audio":
                audios.append(next(audio_audios_iter))

    model_inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data  # batch_feature to dict
    # process audio inputs:
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0  # filter videos without audios
        input_features = input_features[valid_mask].permute(0, 2, 1)[
            feature_attention_mask[valid_mask].bool()
        ]  # l, dim

        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None  # no video & no audio

    input_ids = model_inputs["input_ids"].squeeze(0)
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    audio_mask = input_ids == audio_token_id
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    model_inputs["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
        second_per_grids=model_inputs.pop("video_second_per_grid", None),
    )["position_ids"]  # (dim, l)

    model_inputs["position_ids"] = model_inputs["position_ids"].clone()
    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    user_token_id = vocab.get("user")
    assistant_token_id = vocab.get("assistant")
    if user_token_id is None or assistant_token_id is None:
        raise ValueError("Cannot find user/assistant tokens in tokenizer vocab.")
    user_start_index = torch.where(input_ids == user_token_id)[0].tolist()
    assistant_start_index = torch.where(input_ids == assistant_token_id)[0].tolist()
    user_start_index.append(len(input_ids) + 1)
    user_i = 0
    for assis_i in assistant_start_index:
        while user_start_index[user_i] < assis_i:
            user_i += 1
        labels[assis_i + 2 : user_start_index[user_i] - 1] = input_ids[assis_i + 2 : user_start_index[user_i] - 1]
    model_inputs["labels"] = labels
    return [model_inputs]


def process_sample_qwen_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example for Qwen-Omni family models (Qwen2.5-Omni and Qwen3-Omni-MoE).
    Token IDs are resolved dynamically from the processor vocab to support both variants:
      Qwen2.5-Omni:   image=151655 (<|IMAGE|>),     video=151656 (<|VIDEO|>),     audio=151646 (<|AUDIO|>)
      Qwen3-Omni-MoE: image=151655 (<|image_pad|>), video=151656 (<|video_pad|>), audio=151675 (<|audio_pad|>)
    """
    image_token_id, video_token_id, audio_token_id = get_omni_token_ids(processor)
    return _process_sample_omni(
        sample, processor, position_id_func, image_token_id, video_token_id, audio_token_id, **kwargs
    )
