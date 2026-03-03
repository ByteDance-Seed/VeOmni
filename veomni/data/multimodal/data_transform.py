# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from ...utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ..chat_template import ChatTemplate


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
    from . import conv_preprocess
    from .image_utils import fetch_images
    from .video_utils import fetch_videos

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
        video_inputs = processor.video_processor(images=None, videos=videos, return_tensors="pt")
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
    from . import conv_preprocess
    from .image_utils import fetch_images
    from .video_utils import fetch_videos_metadata

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


# ---------------------------------------------------------------------------
# LIBERO robotic action prediction transform for Qwen3-VL
# ---------------------------------------------------------------------------


def process_libero_sample_qwen3_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    task_descriptions: Dict[int, str],
    prompt_template: str = "Predict the next actions for the robot task: {task}",
    obs_index: int = -1,
    **kwargs,
):
    """Transform a single LiberoYoumuDataset sample for Qwen3VLForConditionalGenerationAction.

    Takes the raw dict from ``LiberoYoumuDataset.__getitem__`` (images, state,
    action, episode_index) and produces the tensor dict expected by the VeOmni
    training pipeline (input_ids, pixel_values, position_ids, image_mask, …)
    plus the two action-specific fields (observation_state, labels).

    Args:
        sample: Dict returned by ``LiberoYoumuDataset.__getitem__``.  Expected
            keys: ``observation.images.image`` (obs_len, H, W, 3) uint8,
            ``observation.state`` (obs_len, state_dim) float32,
            ``action`` (pred_len, action_dim) float32,
            ``episode_index`` int.
        processor: A Qwen3-VL processor (provides ``image_processor`` and
            ``tokenizer``).
        position_id_func: Model-specific function that computes 3D position IDs
            (obtained via ``model.get_position_id_func()``).
        task_descriptions: Mapping from ``episode_index`` → task description
            string.  Built from the LIBERO metadata ``tasks`` column.
        prompt_template: Text template with a ``{task}`` placeholder that will
            be filled with the task description.
        obs_index: Which observation frame to use as the image input.
            Default ``-1`` (the anchor / most recent frame).
    """
    # --- Image processing ---
    # Select a single observation frame as the image input.
    # LeRobot returns (obs_len, C, H, W) float32 in [0, 1]; Youmu returns (obs_len, H, W, C) uint8.
    images_tensor = sample["observation.images.image"]  # (obs_len, ...)
    obs_frame = images_tensor[obs_index]  # (C, H, W) float32 or (H, W, C) uint8
    from PIL import Image as PILImage

    if obs_frame.ndim == 3 and obs_frame.shape[0] in (1, 3):
        # CHW float32 [0,1] → HWC uint8 [0,255]
        obs_image = (obs_frame.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    else:
        obs_image = obs_frame.numpy()
    pil_image = PILImage.fromarray(obs_image)

    image_inputs = processor.image_processor(images=[pil_image], return_tensors="pt")
    image_grid_thw = image_inputs["image_grid_thw"]  # (1, 3)
    merge_length = processor.image_processor.merge_size**2
    image_token_num = image_grid_thw.prod(dim=-1) // merge_length  # (1,)

    # --- Text tokenization ---
    ep_idx = sample["episode_index"]
    task_desc = task_descriptions.get(ep_idx, "perform the task")
    prompt_text = prompt_template.format(task=task_desc)

    # Build a minimal input sequence: <image_placeholder> + prompt text tokens
    # The image placeholder tokens will be replaced by visual features in the model.
    num_img_tokens = image_token_num[0].item()
    image_placeholder_ids = [IMAGE_INPUT_INDEX] * num_img_tokens

    text_token_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)

    input_ids = torch.tensor(image_placeholder_ids + text_token_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # --- Position IDs (3D RoPE) ---
    position_ids = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask.unsqueeze(0),
    )["position_ids"]  # (3, 1, seq_len)
    position_ids = position_ids.squeeze().clone()  # (3, seq_len)

    # --- Image / video masks ---
    image_mask = input_ids == IMAGE_INPUT_INDEX
    video_mask = input_ids == VIDEO_INPUT_INDEX
    # Zero-out placeholder token IDs (model uses mask to scatter visual features)
    input_ids[image_mask] = 0
    input_ids[video_mask] = 0

    # --- Append state placeholder token ---
    # A dummy token (id=0) at the end of the sequence; the model will replace
    # its embedding with the projected observation_state via state_mask.
    state_placeholder = torch.zeros(1, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, state_placeholder])
    attention_mask = torch.cat([attention_mask, torch.ones(1, dtype=attention_mask.dtype)])
    next_pos = position_ids[:, -1:] + 1  # (3, 1)
    position_ids = torch.cat([position_ids, next_pos], dim=-1)
    image_mask = torch.cat([image_mask, torch.zeros(1, dtype=image_mask.dtype)])
    video_mask = torch.cat([video_mask, torch.zeros(1, dtype=video_mask.dtype)])
    # state_mask: True only for the appended state token
    state_mask = torch.zeros(len(input_ids), dtype=torch.bool)
    state_mask[-1] = True

    # --- Observation state & action labels ---
    # Use the last obs_len state frames; model will receive (obs_len, state_dim)
    # but we flatten to the last frame to match the single-token state injection.
    observation_state = sample["observation.state"][obs_index]  # (state_dim,)
    labels = sample["action"]  # (pred_len, action_dim)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_mask": image_mask,
        "video_mask": video_mask,
        "state_mask": state_mask,
        "observation_state": observation_state,
        "labels": labels,
    }
    result.update(image_inputs)  # pixel_values, image_grid_thw

    return [result]


def load_libero_task_descriptions(meta_path: str) -> Dict[int, str]:
    """Load episode_index → task description mapping from LIBERO metadata.

    Supports two formats:
    - **Parquet**: ``meta/episodes/chunk-000/file-000.parquet`` with columns
      ``episode_index`` and ``tasks`` (list of strings).
    - **JSONL**: ``meta/episodes.jsonl`` with one JSON object per line
      containing ``episode_index`` (int) and ``tasks`` (list of strings).

    The format is auto-detected from the file extension.

    Args:
        meta_path: Path to the LIBERO episode metadata file (parquet or JSONL).

    Returns:
        Dict mapping ``episode_index`` to the first task description string.
    """
    if meta_path.endswith(".jsonl"):
        return _load_libero_task_descriptions_jsonl(meta_path)
    return _load_libero_task_descriptions_parquet(meta_path)


def _load_libero_task_descriptions_parquet(meta_path: str) -> Dict[int, str]:
    """Load task descriptions from a parquet episode metadata file.

    Args:
        meta_path: Path to the parquet file with ``episode_index`` and ``tasks`` columns.

    Returns:
        Dict mapping ``episode_index`` to the first task description string.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(meta_path, columns=["episode_index", "tasks"])
    task_map: Dict[int, str] = {}
    for i in range(len(table)):
        ep_idx = table["episode_index"][i].as_py()
        tasks_list = table["tasks"][i].as_py()
        # Each episode has a list of task strings; take the first one.
        task_map[ep_idx] = tasks_list[0] if tasks_list else "perform the task"
    return task_map


def _load_libero_task_descriptions_jsonl(meta_path: str) -> Dict[int, str]:
    """Load task descriptions from a JSONL episode metadata file.

    Each line is a JSON object with ``episode_index`` (int) and ``tasks``
    (list of strings).  Falls back to ``"perform the task"`` if ``tasks``
    is missing or empty.

    Args:
        meta_path: Path to the JSONL file (e.g. ``meta/episodes.jsonl``).

    Returns:
        Dict mapping ``episode_index`` to the first task description string.
    """
    import json

    task_map: Dict[int, str] = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ep_idx = record["episode_index"]
            tasks_list = record.get("tasks", [])
            task_map[ep_idx] = tasks_list[0] if tasks_list else "perform the task"
    return task_map


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
    from . import conv_preprocess
    from .audio_utils import fetch_audios
    from .image_utils import fetch_images
    from .video_utils import fetch_videos

    source = (
        kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
    )  # source_name if use multisource_dataset
    conversations = sample["conversations"] if ("conversations" in sample and sample["conversations"]) else sample
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
