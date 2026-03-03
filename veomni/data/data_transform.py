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

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Union

import torch

from veomni.utils.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.registry import Registry


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from .chat_template import ChatTemplate


DATA_TRANSFORM_REGISTRY = Registry("DataTransform")


def build_data_transform(transform_name: str, **kwargs) -> Callable:
    return partial(DATA_TRANSFORM_REGISTRY[transform_name], **kwargs)


def split_into_chunks(sequence: Sequence[int], chunk_size: int) -> List[List[int]]:
    """
    Splits a long sequence into chunks.
    """
    total_len = len(sequence)
    chunks = []
    for i in range(0, total_len, chunk_size):
        chunks.append(sequence[i : i + chunk_size])

    return chunks


@DATA_TRANSFORM_REGISTRY.register("plaintext")
def process_plaintext_example(
    example: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "content_split",
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokens = tokenizer.encode(text_example, add_special_tokens=False) + [tokenizer.eos_token_id]
    for input_ids in split_into_chunks(tokens, max_seq_len):
        examples.append(
            {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor([1] * len(input_ids)),
                "labels": torch.tensor(input_ids),
            }
        )

    return examples


@DATA_TRANSFORM_REGISTRY.register("conversation")
def process_conversation_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate",
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        text_example = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text_example = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    tokenized_example = chat_template.encode_messages(text_example, max_seq_len=max_seq_len)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("classification")
def process_classification_example(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    max_seq_len: int,
    text_keys: Union[str, list[str]] = "text",
    label_key: str = "label",
    **kwargs,
) -> list[dict[str, "torch.Tensor"]]:
    """
    Convert a single raw example into one classification training sample.

    Args:
        example:
            A single record from the dataset. Expected format (minimal):
                {
                    "<text_key>":  str,   # e.g. news article / sentence
                    "<label_key>": int,   # e.g. 0..(num_labels-1)
                    ...                   # other fields are ignored
                }
            By default:
                text_key  = "text"
                label_key = "label"

        tokenizer:
            A HuggingFace tokenizer used to tokenize the input text.

        max_seq_len:
            Maximum sequence length (in tokens). Text longer than this
            will be truncated to the first `max_seq_len` tokens.

        text_keys:
            Keys in `example` that contains the raw input text. If a list, the first key found in `example` will be used.

        label_key:
            Key in `example` that contains the class id. The value should be int-like.

    Returns:
        A list with exactly one sample dict:
            {
                "input_ids":      LongTensor[L],
                "attention_mask": LongTensor[L],
                "labels":         LongTensor[L],
                "position_ids":   LongTensor[L]
            }
    """
    # 1) text
    if isinstance(text_keys, str):
        text = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                text = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    # 2) label
    if label_key not in example:
        raise ValueError(f"Missing label key '{label_key}' in example.")
    try:
        label_val = int(example[label_key])
    except Exception as e:
        raise ValueError(f"Label '{example[label_key]}' is not an int-like value.") from e

    # 3) tokenize
    tokens: list[int] = tokenizer.encode(text, add_special_tokens=True)

    # 4) build samples
    examples: list[dict[str, torch.Tensor]] = []

    def build_sample(seq: list[int]) -> dict[str, "torch.Tensor"]:
        L = len(seq)
        token_labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long)
        token_labels[L - 1] = label_val

        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.ones(len(seq), dtype=torch.long),
            "labels": token_labels,
        }
        sample["position_ids"] = torch.arange(len(seq), dtype=torch.long)
        return sample

    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    examples.append(build_sample(tokens))
    return examples


@DATA_TRANSFORM_REGISTRY.register("qwen2_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen2_5_vl")
def process_sample_qwen2_5_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos

    source = kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]

    if "conversations" in sample and sample["conversations"] is not None and len(sample["conversations"]) > 0:
        conversations = sample["conversations"]
    else:
        conversations = sample
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
    )["position_ids"]
    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("qwen3_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl_moe")
def process_sample_qwen3_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos_metadata

    source = kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]

    if "conversations" in sample and sample["conversations"] is not None and len(sample["conversations"]) > 0:
        conversations = sample["conversations"]
    else:
        conversations = sample

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
        video_inputs = processor.video_processor(
            videos=videos, video_metadata=metadata, return_tensors="pt", return_metadata=True
        )
        video_grid_thw = video_inputs["video_grid_thw"]
        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

        video_metadata = video_inputs.pop("video_metadata")

        tokenized_example = chat_template.encode_messages(
            conversations, token_num_inputs, video_metadata=video_metadata
        )

    if not tokenized_example:
        tokenized_example = chat_template.encode_messages(conversations)

    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }

    input_ids = tokenized_example["input_ids"]
    tokenized_example["position_ids"] = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]

    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("qwen2_5_omni")
@DATA_TRANSFORM_REGISTRY.register("qwen3_omni_moe")
def process_sample_qwen_omni(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    position_id_func: "Callable",
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.audio_utils import fetch_audios
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos

    QWEN_OMNI_SYSTEM_MESSAGE = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def get_omni_token_ids(processor: "ProcessorMixin") -> tuple[int, int, int]:
        tokenizer = getattr(processor, "tokenizer", processor)
        vocab = tokenizer.get_vocab()
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

    image_token_id, video_token_id, audio_token_id = get_omni_token_ids(processor)

    source = kwargs["source_name"] if "source_name" in kwargs else sample["source_name"]
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
    model_inputs = model_inputs.data
    input_features = model_inputs.pop("input_features", None)
    feature_attention_mask = model_inputs.pop("feature_attention_mask", None)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        valid_mask = audio_feature_lengths != 0
        input_features = input_features[valid_mask].permute(0, 2, 1)[feature_attention_mask[valid_mask].bool()]

        model_inputs["input_features"] = input_features
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None

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
    )["position_ids"]

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
