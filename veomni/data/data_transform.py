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
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
from veomni.utils.registry import Registry


_is_transformers_v5 = is_transformers_version_greater_or_equal_to("5.0.0")


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


@DATA_TRANSFORM_REGISTRY.register("dpo")
def process_dpo_example(
    example: Dict[str, Any],
    chat_template: "ChatTemplate" = None,
    tokenizer: "PreTrainedTokenizer" = None,
    max_seq_len: int = 2048,
    **kwargs,
) -> List[Dict[str, "torch.Tensor"]]:
    """Process a DPO preference pair into a single flat sample.

    Chosen and rejected sequences are concatenated into one 1-D tensor with
    ``position_ids`` that reset at the boundary so that flash-attention treats
    them as two independent sequences.  This format is directly compatible with
    ``MainCollator`` (packing + SP) — no DPO-specific collator is needed.

    Supported input formats:
      1. Conversation: {"chosen": [messages...], "rejected": [messages...]}
      2. Plaintext with prompt: {"prompt": str, "chosen": str, "rejected": str}

    Returns:
        A list with one dict.  Each value is a 1-D tensor of length
        ``len_chosen + len_rejected``.  Keys: ``input_ids``, ``attention_mask``,
        ``labels``, ``position_ids``.
    """
    chosen_raw = example["chosen"]
    rejected_raw = example["rejected"]

    if isinstance(chosen_raw, list):
        assert chat_template is not None, "chat_template is required for conversation-format DPO data"
        chosen_tok = chat_template.encode_messages(chosen_raw, max_seq_len=max_seq_len)
        rejected_tok = chat_template.encode_messages(rejected_raw, max_seq_len=max_seq_len)
    else:
        assert tokenizer is not None, "tokenizer is required for plaintext-format DPO data"
        prompt = example.get("prompt", "")
        chosen_text = prompt + chosen_raw
        rejected_text = prompt + rejected_raw

        chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=True)[:max_seq_len]
        rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=True)[:max_seq_len]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True) if prompt else []
        prompt_len = len(prompt_ids)

        chosen_tok = {
            "input_ids": chosen_ids,
            "attention_mask": [1] * len(chosen_ids),
            "labels": [IGNORE_INDEX] * prompt_len + chosen_ids[prompt_len:],
        }
        rejected_tok = {
            "input_ids": rejected_ids,
            "attention_mask": [1] * len(rejected_ids),
            "labels": [IGNORE_INDEX] * prompt_len + rejected_ids[prompt_len:],
        }

    def _to_tensor(v):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v)

    c_ids = _to_tensor(chosen_tok["input_ids"])
    r_ids = _to_tensor(rejected_tok["input_ids"])
    c_len = c_ids.shape[-1]
    r_len = r_ids.shape[-1]

    result = {
        "input_ids": torch.cat([c_ids, r_ids]),
        "attention_mask": torch.cat(
            [_to_tensor(chosen_tok["attention_mask"]), _to_tensor(rejected_tok["attention_mask"])]
        ),
        "labels": torch.cat([_to_tensor(chosen_tok["labels"]), _to_tensor(rejected_tok["labels"])]),
        "position_ids": torch.cat([torch.arange(c_len, dtype=torch.int64), torch.arange(r_len, dtype=torch.int64)]),
    }
    return [result]


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


def _process_sample_qwen_vl_base(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    use_mm_token_type_ids: bool = False,
    **kwargs,
):
    from .multimodal import conv_preprocess
    from .multimodal.image_utils import fetch_images
    from .multimodal.video_utils import fetch_videos_metadata

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")

    if "conversations" in sample and sample["conversations"] is not None and len(sample["conversations"]) > 0:
        conversations = sample["conversations"]
    else:
        conversations = sample
    conversations = conv_preprocess(source, conversations, **kwargs)

    token_num_inputs, image_inputs, video_inputs = {}, {}, {}
    image_grid_thw, video_grid_thw = None, None
    video_metadata = None

    if "images" in sample and sample["images"]:
        images = fetch_images(sample["images"], **kwargs)
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num

    if "videos" in sample and sample["videos"]:
        videos, metadata, _, _ = fetch_videos_metadata(sample["videos"], **kwargs)
        video_inputs = processor.video_processor(
            videos=videos, video_metadata=metadata, return_tensors="pt", return_metadata=True
        )
        video_grid_thw = video_inputs["video_grid_thw"]
        video_metadata = video_inputs.pop("video_metadata", None)

        merge_length = processor.video_processor.merge_size**2
        video_token_num = video_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["video"] = video_token_num

    # Encoding
    encode_kwargs = {}
    if video_metadata is not None:
        encode_kwargs["video_metadata"] = video_metadata

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs, **encode_kwargs)

    tokenized_example = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in tokenized_example.items()
    }

    input_ids = tokenized_example["input_ids"]
    attention_mask = tokenized_example["attention_mask"]

    # Masks and Token Types
    tokenized_example["image_mask"] = input_ids == IMAGE_INPUT_INDEX
    tokenized_example["video_mask"] = input_ids == VIDEO_INPUT_INDEX

    # Position IDs
    position_id_func_kwargs = {
        "input_ids": input_ids.unsqueeze(0),
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
        "attention_mask": attention_mask.unsqueeze(0),
    }

    if use_mm_token_type_ids:
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[tokenized_example["image_mask"]] = 1
        mm_token_type_ids[tokenized_example["video_mask"]] = 2
        tokenized_example["mm_token_type_ids"] = mm_token_type_ids
        position_id_func_kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)

    tokenized_example["position_ids"] = position_id_func(**position_id_func_kwargs)["position_ids"]
    tokenized_example["position_ids"] = tokenized_example["position_ids"].squeeze().clone()

    # Final cleanup
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
    tokenized_example.update(image_inputs)
    tokenized_example.update(video_inputs)

    return [tokenized_example]


@DATA_TRANSFORM_REGISTRY.register("qwen2_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen2_5_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl")
@DATA_TRANSFORM_REGISTRY.register("qwen3_vl_moe")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5")
@DATA_TRANSFORM_REGISTRY.register("qwen3_5_moe")
def process_sample_qwen_vl(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Unified processing function for Qwen-VL series models.
    Automatically determines whether to use mm_token_type_ids based on transformers version.
    """
    return _process_sample_qwen_vl_base(
        sample,
        processor,
        chat_template,
        position_id_func,
        use_mm_token_type_ids=_is_transformers_v5,
        **kwargs,
    )


def compute_qwen3_5_vision_metadata(
    grid_thw_list: List[List[int]],
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> Dict[str, "torch.Tensor"]:
    """Precompute, on the host, the per-vision-token metadata that Qwen3.5-VL's vision tower
    otherwise re-derives from the (GPU) ``grid_thw`` tensor on every forward.

    ``Qwen3_5VisionModel.fast_pos_embed_interpolate`` / ``rot_pos_emb`` / ``forward`` build the
    bilinear position-embedding lookup, the rotary position ids and ``cu_seqlens`` by iterating /
    ``.tolist()``-ing ``grid_thw`` — each a host↔device sync per ViT forward. Since these are a pure
    function of ``grid_thw`` (a tiny ``(num_images, 3)`` tensor the data collator already has on the
    host), we compute them here and feed them in; the model just indexes the result.

    Layout matches the patch-token order in ``pixel_values``: per image ``(t, h, w)``, the
    ``h * w`` tokens of each frame are laid out in spatial-merge-block order
    ``(block_row, block_col, intra_row, intra_col)`` and the ``t`` frames are concatenated, then all
    images are concatenated.

    Returns a dict with:
      - ``pos_embed_indices`` ``(N, 4)`` int64 — the 4 grid-embedding indices per token (bilinear corners)
      - ``pos_embed_weights`` ``(N, 4)`` float32 — the 4 bilinear weights per token
      - ``rot_pos_ids`` ``(N, 2)`` int64 — the (row, col) coords per token (indexes the rotary freq table)
      - ``cu_seqlens`` ``(num_frames + 1,)`` int32 — one varlen segment of length ``h * w`` per frame
      - ``max_hw`` int — ``max(max(h, w))`` over images (size of the rotary freq table needed)
    where ``N == sum(t * h * w)``.
    """
    m = spatial_merge_size
    g = num_grid_per_side

    pos_idx_chunks: List["torch.Tensor"] = []
    pos_w_chunks: List["torch.Tensor"] = []
    rot_chunks: List["torch.Tensor"] = []
    cu_seqlens: List[int] = [0]
    max_hw = 0

    for t, h, w in grid_thw_list:
        mh, mw = h // m, w // m

        # Bilinear interpolation onto the (g x g) position-embedding grid. float64 matches the
        # `torch.linspace(..., dtype=torch.float64)` the model uses; `linspace` is the same simple
        # deterministic recurrence on CPU and GPU, so the interpolated coords (and hence the
        # truncated `floor` corners below) are identical to the on-the-fly path.
        h_idxs = torch.linspace(0, g - 1, h, dtype=torch.float64)
        w_idxs = torch.linspace(0, g - 1, w, dtype=torch.float64)
        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=g - 1)
        w_ceil = torch.clamp(w_floor + 1, max=g - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        dh_g, dw_g = torch.meshgrid(dh, dw, indexing="ij")  # (h, w)
        hf_g, wf_g = torch.meshgrid(h_floor, w_floor, indexing="ij")
        hc_g, wc_g = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
        w11 = dh_g * dw_g
        w10 = dh_g - w11
        w01 = dw_g - w11
        w00 = 1 - dh_g - w01
        # corners: (h_floor, w_floor), (h_floor, w_ceil), (h_ceil, w_floor), (h_ceil, w_ceil)
        h_grid = torch.stack([hf_g, hf_g, hc_g, hc_g])  # (4, h, w)
        w_grid = torch.stack([wf_g, wc_g, wf_g, wc_g])  # (4, h, w)
        idx_hw = h_grid * g + w_grid  # (4, h, w) long
        wt_hw = torch.stack([w00, w01, w10, w11])  # (4, h, w) float64

        # Reorder the (h, w) axes -> spatial-merge-block flatten (block_row, block_col, intra_row, intra_col),
        # i.e. reshape (h, w) -> (mh, m, mw, m) [=(br, ir, bc, ic)] then permute to (br, bc, ir, ic).
        idx_tok = idx_hw.reshape(4, mh, m, mw, m).permute(0, 1, 3, 2, 4).reshape(4, h * w).transpose(0, 1).contiguous()
        wt_tok = (
            wt_hw.reshape(4, mh, m, mw, m)
            .permute(0, 1, 3, 2, 4)
            .reshape(4, h * w)
            .transpose(0, 1)
            .to(torch.float32)
            .contiguous()
        )  # (h*w, 4)
        r_grid, c_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # (h, w), values r / c
        rot_tok = torch.stack(
            [
                r_grid.reshape(mh, m, mw, m).permute(0, 2, 1, 3).reshape(h * w),
                c_grid.reshape(mh, m, mw, m).permute(0, 2, 1, 3).reshape(h * w),
            ],
            dim=-1,
        ).contiguous()  # (h*w, 2) long

        if t > 1:  # the t frames share the same per-position metadata
            idx_tok = idx_tok.repeat(t, 1)
            wt_tok = wt_tok.repeat(t, 1)
            rot_tok = rot_tok.repeat(t, 1)

        pos_idx_chunks.append(idx_tok)
        pos_w_chunks.append(wt_tok)
        rot_chunks.append(rot_tok)
        for _ in range(t):
            cu_seqlens.append(cu_seqlens[-1] + h * w)
        max_hw = max(max_hw, h, w)

    return {
        "pos_embed_indices": torch.cat(pos_idx_chunks, dim=0),
        "pos_embed_weights": torch.cat(pos_w_chunks, dim=0),
        "rot_pos_ids": torch.cat(rot_chunks, dim=0),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "max_hw": max_hw,
    }


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

    source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
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
