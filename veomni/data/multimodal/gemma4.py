# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Gemma4 native-processor inputs and assistant-only SFT labels."""

import torch

from ...utils.constants import IGNORE_INDEX


def _messages(sample, kwargs):
    from .preprocess import conv_preprocess

    messages = sample.get("messages")
    if messages is None:
        conversations = sample.get("conversations", sample)
        if isinstance(conversations, list) and conversations and "role" in conversations[0]:
            messages = conversations
        else:
            source = kwargs.get("source_name") or sample.get("source") or sample.get("source_name")
            conversations = conv_preprocess(source, conversations, **kwargs)
            messages = [
                {"role": conversation[0], "content": [{"type": kind, kind: value} for kind, value in conversation[1:]]}
                for conversation in conversations
            ]
    if not messages:
        raise ValueError("Gemma4 requires a nonempty conversation.")
    return messages


def _media(sample, messages):
    inputs = {}
    for kind, key in (("image", "images"), ("video", "videos"), ("audio", "audios")):
        embedded = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            for item in content:
                if item["type"] == kind:
                    if message["role"] == "assistant":
                        raise ValueError("Gemma4 SFT supports text-only assistant responses.")
                    embedded.append(item.get(kind))
        values = sample.get(key)
        if values is None:
            values = embedded
        if len(values) != len(embedded) or any(value is None for value in values):
            raise ValueError(f"Gemma4 {kind} placeholders must match the supplied {key} in conversation order.")
        inputs[kind] = values
    return inputs


def _assistant_labels(processor, messages, prompt, offsets, replacements, input_ids):
    """Map native template spans through HF's actual placeholder expansions.

    Only rendering is repeated per turn. Images, video and audio are processed
    once; replacement offsets account for all variable-length modal tokens.
    Prefix validation rejects templates that rewrite earlier turns instead of
    silently assigning labels to a different conversation.
    """

    def expanded_offset(position):
        shift = 0
        for replacement in replacements:
            start, end = replacement["span"]
            if start < position < end:
                raise ValueError("Gemma4 assistant boundary intersects a multimodal placeholder.")
            if end <= position:
                shift += len(replacement["replacement"]) - (end - start)
        return position + shift

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or message.get("loss_mask", 1) == 0:
            continue
        before = (
            processor.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
            if index
            else ""
        )
        after = processor.apply_chat_template(messages[: index + 1], tokenize=False, add_generation_prompt=False)
        if not prompt.startswith(before) or not prompt.startswith(after) or len(after) < len(before):
            raise ValueError("Gemma4 assistant-only labels require prefix-stable native chat rendering.")
        start, end = expanded_offset(len(before)), expanded_offset(len(after))
        selected = (offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])
        labels[selected] = input_ids[selected]
    if not labels.ne(IGNORE_INDEX).any():
        raise ValueError("Gemma4 SFT requires at least one supervised assistant response.")
    return labels


def process_gemma4_sample(sample, processor, position_id_func, max_seq_len=8192, **kwargs):
    """Return one unpadded feature dict using HF's Gemma4 media/token contract."""
    from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

    from .audio_utils import load_audio_from_bytes
    from .image_utils import load_image, smart_resize
    from .video_utils import fetch_videos_metadata

    if not isinstance(processor, Gemma4Processor):
        raise ValueError("Gemma4 VLM training requires a native Gemma4Processor.")
    messages = _messages(sample, kwargs)
    media = _media(sample, messages)
    images = [load_image(value) if isinstance(value, (str, bytes)) else value for value in media["image"]]
    images = [
        smart_resize(value, **kwargs) if hasattr(value, "resize") and not hasattr(value, "shape") else value
        for value in images
    ]
    audios = [load_audio_from_bytes(value) if isinstance(value, bytes) else value for value in media["audio"]]
    videos, metadata = None, None
    if media["video"]:
        video_kwargs = dict(kwargs)
        video_kwargs.setdefault("min_frames", processor.video_processor.num_frames)
        video_kwargs.setdefault("max_frames", processor.video_processor.num_frames)
        video_kwargs["use_audio_in_video"] = False
        videos, metadata, _, _ = fetch_videos_metadata(media["video"], **video_kwargs)

    options = dict(kwargs.get("processor_kwargs") or {})
    text_options = dict(options.pop("text_kwargs", {}))
    if options.get("truncation") or text_options.get("truncation"):
        raise ValueError("Gemma4 does not truncate multimodal inputs; use max_seq_len and media budgets.")
    text_options.update(
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_text_replacement_offsets=True,
        return_mm_token_type_ids=True,
    )
    options.update(text_kwargs=text_options, return_tensors="pt")
    if videos is not None:
        video_options = dict(options.pop("videos_kwargs", {}))
        video_options.update(video_metadata=metadata, do_sample_frames=False)
        options["videos_kwargs"] = video_options
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    feature = dict(processor(text=prompt, images=images or None, videos=videos, audio=audios or None, **options))
    offsets = feature.pop("offset_mapping")[0]
    replacements = feature.pop("text_replacement_offsets")[0]
    feature.pop("video_metadata", None)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        feature[key] = feature[key].squeeze(0)
    input_ids = feature["input_ids"]
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"Gemma4 sample length {len(input_ids)} exceeds max_seq_len={max_seq_len}; reduce media budgets."
        )
    feature["labels"] = _assistant_labels(processor, messages, prompt, offsets, replacements, input_ids)
    feature["position_ids"] = position_id_func(input_ids=input_ids.unsqueeze(0))["position_ids"].squeeze(0)
    # Frames are independent vision-tower rows. Keep a singleton frame axis so
    # generic packing can concatenate clips with different sampled frame counts.
    for key in ("pixel_values_videos", "video_position_ids"):
        if key in feature:
            value = feature[key]
            feature[key] = value.flatten(0, 1).unsqueeze(1)
    return [feature]
