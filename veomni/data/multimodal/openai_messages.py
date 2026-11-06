# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Normalize image-and-text conversations without changing message supervision."""

import base64
import binascii
import json
import math
import numbers
import sys
from collections.abc import Mapping

import numpy as np


OPENAI_CHATML_VLM_JINJA = """\
{%- for message in messages %}
    {%- set rendered = namespace(content='') %}
    {%- if message['content'] is string %}
        {%- set rendered.content = message['content'] %}
    {%- else %}
        {%- for block in message['content'] %}
            {%- if block['type'] == 'text' %}
                {%- set rendered.content = rendered.content + block['text'] %}
            {%- elif block['type'] == 'image' %}
                {%- set rendered.content = rendered.content + '<|vision_start|><|image_pad|><|vision_end|>' %}
            {%- else %}
                {{- raise_exception('Unsupported multimodal ChatML content type') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<|im_start|>' + message['role'] + '\\n' + rendered.content | trim + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
"""


def validate_openai_chatml_model_config(model_config):
    """Check the actual model config, which can differ from tokenizer assets."""
    text_config = (
        model_config.get("text_config")
        if isinstance(model_config, Mapping)
        else getattr(model_config, "text_config", None)
    ) or model_config
    if isinstance(text_config, Mapping):
        weight = text_config.get("mtp_loss_weight")
    else:
        weight = getattr(text_config, "mtp_loss_weight", None)
    if weight is not None and weight != 0:
        raise ValueError("OpenAI ChatML VLM does not yet support MTP labels")


def _sequence(value, field):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    return list(value)


def _image_value(value):
    if isinstance(value, bytes) and value:
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("image must be nonempty bytes, path, URL, or image data URI")
    if not value.startswith("data:"):
        return value
    header, sep, payload = value.partition(",")
    if not sep or header not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise ValueError("image data URI must be base64 PNG or JPEG")
    try:
        result = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid image data URI") from exc
    if not result:
        raise ValueError("empty image data URI")
    return result


def normalize_openai_messages(sample, *, tools_in_system=False):
    """Return ordered messages and image values from an OpenAI-style sample.

    Text-only content can be a string or text blocks. Images are user input and
    may be inline or supplied as a top-level ``images`` array matched to null
    image blocks. Mixed inline/top-level sources and partial matching are errors.
    """
    if not isinstance(sample, Mapping):
        raise ValueError("sample must be a mapping")
    tools = sample.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if tools is not None:
        tools = _sequence(tools, "tools")
    if tools and not tools_in_system:
        raise ValueError("nonempty tools require tools_in_system")
    for field in (
        "image",
        "video",
        "videos",
        "audio",
        "audios",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_mask",
        "video_mask",
    ):
        value = sample.get(field)
        if value is not None and (not hasattr(value, "__len__") or len(value)):
            raise ValueError(f"unsupported media field: {field}")
    messages = sample.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    messages = _sequence(messages, "messages")
    if not messages:
        raise ValueError("messages must not be empty")
    supplied = sample.get("images")
    external = _sequence(supplied, "images") if supplied is not None else []
    external_iter = iter(external)
    images, normalized = [], []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping")
        if message.get("tool_calls") or message.get("function_call"):
            raise ValueError("structured tool calls must be serialized in content")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unsupported message role")
        mask = message.get("loss_mask", int(role == "assistant"))
        if not isinstance(mask, numbers.Real) or mask not in (0, 1):
            raise ValueError("loss_mask must be binary")
        weight = message.get("loss_weight")
        if weight is not None and (
            not isinstance(weight, numbers.Real)
            or not math.isfinite(float(weight))
            or not 0 <= weight <= 1
            or (0 < weight < np.finfo(np.float32).tiny)
        ):
            raise ValueError("loss_weight must be finite and in [0, 1]")
        content = message.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else _sequence(content, "content")
        output = []
        adjacent_text = ""
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("content block must be a mapping")
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str) or any(
                    block.get(key) is not None for key in ("image", "url")
                ):
                    raise ValueError("text block must contain only text")
                adjacent_text += block["text"]
                if any(
                    token in adjacent_text
                    for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
                ):
                    raise ValueError("text cannot contain reserved vision tokens")
                output.append(("text", block["text"]))
            elif kind == "image":
                adjacent_text = ""
                if role != "user" or mask:
                    raise ValueError("image inputs require an unsupervised user message")
                if block.get("text") not in (None, ""):
                    raise ValueError("image block cannot contain text")
                inline = [block[key] for key in ("image", "url") if block.get(key) is not None]
                if len(inline) > 1:
                    raise ValueError("image block contains multiple image values")
                value = inline[0] if inline else None
                if external:
                    if value is not None:
                        raise ValueError("cannot mix inline and top-level images")
                    try:
                        value = next(external_iter)
                    except StopIteration as exc:
                        raise ValueError("too few top-level images") from exc
                images.append(_image_value(value))
                output.append(("image", None))
            else:
                raise ValueError(f"unsupported content type: {kind!r}")
        converted = {"role": role, "content": output, "loss_mask": int(mask)}
        if weight is not None and weight != 1:
            converted["loss_weight"] = float(weight)
        normalized.append(converted)
    sentinel = object()
    if next(external_iter, sentinel) is not sentinel:
        raise ValueError("too many top-level images")
    if tools:
        system_text = "".join(
            value
            for message in normalized
            if message["role"] == "system"
            for kind, value in message["content"]
            if kind == "text"
        )
        for tool in tools:
            function = tool.get("function", tool) if isinstance(tool, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if not isinstance(name, str) or not name or name not in system_text:
                raise ValueError("each tool name must already appear in the system message")
    return normalized, images


class OpenAIChatMLVisionTemplate:
    """Render multimodal messages with the same labels as the text ChatML path."""

    def __init__(self, template, max_seq_len):
        from ..chat_template import ChatmlTemplate

        self.template = template
        self.text_template = ChatmlTemplate(template.tokenizer)
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len

    def encode_messages(self, messages, num_tokens=None, **kwargs):
        import torch

        from ...utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX

        counts = iter((num_tokens or {}).get("image", []))
        rendered = []
        has_image = False
        for message in messages:
            content = []
            for kind, value in message["content"]:
                if kind == "text":
                    content.append(value)
                else:
                    has_image = True
                    try:
                        count = next(counts)
                    except StopIteration as exc:
                        raise ValueError("image token count is missing") from exc
                    content.append(self.template.image_pattern(int(count)))
            weight = message.get("loss_weight", 1)
            rendered_message = {
                "role": message["role"],
                "content": "".join(content),
                "loss_mask": int(bool(message["loss_mask"]) and weight != 0),
            }
            if rendered_message["loss_mask"] and 0 < weight < 1:
                rendered_message["loss_weight"] = weight
            rendered.append(rendered_message)
        if next(counts, None) is not None:
            raise ValueError("unused image token count")
        encoded = self.text_template.encode_messages(
            rendered, max_seq_len=sys.maxsize if has_image else self.max_seq_len
        )
        if any(message["loss_mask"] and 0 < message.get("loss_weight", 1) < 1 for message in rendered) and (
            "loss_weights" not in encoded
        ):
            raise ValueError("non-default loss_weight requires a weighted ChatML renderer")
        if has_image and len(encoded["input_ids"]) > self.max_seq_len:
            raise ValueError("image conversation exceeds max_seq_len; refusing to truncate image tokens")
        encoded = {
            key: torch.tensor(value, dtype=torch.float32 if key == "loss_weights" else torch.long)
            for key, value in encoded.items()
        }
        ids = encoded["input_ids"]
        ids[ids == self.template.image_token_id] = IMAGE_INPUT_INDEX
        ids[ids == self.template.video_token_id] = VIDEO_INPUT_INDEX
        return encoded
