"""SeedOmni V2 dataset preprocessors.

Maps each multisource ``names`` entry to a conversation layout understood by
``veomni.data.seed_omni.seedomni_transform``.
"""

from __future__ import annotations

import random

from ...utils.registry import Registry


SEED_OMNI_PREPROCESSOR_REGISTRY = Registry("SeedOmniPreprocessor")


def conv_preprocess(source: str, conversations, **kwargs):
    return SEED_OMNI_PREPROCESSOR_REGISTRY[source](conversations, **kwargs)


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("imagenet1k")
def imagenet1k_preprocess(conversations, **kwargs):
    del kwargs
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    return [
        ["user", ("text", class_label)],
        ["assistant", ("image", None)],
    ]


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("tulu-3-sft-mixture")
def tulu_3_sft_mixture_preprocess(conversations, **kwargs):
    del kwargs
    text_example = conversations["messages"]
    constructed_conversation = []
    for conversation in text_example:
        constructed_conversation.append([conversation["role"], ("text", conversation["content"])])
    return constructed_conversation


def _sharegpt4v_sft_layout(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value:
            value = value.replace("<image>", "")
            constructed_conversation.append([role, ("image", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("sharegpt4v_cap_100k")
@SEED_OMNI_PREPROCESSOR_REGISTRY.register("sharegpt4v_sft")
def sharegpt4v_cap_preprocess(conversations, **kwargs):
    del kwargs
    return _sharegpt4v_sft_layout(conversations)


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("llava_video")
def llava_video_preprocess(conversations, **kwargs):
    """LLaVA-Video-178K layout — like ShareGPT4V but the media turn is a video.

    The upstream marks the video position with a ``<image>`` (occasionally
    ``<video>``) token in the first human turn; we strip it and emit a
    ``("video", None)`` turn whose value is paired with the per-sample ``videos``
    list in source order by the transform.
    """
    del kwargs
    role_mapping = {"human": "user", "gpt": "assistant"}
    if conversations[0]["from"] != "human":
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    constructed_conversation = []
    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if "<image>" in value or "<video>" in value:
            value = value.replace("<image>", "").replace("<video>", "").strip()
            constructed_conversation.append([role, ("video", None), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


__all__ = ["SEED_OMNI_PREPROCESSOR_REGISTRY", "conv_preprocess"]
