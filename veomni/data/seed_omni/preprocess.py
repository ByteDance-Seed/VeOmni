"""SeedOmni V2 dataset preprocessors.

Maps each multisource ``names`` entry to a conversation layout understood by
``veomni.data.seed_omni.seedomni_transform``.
"""

from __future__ import annotations

import random

from ...models.seed_omni.utils.conversation import _IMG_TAG_KEY
from ...utils.registry import Registry


SEED_OMNI_PREPROCESSOR_REGISTRY = Registry("SeedOmniPreprocessor")


def conv_preprocess(source: str, conversations, example, **kwargs):
    """Dispatch ``source`` to its registered preprocessor.

    Returns a 3-tuple ``(constructed, image_refs, video_refs)``: the
    ``[[role, (type, value) | (type, value, meta), ...], ...]`` layout plus the
    per-sample media ref lists the transform decodes via ``fetch_images`` /
    ``fetch_videos``. The ref lists' lengths must match the flattened
    image / video entry counts so ``_build_conversation_list`` can pair them by
    pure sequential order — the preprocessor owns that alignment (the
    multi-turn edit preprocessor duplicates copy-image refs to express reuse).
    """
    return SEED_OMNI_PREPROCESSOR_REGISTRY[source](conversations, example, **kwargs)


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("imagenet1k")
def imagenet1k_preprocess(conversations, example, **kwargs):
    del kwargs
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed = [
        ["user", ("text", class_label)],
        ["assistant", ("image", None, {_IMG_TAG_KEY: "gen"})],
    ]
    return constructed, list(example.get("images", []) or []), []


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("tulu-3-sft-mixture")
def tulu_3_sft_mixture_preprocess(conversations, example, **kwargs):
    del kwargs, example
    text_example = conversations["messages"]
    constructed_conversation = []
    for conversation in text_example:
        constructed_conversation.append([conversation["role"], ("text", conversation["content"])])
    return constructed_conversation, [], []


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
            constructed_conversation.append([role, ("image", None, {_IMG_TAG_KEY: "und"}), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("sharegpt4v_cap_100k")
@SEED_OMNI_PREPROCESSOR_REGISTRY.register("sharegpt4v_sft")
def sharegpt4v_cap_preprocess(conversations, example, **kwargs):
    del kwargs
    return _sharegpt4v_sft_layout(conversations), list(example.get("images", []) or []), []


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("llava_video")
def llava_video_preprocess(conversations, example, **kwargs):
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
            constructed_conversation.append([role, ("video", None, {_IMG_TAG_KEY: "und"}), ("text", value)])
        else:
            constructed_conversation.append([role, ("text", value)])
    return constructed_conversation, [], list(example.get("videos", []) or [])


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("seed_edit_p23_multi_turn")
def seed_edit_p23_multi_turn_preprocess(conversations, example, **kwargs):
    """Preprocess the ``seed_edit_p23_multi_turn`` multi-turn image-edit chain.

    Input is ShareGPT-style: ``conversations=[{"from": "human"|"gpt",
    "value": "<image>..."}]`` with a per-sample ``images`` list whose bytes are
    ordered by ``<image>`` appearance. Every sample is one human source image
    followed by one or more gpt target images — the ``<image>`` ``from``-sequence
    is always ``(human, gpt, gpt, ..., gpt)`` with 2–7 images per sample, exactly
    one ``<image>`` per message value, and ``len(images)`` == ``<image>`` token
    count (verified across all 21377 rows).

    The chain ``[human:<image>instr1, gpt:<image>, human:instr2, gpt:<image>,
    ...]`` flattens to ``[source0, instr1, target1, source1, instr2, target2,
    ...]`` where ``source0`` is the human source image and each ``source_{i+1}``
    is a copy of ``target_i`` (the next edit operates on the previous result).
    Source and target are emitted as separate image items so the model can route
    them by ``_img_tag`` — source = clean VAE context + SigLIP, target = noised
    VAE target:

    - human ``<image>`` → ``role=user``, ``_img_tag="edit"`` (a source).
    - gpt ``<image>`` → ``role=assistant``, ``_img_tag="gen"`` (a target); if it
      is not the last ``<image>`` overall, a same-image copy with
      ``role=assistant``, ``_img_tag="edit"`` follows it as the next source.

    Instruction text is emitted as a 2-tuple (no ``_img_tag``) with the turn's
    role.

    Returns ``(constructed, image_refs, video_refs)``. ``image_refs`` is a NEW
    list (``example["images"]`` is never mutated) whose length equals the
    flattened image entry count: each primary image appends
    ``example["images"][img_seq]`` (``img_seq`` counts primary ``<image>`` tokens
    only — copies append the same ref again without advancing it). This lets
    ``_build_conversation_list`` pair images by pure sequential order; repeated
    refs still decode once because ``fetch_images`` caches duplicate refs and
    returns independent tensor clones on reuse.
    """
    del kwargs
    role_mapping = {"human": "user", "gpt": "assistant"}
    total = sum(message["value"].count("<image>") for message in conversations)
    sample_images = example.get("images", []) or []

    constructed_conversation = []
    image_refs: list = []
    img_seq = 0
    for message in conversations:
        role = role_mapping[message["from"]]
        value = message["value"]
        n_img = value.count("<image>")
        if n_img > 0:
            for _ in range(n_img):
                if role == "user":
                    constructed_conversation.append([role, ("image", None, {_IMG_TAG_KEY: "edit"})])
                    image_refs.append(sample_images[img_seq])
                else:  # assistant target
                    constructed_conversation.append([role, ("image", None, {_IMG_TAG_KEY: "gen"})])
                    image_refs.append(sample_images[img_seq])
                    if img_seq < total - 1:
                        # Next turn's source is a copy of this target's image —
                        # append the same ref again so the expanded ref list
                        # matches the duplicated layout (decoded twice
                        # downstream). The copy does not advance ``img_seq``.
                        constructed_conversation.append([role, ("image", None, {_IMG_TAG_KEY: "edit"})])
                        image_refs.append(sample_images[img_seq])
                img_seq += 1
            stripped = value.replace("<image>", "").strip()
            if stripped:
                constructed_conversation.append([role, ("text", stripped)])
        else:
            text = value.strip()
            if text:
                constructed_conversation.append([role, ("text", text)])
    return constructed_conversation, image_refs, []


__all__ = ["SEED_OMNI_PREPROCESSOR_REGISTRY", "conv_preprocess"]
