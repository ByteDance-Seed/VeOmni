"""SeedOmni dataset preprocessors.

Maps each multisource ``names`` entry to a conversation layout understood by
``veomni.data.seed_omni.seedomni_transform``.
"""

from __future__ import annotations

import random

from ...models.seed_omni.utils.conversation import _IMG_TAG_KEY
from ...utils.registry import Registry


SEED_OMNI_PREPROCESSOR_REGISTRY = Registry("SeedOmniPreprocessor")


def conv_preprocess(source: str, conversations, example, **kwargs) -> tuple[list, dict[str, list]]:
    """Dispatch ``source`` to its registered preprocessor.

    Returns ``(constructed, media_refs)``:

    * ``constructed`` — the ``[[role, (type, value) | (type, value, meta), ...],
      ...]`` layout.
    * ``media_refs`` — per-sample media refs **keyed by the item type they pair
      with**: ``{"image": [...], "video": [...], "audio": [...]}``. An absent key
      means the sample has none of that modality, so a text-only preprocessor
      returns ``{}``.

    Keyed rather than positional because this registry is an extension point —
    a preprocessor for a private corpus lives outside this repo — and the
    modality list is still growing (action, 3d, camera). A tuple makes every
    new modality a change to every implementation and to every unpack site; a
    dict makes it one new key, read by whoever knows that key.

    Using the item ``type`` as the key is what keeps the pairing readable:
    ``media_refs["image"]`` is consumed by the ``("image", None)`` turns, in
    order. Lengths must match per modality — pairing is positional and the
    preprocessor owns that alignment (the multi-turn edit preprocessor
    duplicates copy-image refs to express reuse).
    """
    result = SEED_OMNI_PREPROCESSOR_REGISTRY[source](conversations, example, **kwargs)
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)):
        # Named rather than left to unpack sideways: the old contract was a
        # 3- or 4-tuple of positional ref lists, and an out-of-repo
        # preprocessor still on it would otherwise fail with a bare "too many
        # values to unpack" naming neither the source nor the fix.
        raise ValueError(
            f"conv_preprocess: preprocessor for source {source!r} returned {type(result).__name__} "
            f"{f'of length {len(result)}' if isinstance(result, tuple) else ''}; expected "
            f"(constructed, media_refs) where media_refs is a dict keyed by item type. Migrate a legacy "
            f"(constructed, image_refs, video_refs[, audio_refs]) return to "
            f'(constructed, {{"image": image_refs, "video": video_refs, "audio": audio_refs}}) and drop the '
            f"keys the source has no media for."
        )
    return result


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("imagenet1k")
def imagenet1k_preprocess(conversations, example, **kwargs):
    del kwargs
    class_labels = [item.strip() for item in conversations.split(",")]
    class_label = random.choice(class_labels)
    constructed = [
        ["user", ("text", class_label)],
        ["assistant", ("image", None, {_IMG_TAG_KEY: "gen"})],
    ]
    return constructed, {"image": list(example.get("images", []) or [])}


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("tulu-3-sft-mixture")
def tulu_3_sft_mixture_preprocess(conversations, example, **kwargs):
    del kwargs, example
    text_example = conversations["messages"]

    system_messages = [message for message in text_example if message["role"] == "system"]
    if system_messages:
        if len(system_messages) != 1 or text_example[0]["role"] != "system":
            raise ValueError("Tulu conversations must contain at most one system message, at the beginning.")

        first_user_index = next(
            (index for index, message in enumerate(text_example) if message["role"] == "user"),
            None,
        )
        if first_user_index is None:
            raise ValueError("Tulu conversations with a system message must contain a user message.")

        first_user = text_example[first_user_index]
        first_user_content = f"<system>\n{system_messages[0]['content']}\n</system>\n{first_user['content']}"
        text_example = [
            {**message, "content": first_user_content} if index == first_user_index else message
            for index, message in enumerate(text_example)
            if message["role"] != "system"
        ]

    constructed_conversation = []
    for conversation in text_example:
        constructed_conversation.append([conversation["role"], ("text", conversation["content"])])
    # Text-only, so no media keys at all.
    return constructed_conversation, {}


@SEED_OMNI_PREPROCESSOR_REGISTRY.register("voice_assistant")
def voice_assistant_preprocess(conversations, example, **kwargs):
    """Spoken question -> text answer: one ``user`` audio turn per sample.

    Exercises the thinker's audio tower, which no other shipped source does.
    The clip is the *user* side, so it is an encoder input rather than a talker
    target — that distinction is carried by ``role`` alone, since both uses are
    ``type="audio"``.

    The ``human`` turn's text is the clip's transcript. It is dropped rather
    than emitted beside the audio: fed as text the model would answer from the
    transcript and the tower would carry no signal, which is the opposite of
    what this source is here to test.

    Clips are 22.05 kHz, matching neither the tower (16 kHz) nor the codec
    (24 kHz), so this is also the source that keeps the declared-rate path
    honest — the transform reads the rate off the file and each module
    resamples for itself.
    """
    del kwargs
    audio_refs = list(example.get("audios", []) or [])

    constructed = []
    spoken = 0
    for message in conversations:
        if message["from"] == "human":
            constructed.append(["user", ("audio", None)])
            spoken += 1
        elif message["from"] == "gpt":
            constructed.append(["assistant", ("text", message["value"])])
        else:
            raise ValueError(f"voice_assistant: unexpected speaker {message['from']!r}")

    # Counted up front and in both directions. Pairing downstream is positional,
    # so a count that does not match does not fail — it attaches the wrong clip
    # to the wrong turn, and every later turn shifts with it.
    if spoken != len(audio_refs):
        raise ValueError(
            f"voice_assistant: sample has {spoken} spoken turn(s) but {len(audio_refs)} clip(s); "
            f"they are paired by position, so an unequal count would misattach them."
        )
    return constructed, {"audio": audio_refs}


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
    return _sharegpt4v_sft_layout(conversations), {"image": list(example.get("images", []) or [])}


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
    return constructed_conversation, {"video": list(example.get("videos", []) or [])}


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

    Returns ``(constructed, {"image": image_refs})``. ``image_refs`` is a NEW
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
    return constructed_conversation, {"image": image_refs}


__all__ = ["SEED_OMNI_PREPROCESSOR_REGISTRY", "conv_preprocess"]
