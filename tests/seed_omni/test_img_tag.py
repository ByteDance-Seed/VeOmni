"""Per-image ``_img_tag`` (und/gen/edit) data-layer support — Plan 1.

Covers the model-agnostic data-layer work only:

* :func:`iter_desired_images` — an image-only selector that reuses
  :func:`iter_desired_items` with ``types=["image"]`` and adds an ``img_tag``
  filter over ``meta[_IMG_TAG_KEY]``.
* The optional ``(type, value, meta)`` 3-tuple meta channel in
  ``_build_conversation_list`` (2-tuple stays backward compatible) — pure
  sequential image/video pairing; the preprocessor owns the ref-list layout.
* Existing preprocessors tag their image entries (imagenet1k=gen,
  sharegpt4v=und, llava_video=und); text-only tulu stays untagged. Each
  preprocessor returns ``(constructed, image_refs, video_refs)`` whose media
  ref lengths match the flattened image/video entry counts.
* The ``seed_edit_p23_multi_turn`` preprocessor emits the source/target/copy
  edit chain with correct roles and tags, and returns an expanded
  ``image_refs`` list (copy-image refs duplicated) so
  ``_build_conversation_list`` pairs them by pure sequential order.
"""

from __future__ import annotations

import pytest
import torch

from veomni.data.seed_omni.preprocess import conv_preprocess
from veomni.data.seed_omni.seedomni_transform import _build_conversation_list
from veomni.models.seed_omni.utils.conversation import (
    _IMG_TAG_KEY,
    ConversationItem,
    iter_desired_images,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _img(tag: str | None, role: str = "user", source: str | None = None) -> ConversationItem:
    meta = {} if tag is None else {_IMG_TAG_KEY: tag}
    return ConversationItem(
        type="image",
        value=torch.zeros(3, 2, 2, dtype=torch.uint8),
        role=role,
        source=source,
        meta=meta,
    )


def _txt(role: str = "user") -> ConversationItem:
    return ConversationItem(type="text", value="hello", role=role)


def _fake_images(n: int) -> list[torch.Tensor]:
    # Distinct fill values so sequential pairing is checkable by content.
    return [torch.full((3, 2, 2), i, dtype=torch.uint8) for i in range(n)]


def _fake_tensors_for_refs(refs) -> list[torch.Tensor]:
    # One distinct uint8 tensor per unique ref (by identity), so duplicate refs
    # (edit copy-images) decode to the same content — mirroring ``fetch_images``
    # on a repeated ref without actually decoding.
    cache: dict[int, torch.Tensor] = {}
    out: list[torch.Tensor] = []
    for ref in refs:
        key = id(ref)
        if key not in cache:
            cache[key] = torch.full((3, 2, 2), len(cache), dtype=torch.uint8)
        out.append(cache[key])
    return out


def _build(source: str, conversations, example: dict):
    """Run a preprocessor end-to-end through ``_build_conversation_list``.

    ``example`` carries raw image/video refs (opaque objects are fine — they are
    not decoded here); the helper builds distinct fake tensors keyed on ref
    identity (one per unique ref) of the lengths the preprocessor reports via
    ``image_refs`` / ``video_refs``, so the sequential pairing in
    ``_build_conversation_list`` is exercised and duplicate copy-image refs
    resolve to identical content.
    """
    constructed, image_refs, video_refs = conv_preprocess(source, conversations, example)
    images = _fake_tensors_for_refs(image_refs)
    videos = [object() for _ in range(len(video_refs))]  # opaque VideoInputs stand-ins
    items = _build_conversation_list(constructed, images, videos)
    return constructed, image_refs, video_refs, items


# ── iter_desired_images selector ───────────────────────────────────────────────


def test_iter_desired_images_no_filter_returns_all_images():
    batch = [[_img("und"), _img("gen"), _img("edit")]]
    items = list(iter_desired_images(batch))
    assert [it.meta.get(_IMG_TAG_KEY) for it in items] == ["und", "gen", "edit"]


def test_iter_desired_images_filters_by_each_tag():
    batch = [[_img("und"), _img("gen"), _img("edit"), _img("und"), _img("gen")]]
    assert [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["und"])] == ["und", "und"]
    assert [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["gen"])] == ["gen", "gen"]
    assert [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["edit"])] == ["edit"]


def test_iter_desired_images_tag_set_matches_multiple():
    batch = [[_img("und"), _img("gen"), _img("edit")]]
    got = [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["und", "edit"])]
    assert got == ["und", "edit"]


def test_iter_desired_images_never_selects_text_items():
    batch = [[_img("und"), _txt("user"), _txt("assistant")]]
    items = list(iter_desired_images(batch))
    assert len(items) == 1
    assert items[0].type == "image"


def test_iter_desired_images_composes_with_roles_and_sources():
    batch = [
        [
            _img("und", role="user", source="siglip"),
            _img("gen", role="assistant", source="vqvae"),
            _img("edit", role="user", source="siglip"),
        ]
    ]
    by_role = [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, roles=["user"])]
    assert by_role == ["und", "edit"]
    by_source = [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, sources=["siglip"])]
    assert by_source == ["und", "edit"]
    by_both = [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, roles=["assistant"], sources=["vqvae"])]
    assert by_both == ["gen"]
    with_tag_and_role = [
        it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["edit"], roles=["user"])
    ]
    assert with_tag_and_role == ["edit"]


def test_iter_desired_images_micro_batch_order_and_reverse():
    a, b, c = _img("und"), _img("gen"), _img("edit")
    batch = [[a, b], [c]]
    assert list(iter_desired_images(batch)) == [a, b, c]
    # reverse_item reverses within each sample.
    assert list(iter_desired_images(batch, reverse_item=True)) == [b, a, c]


def test_iter_desired_images_untagged_image_excluded_when_tag_filter_set():
    # An image with no _img_tag is dropped when a tag filter is active, but kept
    # when img_tag is None (selector must not crash on missing meta key).
    untagged = ConversationItem(type="image", value=torch.zeros(3, 2, 2, dtype=torch.uint8), role="user")
    batch = [[untagged, _img("gen")]]
    assert [it.meta.get(_IMG_TAG_KEY) for it in iter_desired_images(batch, img_tag=["gen"])] == ["gen"]
    assert len(list(iter_desired_images(batch))) == 2


# ── 3-tuple meta channel in _build_conversation_list ───────────────────────────


def test_two_tuple_still_builds_with_empty_meta():
    constructed = [["user", ("text", "hi")], ["assistant", ("image", None)]]
    items = _build_conversation_list(constructed, _fake_images(1), [])
    assert items[0].type == "text" and items[0].value == "hi"
    assert items[1].type == "image" and items[1].meta == {}
    # Fill value 0 — sequential pairing of the first (only) image tensor.
    assert int(items[1].value.float().mean()) == 0


def test_three_tuple_meta_merges_into_item_meta():
    constructed = [["assistant", ("image", None, {_IMG_TAG_KEY: "gen", "extra": 7})]]
    items = _build_conversation_list(constructed, _fake_images(1), [])
    assert items[0].meta == {_IMG_TAG_KEY: "gen", "extra": 7}


def test_three_tuple_meta_does_not_alias_preprocessor_dict():
    src_meta = {_IMG_TAG_KEY: "edit"}
    constructed = [["assistant", ("image", None, src_meta)]]
    items = _build_conversation_list(constructed, _fake_images(1), [])
    items[0].meta["mutated"] = True
    assert "mutated" not in src_meta  # item meta is a copy, not the source dict


def test_leftover_assert_fires_when_image_turns_underconsume():
    # Pure sequential: 1 image turn but 2 tensors supplied → the leftover-image
    # assert must catch the count mismatch.
    constructed = [["user", ("image", None)]]
    with pytest.raises(AssertionError):
        _build_conversation_list(constructed, _fake_images(2), [])


def test_leftover_assert_fires_when_image_turns_overconsume():
    # 2 image turns but only 1 tensor → next() raises StopIteration (the build
    # itself must fail loudly rather than silently misaligning).
    constructed = [["user", ("image", None)], ["assistant", ("image", None)]]
    with pytest.raises((AssertionError, StopIteration, RuntimeError)):
        _build_conversation_list(constructed, _fake_images(1), [])


# ── existing preprocessors tag image entries ───────────────────────────────────


def test_imagenet1k_tags_assistant_image_gen():
    example = {"images": [object()]}
    _, _, _, items = _build("imagenet1k", "cat", example)
    types = [(it.type, it.role) for it in items]
    assert types == [("text", "user"), ("image", "assistant")]
    assert items[0].meta.get(_IMG_TAG_KEY) is None  # text entry untagged
    assert items[1].meta.get(_IMG_TAG_KEY) == "gen"


def test_sharegpt4v_tags_user_image_und():
    conv = [
        {"from": "human", "value": "<image>describe this"},
        {"from": "gpt", "value": "a cat"},
    ]
    example = {"images": [object()]}
    _, _, _, items = _build("sharegpt4v_sft", conv, example)
    img = next(it for it in items if it.type == "image")
    assert img.role == "user"
    assert img.meta.get(_IMG_TAG_KEY) == "und"
    # Text entries (user instruction + assistant reply) stay untagged.
    for it in items:
        if it.type == "text":
            assert _IMG_TAG_KEY not in it.meta


def test_llava_video_tags_user_video_und():
    conv = [
        {"from": "human", "value": "<image>describe the clip"},
        {"from": "gpt", "value": "a running dog"},
    ]
    example = {"videos": [object()]}
    _, _, _, items = _build("llava_video", conv, example)
    video = next(it for it in items if it.type == "video")
    assert video.role == "user"
    assert video.meta.get(_IMG_TAG_KEY) == "und"
    for it in items:
        if it.type == "text":
            assert _IMG_TAG_KEY not in it.meta


def test_tulu_text_items_have_no_img_tag():
    conv = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    example: dict = {}
    _, _, _, items = _build("tulu-3-sft-mixture", conv, example)
    assert all(it.type == "text" for it in items)
    for it in items:
        assert _IMG_TAG_KEY not in it.meta
        assert it.meta == {}


def test_preprocessors_return_matching_refs():
    # Non-edit preprocessors echo the sample's media refs unchanged so the
    # transform's fetch_images/fetch_videos see a ref list whose length matches
    # the flattened image/video entry count (no expansion needed).
    img_ref = object()
    vid_ref = object()

    _, image_refs, video_refs = conv_preprocess("imagenet1k", "cat", {"images": [img_ref]})
    assert image_refs == [img_ref] and video_refs == []

    conv = [{"from": "human", "value": "<image>describe"}, {"from": "gpt", "value": "a cat"}]
    _, image_refs, video_refs = conv_preprocess("sharegpt4v_sft", conv, {"images": [img_ref]})
    assert image_refs == [img_ref] and video_refs == []

    _, image_refs, video_refs = conv_preprocess("llava_video", conv, {"videos": [vid_ref]})
    assert image_refs == [] and video_refs == [vid_ref]

    tulu = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    _, image_refs, video_refs = conv_preprocess("tulu-3-sft-mixture", tulu, {})
    assert image_refs == [] and video_refs == []


# ── seed_edit_p23_multi_turn preprocessor ──────────────────────────────────────


def _edit_conv(n_targets: int) -> list[dict]:
    # human source first, then (human instr, gpt target) pairs. n_targets gpt
    # images ⇒ total <image> tokens = 1 + n_targets.
    conv = [{"from": "human", "value": "<image>edit it to be red"}]
    for i in range(n_targets):
        conv.append({"from": "gpt", "value": "<image>"})
        if i < n_targets - 1:
            conv.append({"from": "human", "value": f"now make it color {i + 2}"})
    return conv


def test_seed_edit_multi_turn_chain_structure_three_targets():
    n_targets = 3
    conv = _edit_conv(n_targets)
    refs = [f"img{i}" for i in range(1 + n_targets)]  # 4 distinct string refs
    example = {"images": refs}
    constructed, image_refs, video_refs = conv_preprocess("seed_edit_p23_multi_turn", conv, example)

    # Expected flattened chain: S0, instr1, T1, S1(copy), instr2, T2, S2(copy),
    # instr3, T3 (final, no copy).
    expected = [
        ("image", "user", "edit"),
        ("text", "user", None),
        ("image", "assistant", "gen"),
        ("image", "assistant", "edit"),
        ("text", "user", None),
        ("image", "assistant", "gen"),
        ("image", "assistant", "edit"),
        ("text", "user", None),
        ("image", "assistant", "gen"),
    ]
    actual = [
        (e[0], turn[0], e[2].get(_IMG_TAG_KEY) if len(e) == 3 else None) for turn in constructed for e in turn[1:]
    ]
    assert actual == expected

    # image_refs length matches the flattened image entry count, with copies
    # duplicating the preceding target's ref: [img0, img1, img1, img2, img2, img3].
    n_image_entries = sum(1 for turn in constructed for e in turn[1:] if e[0] == "image")
    assert len(image_refs) == n_image_entries == 6
    assert image_refs == [refs[0], refs[1], refs[1], refs[2], refs[2], refs[3]]
    assert video_refs == []


def test_seed_edit_multi_turn_image_refs_expansion_matches_primaries_and_copies():
    # Direct check of the ref-list expansion the preprocessor now owns: each
    # primary <image> gets example["images"][idx]; each copy re-appends the
    # preceding target's ref (same object) without advancing the primary index.
    n_targets = 3
    conv = _edit_conv(n_targets)
    refs = [object() for _ in range(1 + n_targets)]  # 4 distinct opaque refs
    example = {"images": refs}
    _, image_refs, _ = conv_preprocess("seed_edit_p23_multi_turn", conv, example)

    assert image_refs == [refs[0], refs[1], refs[1], refs[2], refs[2], refs[3]]
    # Copies are the very same object as the preceding target's ref.
    assert image_refs[2] is refs[1]
    assert image_refs[4] is refs[2]
    # example["images"] is never mutated.
    assert example["images"] == refs
    assert len(example["images"]) == 1 + n_targets


def test_seed_edit_multi_turn_sequential_pairing_resolves_copies():
    # End-to-end through _build_conversation_list: pure sequential pairing of the
    # expanded ref list. Copy items get the same content as their preceding
    # target because the helper keys fake tensors on ref identity.
    n_targets = 3
    conv = _edit_conv(n_targets)
    example = {"images": [object() for _ in range(1 + n_targets)]}
    _, _, _, items = _build("seed_edit_p23_multi_turn", conv, example)
    img_items = [it for it in items if it.type == "image"]

    assert len(img_items) == 6  # 4 primaries + 2 copies
    # source0 → fill 0; target_i → fill i; copy of target_i reuses the same ref.
    assert int(img_items[0].value.float().mean()) == 0  # source0
    assert int(img_items[1].value.float().mean()) == 1  # target1
    assert torch.equal(img_items[2].value, img_items[1].value)  # source1 = copy of target1
    assert int(img_items[3].value.float().mean()) == 2  # target2
    assert torch.equal(img_items[4].value, img_items[3].value)  # source2 = copy of target2
    assert int(img_items[5].value.float().mean()) == 3  # target3 (final)
    assert img_items[-1].meta.get(_IMG_TAG_KEY) == "gen"

    # Text items carry no _img_tag.
    for it in items:
        if it.type == "text":
            assert _IMG_TAG_KEY not in it.meta


def test_seed_edit_multi_turn_last_target_has_no_copy_single_target():
    # Minimal non-trivial chain: source + exactly one target (the last), so no
    # copy should be emitted at all.
    conv = [
        {"from": "human", "value": "<image>edit it"},
        {"from": "gpt", "value": "<image>"},
    ]
    example = {"images": [object(), object()]}
    constructed, image_refs, video_refs = conv_preprocess("seed_edit_p23_multi_turn", conv, example)
    flat = [(e[0], turn[0], e[2].get(_IMG_TAG_KEY) if len(e) == 3 else None) for turn in constructed for e in turn[1:]]
    assert flat == [
        ("image", "user", "edit"),
        ("text", "user", None),
        ("image", "assistant", "gen"),
    ]
    # No copy emitted for the final (only) target → image_refs == the two primaries.
    assert len(image_refs) == 2
    assert image_refs == [example["images"][0], example["images"][1]]
    assert video_refs == []

    _, _, _, items = _build("seed_edit_p23_multi_turn", conv, example)
    img_items = [it for it in items if it.type == "image"]
    assert len(img_items) == 2  # no copy
    assert img_items[-1].meta.get(_IMG_TAG_KEY) == "gen"


# ── fetch_images ref dedupe ────────────────────────────────────────────────────


def test_fetch_images_dedupes_repeated_bytes_ref_and_clones():
    import io

    from PIL import Image

    from veomni.data.seed_omni.image_utils import fetch_images

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (7, 7, 7)).save(buf, format="PNG")
    ref = buf.getvalue()  # same bytes object appended 3x
    out = fetch_images([ref, ref, ref])
    assert len(out) == 3
    assert all(tuple(t.shape) == (3, 4, 4) for t in out)
    # Same decoded content, but each is an independent tensor (clone on reuse).
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])
    assert out[0] is not out[1] and out[1] is not out[2]
    out[0][0, 0, 0] = 255
    assert int(out[1][0, 0, 0]) == 7  # clone is independent of the first


def test_fetch_images_decodes_repeated_ref_once_and_returns_independent_clones(monkeypatch):
    from PIL import Image

    from veomni.data.seed_omni import image_utils

    calls = 0

    def fake_load_image(ref):
        nonlocal calls
        calls += 1
        assert ref == "same-image"
        return Image.new("RGB", (2, 2), (11, 11, 11))

    monkeypatch.setattr(image_utils, "load_image", fake_load_image)

    out = image_utils.fetch_images(["same-image", "same-image", "same-image"])

    assert calls == 1
    assert len(out) == 3
    assert all(torch.equal(out[0], item) for item in out[1:])
    out[0][0, 0, 0] = 255
    assert int(out[1][0, 0, 0]) == 11


def test_fetch_images_unique_refs_each_decoded():
    import io

    from PIL import Image

    from veomni.data.seed_omni.image_utils import fetch_images

    buf1 = io.BytesIO()
    Image.new("RGB", (4, 4), (7, 7, 7)).save(buf1, format="PNG")
    ref1 = buf1.getvalue()
    buf2 = io.BytesIO()
    Image.new("RGB", (4, 4), (9, 9, 9)).save(buf2, format="PNG")
    ref2 = buf2.getvalue()
    out = fetch_images([ref1, ref2])
    assert int(out[0].float().mean()) == 7
    assert int(out[1].float().mean()) == 9


def test_fetch_images_dedupes_repeated_pil_ref_by_identity():
    from PIL import Image

    from veomni.data.seed_omni.image_utils import fetch_images

    pil = Image.new("RGB", (4, 4), (5, 5, 5))
    out = fetch_images([pil, pil])  # same PIL object twice → identity-dedupe
    assert len(out) == 2
    assert torch.equal(out[0], out[1])
    assert out[0] is not out[1]  # clone on reuse


def test_fetch_videos_decodes_repeated_ref_once_and_returns_independent_copies(monkeypatch):
    import numpy as np

    from veomni.data.seed_omni import video_utils

    calls = 0

    def fake_load_video(ref, **kwargs):
        nonlocal calls
        calls += 1
        assert ref == "same-video"
        return video_utils.VideoInputs(
            video=torch.full((2, 3, 2, 2), 13, dtype=torch.uint8),
            video_fps=kwargs["fps"],
            audio=np.ones((4,), dtype=np.float32),
            audio_fps=16000.0,
        )

    monkeypatch.setattr(video_utils, "load_video", fake_load_video)

    out = video_utils.fetch_videos(["same-video", "same-video"], fps=3.0, use_audio_in_video=True)

    assert calls == 1
    assert len(out) == 2
    assert out[0] is not out[1]
    assert torch.equal(out[0].video, out[1].video)
    assert out[0].audio is not out[1].audio

    out[0].video[0, 0, 0, 0] = 255
    out[0].audio[0] = 7.0
    assert int(out[1].video[0, 0, 0, 0]) == 13
    assert float(out[1].audio[0]) == 1.0
