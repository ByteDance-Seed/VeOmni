"""Worker-side CPU preprocessor hooks + naflatten/unflatten CPU-shape fix.

Covers the SeedOmni V2 optimization that moves each module's heavy CPU input-prep
(chat-template + tokenize, image normalize) into the DataLoader worker via a
picklable ``CPUPreprocessor`` run inside ``SeedOmniCollator``:

* ``naflatten``/``unflatten`` keep shape metadata on CPU (no per-segment D2H sync)
  and round-trip correctly.
* The shared :class:`TextEncoderCPUPreprocessor` (wrapping a per-model
  ``TextEncoderChatTemplate``) produces the same tokens the in-module
  ``tokenize_conversation`` would, targets the right role, and is picklable.
* The siglip / vqvae / qwen3vl-vision preprocessors normalize the right images,
  append dummies, are idempotent, and are picklable (worker-safe).
* ``SeedOmniCollator`` runs the preprocessors in order over the grouped batch and
  is a pure grouper when none are supplied.
"""

import copy
import pickle

import torch

from veomni.data.data_collator import SeedOmniCollator
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    _OMNI_POSITION_IDS as BAGEL_SIGLIP_POSITION_IDS,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    _OMNI_TOKEN_LEN as BAGEL_SIGLIP_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.modulemixin import (
    BagelSiglipNavitCPUPreprocessor,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import BagelSiglipNavitProcessor
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.text_encoder.chat_template import BagelChatTemplate
from veomni.models.seed_omni.modules.bagel.text_encoder.modulemixin import (
    _OMNI_TOKENIZED as BAGEL_TOK,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.modulemixin import (
    BagelTextEncoderCPUPreprocessor,
)
from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BagelVAECPUPreprocessor
from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor
from veomni.models.seed_omni.modules.janus.siglip.modulemixin import (
    JanusSiglipCPUPreprocessor,
)
from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate
from veomni.models.seed_omni.modules.janus.text_encoder.modulemixin import JanusTextEncoderCPUPreprocessor
from veomni.models.seed_omni.modules.janus.vqvae.modulemixin import (
    JanusVqvaeCPUPreprocessor,
)
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.modules.qwen3.text_encoder.modulemixin import Qwen3TextEncoderCPUPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.modulemixin import Qwen3VLTextEncoderCPUPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.vision.modulemixin import (
    _OMNI_GRID,
    Qwen3VLVisionCPUPreprocessor,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem, iter_desired_items
from veomni.utils.tensor_utils import naflatten, unflatten


def _worker_dummies(conversation_list, source):
    """Test helper: worker-appended ``role="dummy"`` placeholders for ``source``."""
    return list(iter_desired_items(conversation_list, roles=["dummy"], sources=[source]))


# Module-level fakes so the preprocessors stay picklable (workers fork/spawn them).
class FakeTokenizer:
    """Char-ordinal tokenizer with the marker tokens/ids the chat templates resolve."""

    bos_token = "<s>"
    eos_token = "</s>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    bos_token_id = 1
    eos_token_id = 2
    boi_token_id = 3
    eoi_token_id = 4
    pad_token_id = 0
    unk_token_id = -1
    special_tokens = {
        "<s>": 1,
        "</s>": 2,
        "<boi>": 3,
        "<eoi>": 4,
        "<|im_start|>": 5,
        "<|im_end|>": 2,
        "<|vision_start|>": 7,
        "<|vision_end|>": 8,
    }

    def convert_tokens_to_ids(self, token):
        return self.special_tokens.get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        input_ids = []
        index = 0
        while index < len(text):
            matched = False
            for token, token_id in sorted(self.special_tokens.items(), key=lambda item: len(item[0]), reverse=True):
                if text.startswith(token, index):
                    input_ids.append(token_id)
                    index += len(token)
                    matched = True
                    break
            if matched:
                continue
            input_ids.append(ord(text[index]))
            index += 1
        return {"input_ids": input_ids}


class FakeImageProcessor:
    """Returns a fixed-shape normalized pixel batch (fp32) for the given images."""

    def __init__(self, channels=3, size=4):
        self.channels = channels
        self.size = size

    def __call__(self, images, return_tensors="pt"):
        # Deterministic content keyed on the (uint8) input so equivalence is checkable.
        px = torch.stack([img.float().mean() + torch.zeros(self.channels, self.size, self.size) for img in images])
        return {"pixel_values": px}


class FakeQwen3VLImageProcessor:
    """Returns (pixel_values, image_grid_thw) with a fixed grid per image."""

    def __init__(self, patch_dim=8, grid=(1, 2, 2)):
        self.patch_dim = patch_dim
        self.grid = list(grid)

    def __call__(self, images, return_tensors="pt"):
        grids = [self.grid for _ in images]
        total = sum(g[0] * g[1] * g[2] for g in grids)
        pv = torch.arange(total * self.patch_dim, dtype=torch.float32).reshape(total, self.patch_dim)
        return {"pixel_values": pv, "image_grid_thw": torch.tensor(grids, dtype=torch.long)}


def _janus_template() -> JanusChatTemplate:
    return JanusChatTemplate(FakeTokenizer())


def _qwen3_template() -> Qwen3ChatTemplate:
    return Qwen3ChatTemplate(FakeTokenizer())


def _qwen3vl_template() -> Qwen3VLChatTemplate:
    return Qwen3VLChatTemplate(FakeTokenizer())


def _bagel_template() -> BagelChatTemplate:
    return BagelChatTemplate(FakeTokenizer())


# ── naflatten / unflatten: shape stays on CPU, round-trips ──────────────────────


def test_naflatten_shape_on_cpu_and_roundtrip_1d():
    parts = [torch.arange(3), torch.arange(5), torch.arange(2)]
    flat, shape = naflatten(parts)
    assert shape.device.type == "cpu"
    out = unflatten(flat, shape)
    assert all(torch.equal(a, b) for a, b in zip(out, parts))


def test_naflatten_shape_on_cpu_and_roundtrip_2d():
    parts = [torch.randn(3, 4), torch.randn(5, 4), torch.randn(1, 4)]
    flat, shape = naflatten(parts)
    assert shape.device.type == "cpu"
    assert tuple(flat.shape) == (9, 4)
    out = unflatten(flat, shape)
    assert all(torch.equal(a, b) for a, b in zip(out, parts))


def test_unflatten_accepts_non_cpu_shape_without_error():
    # Robustness: a caller passing a shape on another device must still work.
    parts = [torch.randn(2, 4), torch.randn(3, 4)]
    flat, shape = naflatten(parts)
    out = unflatten(flat, shape.to("cpu"))  # explicit cpu path is a no-op move
    assert all(torch.equal(a, b) for a, b in zip(out, parts))


# ── Text encoder preprocessor (shared TextEncoderCPUPreprocessor) ────────────────


def _raw_text_sample():
    return [
        ConversationItem(type="text", value="describe", role="user"),
        ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user"),
        ConversationItem(type="text", value="more", role="user"),
        ConversationItem(type="text", value="hi", role="assistant"),
    ]


def test_text_preprocessor_matches_inmodule_pipeline():
    tmpl = _janus_template()
    batch = [_raw_text_sample(), _raw_text_sample()]

    # Worker path (mutates batch in place).
    JanusTextEncoderCPUPreprocessor(tmpl)(batch)
    worker_ids = []
    for sample in batch:
        worker_ids.extend(tmpl.pack_input_ids(sample))
    worker_flat, worker_shape = naflatten(worker_ids)

    # Independent reconstruction via the chat-template's own pipeline.
    ref_ids = []
    for sample in [_raw_text_sample(), _raw_text_sample()]:
        parts = tmpl.tokenize_conversation(sample)
        ref_ids.extend(tmpl.pack_input_ids(parts))
    ref_flat, ref_shape = naflatten(ref_ids)

    assert torch.equal(worker_flat, ref_flat)
    assert torch.equal(worker_shape, ref_shape)


def test_text_preprocessor_sets_labels_and_mask_on_cpu():
    tmpl = _janus_template()
    batch = [_raw_text_sample()]
    JanusTextEncoderCPUPreprocessor(tmpl)(batch)
    for part in batch[0]:
        if part.type == "text":
            assert isinstance(part.value, torch.Tensor) and part.value.dtype == torch.long
            assert part.value.device.type == "cpu"
            assert part.meta["labels"].shape == part.value.shape
            assert part.meta["attention_mask"].shape == part.value.shape


def test_bagel_text_preprocessor_tokenizes_plain_items_and_is_idempotent():
    pre = BagelTextEncoderCPUPreprocessor(_bagel_template())
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=torch.zeros(3, 4, 4), role="user"),
            ConversationItem(type="text", value="ok", role="assistant"),
        ]
    ]

    pre(batch)
    user_text, image_start, image, image_end, assistant_text = batch[0]

    assert image.type == "image"
    assert image.source == BAGEL_SIGLIP_CONTEXT
    assert image_start.type == "text"
    assert image_start.source == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(image_start.value, torch.tensor([7]))
    assert image_end.type == "text"
    assert image_end.source == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(image_end.value, torch.tensor([8]))
    assert torch.equal(user_text.value, torch.tensor([5, ord("h"), ord("i"), 2]))
    assert user_text.value.device.type == "cpu"
    assert user_text.meta[BAGEL_TOK] is True
    assert torch.equal(user_text.meta["labels"], torch.full_like(user_text.value, -100))
    assert torch.equal(user_text.meta["attention_mask"], torch.ones_like(user_text.value))

    assert torch.equal(assistant_text.value, torch.tensor([5, ord("o"), ord("k"), 2]))
    assert assistant_text.meta[BAGEL_TOK] is True
    assert torch.equal(assistant_text.meta["labels"], assistant_text.value)

    snapshot = copy.deepcopy(batch)
    pre(batch)
    assert len(batch[0]) == len(snapshot[0])
    for actual, expected in zip(batch[0], snapshot[0]):
        if isinstance(actual.value, torch.Tensor):
            assert torch.equal(actual.value, expected.value)


def test_bagel_siglip_preprocessor_patchifies_and_tags_context():
    pre = BagelSiglipNavitCPUPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            )
        ]
    ]

    pre(batch)
    item = batch[0][0]
    assert item.source == BAGEL_SIGLIP_CONTEXT
    assert item.meta[BAGEL_SIGLIP_TOKEN_LEN] == 4
    assert item.meta[BAGEL_SIGLIP_POSITION_IDS].tolist() == [0, 1, 2, 3]
    assert item.value.shape == (4, 2 * 2 * 3)
    assert item.value.dtype == torch.bfloat16


def test_bagel_text_preprocessor_routes_inference_edit_prompt_context():
    text_pre = BagelTextEncoderCPUPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitCPUPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
    )
    vae_pre = BagelVAECPUPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
    )
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user"),
            ConversationItem(type="image", value=assistant_image.clone(), role="assistant"),
        ]
    ]

    for preprocessor in (text_pre, siglip_pre, vae_pre):
        preprocessor(batch, inference=True, generation_kwargs={"infer_type": "infer_edit"})

    sample = batch[0]
    assert [item.type for item in sample] == [
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
        "text",
        "image",
        "text",
    ]
    assert torch.equal(sample[0].value, torch.tensor([5, ord("h"), ord("i"), 2]))
    assert sample[2].source == BAGEL_VAE_CONTEXT
    assert sample[5].source == BAGEL_SIGLIP_CONTEXT
    assert sample[8].source == BAGEL_VAE_CONTEXT
    assert torch.equal(sample[2].value, user_image)
    assert torch.equal(sample[5].value, user_image)
    assert torch.equal(sample[8].value, assistant_image)
    assert [sample[i].source for i in [1, 3, 4, 6, 7, 9]] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [int(sample[i].value.numel()) for i in [1, 3, 4, 6, 7, 9]] == [1, 1, 1, 1, 1, 1]


def test_bagel_text_preprocessor_routes_inference_und_user_image_to_siglip_only():
    text_pre = BagelTextEncoderCPUPreprocessor(_bagel_template())
    image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    batch = [[ConversationItem(type="image", value=image.clone(), role="user")]]

    text_pre(batch, inference=True, generation_kwargs={"infer_type": "infer_und"})

    assert [item.type for item in batch[0]] == ["text", "image", "text"]
    assert batch[0][1].source == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(batch[0][1].value, image)


# ── Qwen3 / Qwen3-VL text preprocessors ─────────────────────────────────────────


def _qwen3_text_sample():
    return [
        ConversationItem(type="text", value="hello", role="user"),
        ConversationItem(type="text", value="hi there", role="assistant"),
    ]


def test_qwen3_text_preprocessor_matches_inmodule_pipeline():
    tmpl = _qwen3_template()
    batch = [_qwen3_text_sample(), _qwen3_text_sample()]

    Qwen3TextEncoderCPUPreprocessor(tmpl)(batch)
    worker_ids = []
    for sample in batch:
        for part in sample:
            if part.type == "text":
                assert part.value.dtype == torch.long and part.value.device.type == "cpu"
        worker_ids.extend(tmpl.pack_input_ids(sample))
    worker_flat, worker_shape = naflatten(worker_ids)

    ref_ids = []
    for sample in [_qwen3_text_sample(), _qwen3_text_sample()]:
        parts = tmpl.tokenize_conversation(sample)
        ref_ids.extend(tmpl.pack_input_ids(parts))
    ref_flat, ref_shape = naflatten(ref_ids)

    assert torch.equal(worker_flat, ref_flat)
    assert torch.equal(worker_shape, ref_shape)


def test_qwen3vl_text_preprocessor_tokenizes():
    tmpl = _qwen3vl_template()
    batch = [_qwen3_text_sample()]
    Qwen3VLTextEncoderCPUPreprocessor(tmpl)(batch)
    sample = batch[0]
    for part in sample:
        if part.type == "text":
            assert part.value.dtype == torch.long and part.value.device.type == "cpu"
            assert part.meta["labels"].shape == part.value.shape


# ── Qwen3-VL vision preprocessor (patchify/normalize split + recombine) ──────────


def test_qwen3vl_vision_preprocessor_splits_and_recombines():
    proc = FakeQwen3VLImageProcessor(patch_dim=8, grid=(1, 2, 2))  # 4 patches/image
    items = [
        ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user") for _ in range(3)
    ]
    batch = [items]
    ref = proc(images=[it.value for it in items])["pixel_values"]

    Qwen3VLVisionCPUPreprocessor(
        proc,
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )(batch)

    recombined = torch.cat([it.value for it in items], dim=0)
    assert recombined.dtype == torch.bfloat16
    # Worker casts to bf16 per-item; recombine must equal the bf16 of the raw output.
    assert torch.equal(recombined, ref.to(torch.bfloat16))
    for it in items:
        # The worker stashes per-item grid on meta (the "already-processed" marker).
        assert it.meta[_OMNI_GRID] == [1, 2, 2]
        assert it.value.shape == (4, 8)


def test_qwen3vl_vision_preprocessor_normalizes_user_and_leaves_assistant_untouched():
    proc = FakeQwen3VLImageProcessor()
    user_img = ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user")
    asst_img = ConversationItem(type="image", value=torch.ones(3, 4, 4, dtype=torch.uint8), role="assistant")
    batch = [[user_img, asst_img]]
    pre = Qwen3VLVisionCPUPreprocessor(
        proc,
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )
    pre(batch)
    # User image normalized (patches on value + grid stashed); assistant image is
    # left raw — the vision tower only consumes user images.
    assert user_img.value.dtype == torch.bfloat16 and user_img.meta[_OMNI_GRID] == [1, 2, 2]
    assert asst_img.value.dtype == torch.uint8 and _OMNI_GRID not in asst_img.meta


# ── Image preprocessors (siglip = user, vqvae = assistant) ──────────────────────


def _raw_image_sample():
    return [
        ConversationItem(type="image", value=torch.full((3, 4, 4), 7, dtype=torch.uint8), role="user"),
        ConversationItem(type="text", value="caption", role="user"),
        ConversationItem(type="image", value=torch.full((3, 4, 4), 9, dtype=torch.uint8), role="assistant"),
    ]


def test_siglip_preprocessor_normalizes_only_user_images():
    pre = JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [_raw_image_sample()]
    pre(batch)
    user_img, _, assistant_img = batch[0]
    # User image normalized (uint8 → model-dtype pixel tensor).
    assert user_img.value.shape == (3, 4, 4) and user_img.value.dtype == torch.bfloat16
    # Assistant image untouched by the siglip (user-only) preprocessor.
    assert assistant_img.value.dtype == torch.uint8


def test_vqvae_preprocessor_normalizes_only_assistant_images():
    pre = JanusVqvaeCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [_raw_image_sample()]
    pre(batch)
    user_img, _, assistant_img = batch[0]
    # Assistant image normalized (uint8 → model-dtype pixel tensor); user image
    # left to siglip.
    assert assistant_img.value.shape == (3, 4, 4) and assistant_img.value.dtype == torch.bfloat16
    assert user_img.value.dtype == torch.uint8


# ── Collator wiring ─────────────────────────────────────────────────────────────


def test_collator_runs_preprocessors_in_order():
    preprocessors = [
        JanusTextEncoderCPUPreprocessor(_janus_template()),
        JanusSiglipCPUPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        JanusVqvaeCPUPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
    ]
    collator = SeedOmniCollator(cpu_preprocessors=tuple(preprocessors))
    features = [{"conversation_list": _raw_image_sample()}, {"conversation_list": _raw_text_sample()}]
    batch = collator(features)
    assert set(batch.keys()) == {"conversation_list"}
    assert len(batch["conversation_list"]) == 2
    # Text rows tokenized, user/assistant images normalized (siglip on user images,
    # vqvae on assistant images).
    for sample in batch["conversation_list"]:
        assert isinstance(sample[0].value, torch.Tensor)  # text rows tokenized
        for part in sample:
            if part.type == "image":
                assert part.value.dtype == torch.bfloat16  # normalized to model dtype


def test_repr_after_preprocessing_does_not_raise():
    # The preprocessors populate item.meta (labels / attention_mask / sentinels)
    # before BaseTrainer.preforward -> print_example reprs the micro-batch.
    # ConversationItem.__repr__ must handle non-empty meta (regression: it used
    # to crash because __value_repr__ took no value argument).
    batch = [_raw_image_sample()]
    JanusTextEncoderCPUPreprocessor(_janus_template())(batch)
    JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )(batch)
    JanusVqvaeCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )(batch)
    for sample in batch:
        for part in sample:
            assert isinstance(repr(part), str)  # must not raise


def test_collator_default_is_pure_grouper():
    collator = SeedOmniCollator()
    features = [{"conversation_list": _raw_text_sample()}]
    batch = collator(features)
    # No preprocessing: text stays a raw string.
    assert batch["conversation_list"][0][0].value == "describe"
    assert isinstance(batch["conversation_list"][0][0].value, str)


# ── Worker-built dummy placeholders (text-only / no-image micro-batches) ─────────


def _text_only_batch():
    return [
        [ConversationItem(type="text", value="hi", role="user")],
        [ConversationItem(type="text", value="yo", role="user")],
    ]


def test_siglip_appends_one_dummy_per_sample_when_no_user_image():
    pre = JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    pre(batch)
    dummies = _worker_dummies(batch, "janus_siglip")
    assert len(dummies) == len(batch)
    for d in dummies:
        assert d.type == "image" and d.role == "dummy"
        assert d.value.shape == (3, 4, 4) and d.value.dtype == torch.bfloat16
        assert d.source == "janus_siglip"


def test_siglip_no_dummy_when_user_image_present():
    pre = JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [[ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user")]]
    pre(batch)
    assert _worker_dummies(batch, "janus_siglip") == []
    assert batch[0][0].value.dtype == torch.bfloat16  # real image normalized instead


def test_vqvae_appends_dummy_only_when_no_assistant_image():
    pre = JanusVqvaeCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    pre(batch)
    dummies = _worker_dummies(batch, "janus_vqvae")
    assert len(dummies) == len(batch)
    assert all(d.source == "janus_vqvae" and d.value.shape == (3, 4, 4) for d in dummies)


def test_qwen3vl_vision_appends_dummy_with_grid_when_no_visual():
    proc = FakeQwen3VLImageProcessor()
    pre = Qwen3VLVisionCPUPreprocessor(
        proc,
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )
    batch = _text_only_batch()
    pre(batch)
    dummies = _worker_dummies(batch, "qwen3vl_vision")
    assert len(dummies) == len(batch)
    for d in dummies:
        assert d.value.shape == (4, 8) and d.value.dtype == torch.bfloat16
        assert d.meta[_OMNI_GRID] == [1, 2, 2] and d.source == "qwen3vl_vision"


def test_worker_dummy_routes_to_dummy_parts_in_text_template():
    # A worker-appended role="dummy" image item must survive Janus chat-template
    # (routed to dummy_parts at the end, no markers, value untouched).
    batch = _text_only_batch()
    JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )(batch)
    sample = batch[0]
    templated = _janus_template().apply_chat_template(sample)
    dummies = [p for p in templated if p.role == "dummy"]
    assert len(dummies) == 1
    assert dummies[-1] is templated[-1]  # dummy parts kept at the very end
    assert dummies[0].source == "janus_siglip"


# ── Inference flag (no dummies + generation prompt) ─────────────────────────────


def test_image_preprocessors_skip_dummy_in_inference():
    # At inference there is no FSDP gradient anchor, so a no-image request must not
    # gain dummy items (the per-module ``generate`` simply has nothing to encode).
    siglip = JanusSiglipCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    siglip(batch, inference=True)
    assert _worker_dummies(batch, "janus_siglip") == []

    vqvae = JanusVqvaeCPUPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    vqvae(batch, inference=True)
    assert _worker_dummies(batch, "janus_vqvae") == []

    vision = Qwen3VLVisionCPUPreprocessor(
        FakeQwen3VLImageProcessor(),
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )
    batch = _text_only_batch()
    vision(batch, inference=True)
    assert _worker_dummies(batch, "qwen3vl_vision") == []


def test_text_preprocessor_appends_generation_prompt_in_inference():
    # Inference must append the assistant generation prefix (training does not), so
    # the request is left ready for the model to start decoding.
    tmpl = _qwen3_template()
    train_batch = [_qwen3_text_sample()]
    infer_batch = [_qwen3_text_sample()]
    Qwen3TextEncoderCPUPreprocessor(tmpl)(train_batch)
    Qwen3TextEncoderCPUPreprocessor(tmpl)(infer_batch, inference=True)

    train_tokens = sum(p.value.numel() for p in train_batch[0] if p.type == "text")
    infer_tokens = sum(p.value.numel() for p in infer_batch[0] if p.type == "text")
    assert infer_tokens > train_tokens  # the generation prompt adds the assistant prefix

    # And matches the explicit add_generation_prompt pipeline exactly.
    ref = tmpl.tokenize_conversation(_qwen3_text_sample(), add_generation_prompt=True)
    ref_tokens = sum(p.value.numel() for p in ref if p.type == "text")
    assert infer_tokens == ref_tokens


# ── Picklability (worker-safe: no nn.Module captured) ───────────────────────────


def test_preprocessors_are_picklable():
    for pre in (
        JanusTextEncoderCPUPreprocessor(_janus_template()),
        Qwen3TextEncoderCPUPreprocessor(_qwen3_template()),
        Qwen3VLTextEncoderCPUPreprocessor(_qwen3vl_template()),
        JanusSiglipCPUPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        JanusVqvaeCPUPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        BagelTextEncoderCPUPreprocessor(_bagel_template()),
        BagelSiglipNavitCPUPreprocessor(
            BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
            dtype=torch.bfloat16,
        ),
        BagelVAECPUPreprocessor(
            BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
            dtype=torch.bfloat16,
        ),
        Qwen3VLVisionCPUPreprocessor(
            FakeQwen3VLImageProcessor(),
            None,
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
            dummy_grid=[1, 2, 2],
        ),
    ):
        restored = pickle.loads(pickle.dumps(pre))
        assert type(restored) is type(pre)
