"""Worker-side CPU preprocessor hooks + naflatten/unflatten CPU-shape fix.

Covers the SeedOmni V2 optimization that moves each module's heavy CPU input-prep
(chat-template + tokenize, image normalize) into the DataLoader worker via a
picklable ``Preprocessor`` run inside ``SeedOmniCollator``:

* ``naflatten``/``unflatten`` keep shape metadata on CPU (no per-segment D2H sync)
  and round-trip correctly.
* The shared :class:`TextEncoderPreprocessor` (wrapping a per-model
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
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_POSITION_IDS as BAGEL_SIGLIP_POSITION_IDS,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    _OMNI_TOKEN_LEN as BAGEL_SIGLIP_TOKEN_LEN,
)
from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import (
    BagelSiglipNavitPreprocessor,
    BagelSiglipNavitProcessor,
)
from veomni.models.seed_omni.modules.bagel.sources import BAGEL_SIGLIP_CONTEXT, BAGEL_VAE_CONTEXT
from veomni.models.seed_omni.modules.bagel.text_encoder.chat_template import BagelChatTemplate
from veomni.models.seed_omni.modules.bagel.text_encoder.modulemixin import (
    _OMNI_TOKENIZED as BAGEL_TOK,
)
from veomni.models.seed_omni.modules.bagel.text_encoder.processing import (
    BagelTextEncoderPreprocessor,
)
from veomni.models.seed_omni.modules.bagel.vae.configuration import BagelVAEConfig
from veomni.models.seed_omni.modules.bagel.vae.modulemixin import BagelVAEModuleMixin
from veomni.models.seed_omni.modules.bagel.vae.processing import (
    BAGEL_VAE_PIXEL_SHAPE,
    BagelVAEPreprocessor,
    BagelVAEProcessor,
)
from veomni.models.seed_omni.modules.janus.siglip.processing import (
    JanusSiglipPreprocessor,
)
from veomni.models.seed_omni.modules.janus.text_encoder.chat_template import JanusChatTemplate
from veomni.models.seed_omni.modules.janus.text_encoder.processing import JanusTextEncoderPreprocessor
from veomni.models.seed_omni.modules.janus.vqvae.processing import (
    JanusVqvaePreprocessor,
)
from veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template import Qwen3ChatTemplate
from veomni.models.seed_omni.modules.qwen3.text_encoder.processing import Qwen3TextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from veomni.models.seed_omni.modules.qwen3vl.text_encoder.processing import Qwen3VLTextEncoderPreprocessor
from veomni.models.seed_omni.modules.qwen3vl.vision.processing import (
    _OMNI_GRID,
    Qwen3VLVisionPreprocessor,
)
from veomni.models.seed_omni.utils.conversation import _IMG_TAG_KEY, ConversationItem, iter_desired_items
from veomni.utils.tensor_utils import naflatten, unflatten


def _worker_dummies(conversation_list, source):
    """Test helper: worker-appended ``role="dummy"`` placeholders for ``source``."""
    return list(iter_desired_items(conversation_list, roles=["dummy"], sources=[source]))


class _DummyBagelVAE(BagelVAEModuleMixin):
    def __init__(self, support_cache: bool = False, train_type: str = "train") -> None:
        self.config = BagelVAEConfig(support_cache=support_cache, train_type=train_type)
        self._image_processor = object()
        self.dtype = torch.float32
        super().__init__()

    def init_omni_state(self) -> None:
        return None


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


# ── Text encoder preprocessor (shared TextEncoderPreprocessor) ────────────────


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
    JanusTextEncoderPreprocessor(tmpl)(batch)
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
    JanusTextEncoderPreprocessor(tmpl)(batch)
    for part in batch[0]:
        if part.type == "text":
            assert isinstance(part.value, torch.Tensor) and part.value.dtype == torch.long
            assert part.value.device.type == "cpu"
            assert part.meta["labels"].shape == part.value.shape
            assert part.meta["attention_mask"].shape == part.value.shape


def test_bagel_text_preprocessor_tokenizes_plain_items_and_is_idempotent():
    pre = BagelTextEncoderPreprocessor(_bagel_template())
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=torch.zeros(3, 4, 4), role="user", source=BAGEL_SIGLIP_CONTEXT),
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
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
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


def test_bagel_siglip_preprocessor_appends_per_sample_dummy_for_missing_context():
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                role="user",
                source=BAGEL_SIGLIP_CONTEXT,
            )
        ],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]

    pre(batch)

    assert len(_worker_dummies(batch, BAGEL_SIGLIP_CONTEXT)) == 1
    assert len(batch[0]) == 1
    dummy = batch[1][-1]
    assert dummy.type == "image"
    assert dummy.role == "dummy"
    assert dummy.source == BAGEL_SIGLIP_CONTEXT
    assert dummy.meta[BAGEL_SIGLIP_TOKEN_LEN] == 1
    assert dummy.meta[BAGEL_SIGLIP_POSITION_IDS].tolist() == [0]
    assert dummy.value.shape == (1, 2 * 2 * 3)
    assert dummy.value.dtype == torch.bfloat16


def test_bagel_siglip_preprocessor_keeps_bs4_sample_aligned_for_0_2_4_images():
    pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(
            patch_size=2,
            image_size=4,
            min_image_size=2,
            max_pixels=16,
            vit_max_num_patch_per_side=2,
        ),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )

    for real_count in (0, 2, 4):
        batch = []
        for index in range(4):
            if index < real_count:
                batch.append(
                    [
                        ConversationItem(
                            type="image",
                            value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
                            role="user",
                            source=BAGEL_SIGLIP_CONTEXT,
                        )
                    ]
                )
            else:
                batch.append([ConversationItem(type="text", value=f"text-{index}", role="user")])

        pre(batch)

        assert len(_worker_dummies(batch, BAGEL_SIGLIP_CONTEXT)) == 4 - real_count
        for sample in batch:
            context_items = [item for item in sample if item.type == "image" and item.source == BAGEL_SIGLIP_CONTEXT]
            assert len(context_items) == 1


def test_bagel_vae_process_only_skips_preprocessor(tmp_path):
    process_only_dir = tmp_path / "process_only"
    BagelVAEConfig(support_cache=True, train_type="train_with_cache").save_pretrained(str(process_only_dir))
    assert BagelVAEPreprocessor.from_pretrained(str(process_only_dir)) is None

    full_dir = tmp_path / "full"
    BagelVAEConfig().save_pretrained(str(full_dir))
    assert isinstance(BagelVAEPreprocessor.from_pretrained(str(full_dir)), BagelVAEPreprocessor)


def test_bagel_vae_process_only_override_applies_even_when_checkpoint_default_is_full(tmp_path):
    """`modules_train_with_cache.yaml`'s `model_config: {support_cache: true,
    train_type: train_with_cache}` override must reach the preprocessor the same
    way it reaches the live model's config — regardless of the checkpoint's own
    on-disk default (regression: `from_pretrained` used to silently re-read the
    on-disk config and drop this override, always building a real preprocessor).
    """
    full_dir = tmp_path / "full_on_disk"
    BagelVAEConfig().save_pretrained(str(full_dir))

    assert isinstance(BagelVAEPreprocessor.from_pretrained(str(full_dir)), BagelVAEPreprocessor)
    overridden = BagelVAEPreprocessor.from_pretrained(
        str(full_dir), config_overrides={"support_cache": True, "train_type": "train_with_cache"}
    )
    assert overridden is None


def test_bagel_vae_process_only_full_hf_checkpoint_copies_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    _DummyBagelVAE(support_cache=True, train_type="train_with_cache").save_full_hf_checkpoint(
        str(output),
        source_path=str(source),
        trainer=object(),
        state=object(),
    )

    assert (output / "config.json").read_text(encoding="utf-8") == "{}"
    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_bagel_preprocessors_route_inference_edit_prompt_context():
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
    )
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    batch = [
        [
            ConversationItem(type="text", value="hi", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user"),
            ConversationItem(
                type="image", value=assistant_image.clone(), role="assistant", meta={_IMG_TAG_KEY: "gen"}
            ),
        ]
    ]

    for preprocessor in (vae_pre, siglip_pre, text_pre):
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
    assert sample[2].value.shape == (3, 4, 4)
    assert sample[2].value.dtype == torch.bfloat16
    assert sample[5].value.shape == (4, 2 * 2 * 3)
    assert sample[5].value.dtype == torch.bfloat16
    assert sample[5].meta[BAGEL_SIGLIP_TOKEN_LEN] == 4
    assert sample[8].value.shape == (3, 4, 4)
    assert sample[8].value.dtype == torch.bfloat16
    assert [sample[i].source for i in [1, 3, 4, 6, 7, 9]] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [int(sample[i].value.numel()) for i in [1, 3, 4, 6, 7, 9]] == [1, 1, 1, 1, 1, 1]


def test_bagel_preprocessors_route_tagged_edit_without_infer_type():
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    siglip_pre = BagelSiglipNavitPreprocessor(
        BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(1, 12, dtype=torch.bfloat16),
    )
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    user_image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    assistant_image = torch.full((3, 4, 4), 9, dtype=torch.uint8)
    batch = [
        [
            ConversationItem(type="text", value="edit", role="user"),
            ConversationItem(type="image", value=user_image.clone(), role="user", meta={_IMG_TAG_KEY: "edit"}),
            ConversationItem(
                type="image", value=assistant_image.clone(), role="assistant", meta={_IMG_TAG_KEY: "gen"}
            ),
        ]
    ]

    for preprocessor in (vae_pre, siglip_pre, text_pre):
        preprocessor(batch)

    sample = batch[0]
    assert [item.source for item in sample if item.type == "image"] == [
        BAGEL_VAE_CONTEXT,
        BAGEL_SIGLIP_CONTEXT,
        BAGEL_VAE_CONTEXT,
    ]
    assert [item.meta.get(_IMG_TAG_KEY) for item in sample if item.type == "image"] == ["edit", "edit", "gen"]
    vae_source, siglip_source, vae_target = (item for item in sample if item.type == "image")
    assert vae_source.value.shape == (3, 4, 4)
    assert vae_source.value.dtype == torch.bfloat16
    assert siglip_source.value.shape == (4, 2 * 2 * 3)
    assert siglip_source.value.dtype == torch.bfloat16
    assert vae_target.value.shape == (3, 4, 4)
    assert vae_target.value.dtype == torch.bfloat16


def test_bagel_preprocessors_route_inference_und_user_image_to_siglip_only():
    vae_pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    text_pre = BagelTextEncoderPreprocessor(_bagel_template())
    image = torch.full((3, 4, 4), 7, dtype=torch.uint8)
    batch = [[ConversationItem(type="image", value=image.clone(), role="user")]]

    vae_pre(batch, inference=True, generation_kwargs={"infer_type": "infer_und"})
    text_pre(batch, inference=True, generation_kwargs={"infer_type": "infer_und"})

    assert [item.type for item in batch[0]] == ["text", "image", "text"]
    assert batch[0][1].source == BAGEL_SIGLIP_CONTEXT
    assert torch.equal(batch[0][1].value, image)


def test_bagel_vae_preprocessor_appends_per_sample_dummy_for_missing_context():
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )
    batch = [
        [
            ConversationItem(
                type="image",
                value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
                role="assistant",
                meta={_IMG_TAG_KEY: "gen"},
            )
        ],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]

    pre(batch)

    real = batch[0][0]
    assert real.source == BAGEL_VAE_CONTEXT
    assert real.value.shape == (3, 4, 4)
    assert real.value.dtype == torch.bfloat16
    assert real.meta[BAGEL_VAE_PIXEL_SHAPE].tolist() == [4, 4]

    assert len(_worker_dummies(batch, BAGEL_VAE_CONTEXT)) == 1
    dummy = batch[1][-1]
    assert dummy.type == "image"
    assert dummy.role == "dummy"
    assert dummy.source == BAGEL_VAE_CONTEXT
    assert dummy.meta[BAGEL_VAE_PIXEL_SHAPE].tolist() == [4, 4]
    assert dummy.value.shape == real.value.shape
    assert dummy.value.dtype == torch.bfloat16
    torch.stack([real.value, dummy.value], dim=0)


def test_bagel_vae_preprocessor_keeps_bs4_sample_aligned_for_0_2_4_images():
    pre = BagelVAEPreprocessor(
        BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
        dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
    )

    for real_count in (0, 2, 4):
        batch = []
        for index in range(4):
            if index < real_count:
                batch.append(
                    [
                        ConversationItem(
                            type="image",
                            value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
                            role="assistant",
                            meta={_IMG_TAG_KEY: "gen"},
                        )
                    ]
                )
            else:
                batch.append([ConversationItem(type="text", value=f"text-{index}", role="user")])

        pre(batch)

        assert len(_worker_dummies(batch, BAGEL_VAE_CONTEXT)) == 4 - real_count
        for sample in batch:
            context_items = [item for item in sample if item.type == "image" and item.source == BAGEL_VAE_CONTEXT]
            assert len(context_items) == 1
        shapes = [tuple(sample[-1].value.shape) for sample in batch]
        assert len(set(shapes)) == 1
        expected_shape = (3, 2, 2) if real_count == 0 else (3, 4, 4)
        assert shapes[0] == expected_shape


# ── Qwen3 / Qwen3-VL text preprocessors ─────────────────────────────────────────


def _qwen3_text_sample():
    return [
        ConversationItem(type="text", value="hello", role="user"),
        ConversationItem(type="text", value="hi there", role="assistant"),
    ]


def test_qwen3_text_preprocessor_matches_inmodule_pipeline():
    tmpl = _qwen3_template()
    batch = [_qwen3_text_sample(), _qwen3_text_sample()]

    Qwen3TextEncoderPreprocessor(tmpl)(batch)
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


def test_qwen3_text_preprocessor_from_pretrained_applies_enable_image_override(tmp_path, monkeypatch):
    """`modules_train_visual_instruction_tuning.yaml`'s `model_config: {enable_image:
    true}` override must reach the preprocessor's template choice the same way it
    reaches the live model's config — regardless of the checkpoint's own on-disk
    default (regression: `from_pretrained` used to silently re-read the on-disk
    config and always fall back to the text-only template).
    """
    import veomni.models.seed_omni.modules.qwen3.text_encoder.processing as qwen3_text_processing
    from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig

    Qwen3TextEncoderConfig(enable_image=False).save_pretrained(str(tmp_path))
    monkeypatch.setattr(qwen3_text_processing, "build_tokenizer", lambda module_path: FakeTokenizer())

    default_pre = Qwen3TextEncoderPreprocessor.from_pretrained(str(tmp_path))
    assert isinstance(default_pre._chat_template, Qwen3ChatTemplate)

    override_pre = Qwen3TextEncoderPreprocessor.from_pretrained(str(tmp_path), config_overrides={"enable_image": True})
    assert isinstance(override_pre._chat_template, Qwen3VLChatTemplate)


def test_qwen3vl_text_preprocessor_tokenizes():
    tmpl = _qwen3vl_template()
    batch = [_qwen3_text_sample()]
    Qwen3VLTextEncoderPreprocessor(tmpl)(batch)
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

    Qwen3VLVisionPreprocessor(
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
    user_img = ConversationItem(
        type="image",
        value=torch.zeros(3, 4, 4, dtype=torch.uint8),
        role="user",
        meta={_IMG_TAG_KEY: "gen"},
    )
    asst_img = ConversationItem(
        type="image",
        value=torch.ones(3, 4, 4, dtype=torch.uint8),
        role="assistant",
        meta={_IMG_TAG_KEY: "und"},
    )
    batch = [[user_img, asst_img]]
    pre = Qwen3VLVisionPreprocessor(
        proc,
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )
    pre(batch)
    # Qwen3VL is role/type-driven: _img_tag metadata is ignored here.
    assert user_img.value.dtype == torch.bfloat16 and user_img.meta[_OMNI_GRID] == [1, 2, 2]
    assert asst_img.value.dtype == torch.uint8 and _OMNI_GRID not in asst_img.meta


# ── Image preprocessors (siglip = user, vqvae = assistant) ──────────────────────


def _raw_image_sample():
    return [
        ConversationItem(
            type="image",
            value=torch.full((3, 4, 4), 7, dtype=torch.uint8),
            role="user",
            meta={_IMG_TAG_KEY: "gen"},
        ),
        ConversationItem(type="text", value="caption", role="user"),
        ConversationItem(
            type="image",
            value=torch.full((3, 4, 4), 9, dtype=torch.uint8),
            role="assistant",
            meta={_IMG_TAG_KEY: "und"},
        ),
    ]


def test_siglip_preprocessor_normalizes_only_user_images():
    pre = JanusSiglipPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [_raw_image_sample()]
    pre(batch)
    user_img, _, assistant_img = batch[0]
    # Janus SigLIP is role-driven: user image is normalized even if _img_tag says gen.
    assert user_img.value.shape == (3, 4, 4) and user_img.value.dtype == torch.bfloat16
    # Assistant image untouched by the siglip preprocessor even if _img_tag says und.
    assert assistant_img.value.dtype == torch.uint8


def test_vqvae_preprocessor_normalizes_only_assistant_images():
    pre = JanusVqvaePreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [_raw_image_sample()]
    pre(batch)
    user_img, _, assistant_img = batch[0]
    # Janus VQVAE is role-driven: assistant image is normalized even if _img_tag says und.
    assert assistant_img.value.shape == (3, 4, 4) and assistant_img.value.dtype == torch.bfloat16
    # User image left to siglip even if _img_tag says gen.
    assert user_img.value.dtype == torch.uint8


# ── Collator wiring ─────────────────────────────────────────────────────────────


def test_collator_runs_preprocessors_in_order():
    from veomni.models.seed_omni.processing import OmniProcessor

    preprocessors = {
        "text": JanusTextEncoderPreprocessor(_janus_template()),
        "siglip": JanusSiglipPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        "vqvae": JanusVqvaePreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
    }
    processor = OmniProcessor(preprocessors)
    collator = SeedOmniCollator(processor=processor)
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
    JanusTextEncoderPreprocessor(_janus_template())(batch)
    JanusSiglipPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )(batch)
    JanusVqvaePreprocessor(
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
    pre = JanusSiglipPreprocessor(
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
    pre = JanusSiglipPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [[ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user")]]
    pre(batch)
    assert _worker_dummies(batch, "janus_siglip") == []
    assert batch[0][0].value.dtype == torch.bfloat16  # real image normalized instead


def test_siglip_appends_dummy_for_missing_samples_in_mixed_batch():
    pre = JanusSiglipPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [
        [ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user")],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]
    pre(batch)

    dummies = _worker_dummies(batch, "janus_siglip")
    assert len(dummies) == 1
    assert len(batch[0]) == 1
    assert dummies[0] is batch[1][-1]
    assert dummies[0].source == "janus_siglip"


def test_vqvae_appends_dummy_only_when_no_assistant_image():
    pre = JanusVqvaePreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    pre(batch)
    dummies = _worker_dummies(batch, "janus_vqvae")
    assert len(dummies) == len(batch)
    assert all(d.source == "janus_vqvae" and d.value.shape == (3, 4, 4) for d in dummies)


def test_vqvae_appends_dummy_for_missing_samples_in_mixed_batch():
    pre = JanusVqvaePreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = [
        [ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="assistant")],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]
    pre(batch)

    dummies = _worker_dummies(batch, "janus_vqvae")
    assert len(dummies) == 1
    assert len(batch[0]) == 1
    assert dummies[0] is batch[1][-1]
    assert dummies[0].source == "janus_vqvae"


def test_qwen3vl_vision_appends_dummy_with_grid_when_no_visual():
    proc = FakeQwen3VLImageProcessor()
    pre = Qwen3VLVisionPreprocessor(
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


def test_qwen3vl_vision_appends_dummy_for_missing_samples_in_mixed_batch():
    pre = Qwen3VLVisionPreprocessor(
        FakeQwen3VLImageProcessor(),
        None,
        dtype=torch.bfloat16,
        dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
        dummy_grid=[1, 2, 2],
    )
    batch = [
        [ConversationItem(type="image", value=torch.zeros(3, 4, 4, dtype=torch.uint8), role="user")],
        [ConversationItem(type="text", value="text-only", role="user")],
    ]
    pre(batch)

    dummies = _worker_dummies(batch, "qwen3vl_vision")
    assert len(dummies) == 1
    assert len(batch[0]) == 1
    assert dummies[0] is batch[1][-1]
    assert dummies[0].meta[_OMNI_GRID] == [1, 2, 2]
    assert dummies[0].source == "qwen3vl_vision"


def test_worker_dummy_routes_to_dummy_parts_in_text_template():
    # A worker-appended role="dummy" image item must survive Janus chat-template
    # (routed to dummy_parts at the end, no markers, value untouched).
    batch = _text_only_batch()
    JanusSiglipPreprocessor(
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
    siglip = JanusSiglipPreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    siglip(batch, inference=True)
    assert _worker_dummies(batch, "janus_siglip") == []

    vqvae = JanusVqvaePreprocessor(
        FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
    )
    batch = _text_only_batch()
    vqvae(batch, inference=True)
    assert _worker_dummies(batch, "janus_vqvae") == []

    vision = Qwen3VLVisionPreprocessor(
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
    Qwen3TextEncoderPreprocessor(tmpl)(train_batch)
    Qwen3TextEncoderPreprocessor(tmpl)(infer_batch, inference=True)

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
        JanusTextEncoderPreprocessor(_janus_template()),
        Qwen3TextEncoderPreprocessor(_qwen3_template()),
        Qwen3VLTextEncoderPreprocessor(_qwen3vl_template()),
        JanusSiglipPreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        JanusVqvaePreprocessor(
            FakeImageProcessor(), dtype=torch.bfloat16, dummy_pixel_values=torch.zeros(3, 4, 4, dtype=torch.bfloat16)
        ),
        BagelTextEncoderPreprocessor(_bagel_template()),
        BagelSiglipNavitPreprocessor(
            BagelSiglipNavitProcessor(patch_size=2, image_size=4, min_image_size=2, max_pixels=16),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(1, 2 * 2 * 3, dtype=torch.bfloat16),
        ),
        BagelVAEPreprocessor(
            BagelVAEProcessor(image_stride=2, min_image_size=4, max_image_size=4, max_pixels=16),
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(3, 2, 2, dtype=torch.bfloat16),
            dummy_pixel_shape=torch.tensor([2, 2], dtype=torch.long),
        ),
        Qwen3VLVisionPreprocessor(
            FakeQwen3VLImageProcessor(),
            None,
            dtype=torch.bfloat16,
            dummy_pixel_values=torch.zeros(4, 8, dtype=torch.bfloat16),
            dummy_grid=[1, 2, 2],
        ),
    ):
        restored = pickle.loads(pickle.dumps(pre))
        assert type(restored) is type(pre)
