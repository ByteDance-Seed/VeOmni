"""Smoke tests for Qwen3-VL SeedOmni modules (registry + M-RoPE helper)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.seed_omni.modules.helpers import load_from_omni, save_as_omni
from tests.tools.training_utils import make_eager_ops_config
from veomni.models.seed_omni.modules import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)
from veomni.models.seed_omni.modules.qwen3vl.llm.modeling import (
    qwen3vl_vision_position_advance,
    qwen3vl_vision_position_ids,
)
from veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated import build_qwen3vl_vit_metadata


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _save_fake_fast_tokenizer(module_path: Path) -> None:
    """Write a minimal ``PreTrainedTokenizerFast`` sidecar (no sentencepiece needed).

    Text-encoder ``preprocessor_class`` binding (``bind_module_assets``) always
    calls ``AutoTokenizer.from_pretrained`` on the checkpoint dir — real
    checkpoints always ship a tokenizer, so save-reload roundtrip tests need
    one too, matching what a real converted checkpoint looks like.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5}
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>"
    )
    tokenizer.save_pretrained(module_path)


def test_registry_resolves_qwen3vl_modules():
    for key, cfg_name, model_name in [
        ("qwen3vl_vision", "Qwen3VLVisionEncoderConfig", "Qwen3VLVisionEncoder"),
        ("qwen3vl_text_encoder", "Qwen3VLTextEncoderConfig", "Qwen3VLTextEncoder"),
        ("qwen3vl_llm", "Qwen3VLLlmConfig", "Qwen3VLLlm"),
    ]:
        assert OMNI_CONFIG_REGISTRY[key]().__name__ == cfg_name
        assert OMNI_MODEL_REGISTRY[key]().__name__ == model_name
    # Qwen3-VL reuses the Qwen2-VL image processor; video has its own processor.
    # The vision module needs both, so the registry returns an {image, video} dict.
    procs = OMNI_PROCESSOR_REGISTRY["qwen3vl_vision"]()
    assert procs["image"].__name__ == "Qwen2VLImageProcessor"
    assert procs["video"].__name__ == "Qwen3VLVideoProcessor"


def test_vision_dummy_forward_emits_real_shaped_zeros_without_fsdp(monkeypatch):
    """Off-FSDP the dummy vision forward skips the ViT but must still emit zeros
    shaped exactly like a real encode (image_embeds + one feature per deepstack
    layer, no ``None``), so forward_post never branches on the dummy."""
    import veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated as vision_accelerated

    monkeypatch.setattr(
        vision_accelerated, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=False, sp_enabled=False)
    )

    Enc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_vision"]()
    Cfg = OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()
    vc = dict(
        hidden_size=64,
        num_heads=4,
        depth=2,
        intermediate_size=128,
        out_hidden_size=64,
        deepstack_visual_indexes=[1],
        spatial_merge_size=2,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
    )
    enc = Enc(Cfg(vision_config=vc, ops_implementation=make_eager_ops_config())).eval()

    g = [1, 4, 4]  # two dummy placeholders
    pixel_values = torch.zeros(2 * 1 * 4 * 4, 3 * 2 * 16 * 16)
    grid_thw = torch.tensor([g, g], dtype=torch.long)
    vit_metadata = build_qwen3vl_vit_metadata([g, g], spatial_merge_size=2)

    real_emb, real_deep = enc._encode(pixel_values, grid_thw, vit_metadata)
    out = enc.forward(
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        vit_metadata=vit_metadata,
        is_dummy=True,
    )

    assert out["image_embeds"] is not None
    assert out["image_embeds"].shape == real_emb.shape
    assert out["image_embeds"].abs().sum().item() == 0.0
    assert len(out["deepstack_features"]) == len(real_deep)
    assert out["deepstack_features"][0].shape == real_deep[0].shape


def test_vision_dummy_forward_skips_vit_in_eval_even_under_fsdp(monkeypatch):
    """Inference (eval) needs no gradient anchor, so the dummy vision forward
    fabricates zeros even with FSDP enabled — the real ViT must not run."""
    import veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated as vision_accelerated

    monkeypatch.setattr(
        vision_accelerated, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=True, sp_enabled=False)
    )

    Enc = OMNI_ACCELERATED_MODEL_REGISTRY["qwen3vl_vision"]()
    Cfg = OMNI_CONFIG_REGISTRY["qwen3vl_vision"]()
    vc = dict(
        hidden_size=64,
        num_heads=4,
        depth=2,
        intermediate_size=128,
        out_hidden_size=64,
        deepstack_visual_indexes=[1],
        spatial_merge_size=2,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
    )
    enc = Enc(Cfg(vision_config=vc, ops_implementation=make_eager_ops_config())).eval()

    def _boom(*_a, **_k):
        raise AssertionError("ViT must not run for a dummy in eval mode")

    monkeypatch.setattr(enc, "_encode", _boom)
    g = [1, 4, 4]
    out = enc.forward(
        pixel_values=torch.zeros(1 * 4 * 4, 3 * 2 * 16 * 16),
        image_grid_thw=torch.tensor([g], dtype=torch.long),
        vit_metadata=build_qwen3vl_vit_metadata([g], spatial_merge_size=2),
        is_dummy=True,
    )
    assert out["image_embeds"].abs().sum().item() == 0.0
    assert len(out["deepstack_features"]) == 1


def test_vision_meter_counts_the_layers_the_vit_actually_has():
    """Check the vision meter against the ViT's own parameter shapes.

    The meter is easy to get wrong in ways that still look plausible: counting
    the MLP as a SwiGLU when it is a two-layer GELU MLP, or sizing the merger
    from `hidden` instead of the widened `hidden * merge²`. Deriving the
    expectations from the instantiated module means the test disagrees with the
    meter whenever either mistake comes back.
    """
    from veomni.models.seed_omni.modules.qwen3vl.vision.accelerated.accelerated import VisionMeterMixin
    from veomni.models.seed_omni.modules.qwen3vl.vision.configuration import Qwen3VLVisionEncoderConfig
    from veomni.models.seed_omni.modules.qwen3vl.vision.modeling import Qwen3VLVisionEncoder

    config = Qwen3VLVisionEncoderConfig(ops_implementation=make_eager_ops_config())
    meter = VisionMeterMixin.__new__(VisionMeterMixin)
    meter.config = config

    vision = Qwen3VLVisionEncoder._from_config(config).visual
    block = vision.blocks[0]
    merger = vision.merger
    merge_area = int(config.vision_config.spatial_merge_size) ** 2

    def linear_params(layer):
        """Input × output of an `nn.Linear`, read off the weight tensor."""
        return layer.weight.numel()

    per_block = linear_params(block.mlp.linear_fc1) + linear_params(block.mlp.linear_fc2)
    for proj in (block.attn.qkv, block.attn.proj):
        per_block += linear_params(proj)
    merger_once = linear_params(merger.linear_fc1) + linear_params(merger.linear_fc2)

    hidden = config.vision_config.hidden_size
    depth = config.vision_config.depth
    head_dim = hidden // config.vision_config.num_heads
    patch_embed = linear_params(vision.patch_embed.proj)

    seqlens = [3, 5]
    raw_seqlens = [n * merge_area for n in seqlens]
    raw_tokens = sum(raw_seqlens)
    raw_sq = sum(n * n for n in raw_seqlens)
    merged_tokens = sum(seqlens)
    merger_count = 1 + len(config.vision_config.deepstack_visual_indexes)

    dense = 6 * per_block * depth * raw_tokens
    attn = 12 * raw_sq * head_dim * config.vision_config.num_heads * depth
    patch = 6 * patch_embed * raw_tokens
    mergers = 6 * merger_once * merged_tokens * merger_count
    assert meter.estimate_flops(seqlens) == pytest.approx((dense + attn + patch + mergers) / 1e12, rel=1e-9)

    # One DeepStack layer runs its own copy of the merger, so dropping it must
    # remove exactly one `merger_once` per merged token.
    no_deepstack = SimpleNamespace(**{**vars(config.vision_config), "deepstack_visual_indexes": []})
    meter.config = SimpleNamespace(vision_config=no_deepstack)
    without = (dense + attn + patch + 6 * merger_once * merged_tokens) / 1e12
    assert meter.estimate_flops(seqlens) == pytest.approx(without, rel=1e-9)


def test_vision_position_ids_layout():
    # grid (t=1, h=4, w=6), merge=2 -> llm grid (1, 2, 3); start offset 5.
    pos = qwen3vl_vision_position_ids(5, torch.tensor([1, 4, 6]), merge=2)
    assert pos.shape == (3, 6)
    # temporal: single frame -> all equal to start
    assert pos[0].tolist() == [5, 5, 5, 5, 5, 5]
    # height: 0,0,0,1,1,1 + start
    assert pos[1].tolist() == [5, 5, 5, 6, 6, 6]
    # width: 0,1,2,0,1,2 + start
    assert pos[2].tolist() == [5, 6, 7, 5, 6, 7]


def test_video_position_ids_restart_per_frame_like_hf():
    """A video is ``t`` single-frame grids: each frame restarts h/w at its own
    start and advances positions by ``max(h, w) // merge`` — matching
    ``Qwen3VLModel.get_rope_index``'s per-timestamp split."""
    grid = torch.tensor([4, 16, 16])  # 4 frames, 16x16, merge 2 -> 8x8 per frame
    per_frame = (16 // 2) * (16 // 2)
    pos = qwen3vl_vision_position_ids(2, grid, merge=2)
    assert pos.shape == (3, 4 * per_frame)

    # Each frame's temporal row is constant and equals that frame's start.
    assert [int(pos[0][i * per_frame].item()) for i in range(4)] == [2, 10, 18, 26]
    for frame in range(4):
        chunk = pos[0][frame * per_frame : (frame + 1) * per_frame]
        assert set(chunk.tolist()) == {2 + frame * 8}

    # h/w restart at each frame's start instead of running across the clip.
    for frame, start in enumerate([2, 10, 18, 26]):
        rows = pos[1][frame * per_frame : (frame + 1) * per_frame]
        assert int(rows.min().item()) == start
        assert int(rows.max().item()) == start + 7

    # The item advances by the summed per-frame steps, not just one frame's.
    assert qwen3vl_vision_position_advance(grid, 2) == 32
    # An image (t == 1) is unaffected by the per-frame split.
    image = torch.tensor([1, 16, 16])
    assert qwen3vl_vision_position_advance(image, 2) == 8
    # One frame of a t=1 image: the temporal row is the caller's start and the
    # spatial rows run over the *merged* grid (16x16 patches, merge 2 -> 8x8 = 64
    # tokens). Pinned as literals because comparing the function to itself would
    # pass for any implementation.
    pos = qwen3vl_vision_position_ids(2, image, merge=2)
    assert tuple(pos.shape) == (3, 64)
    assert set(pos[0].tolist()) == {2}, "temporal row is the start position for every token"
    assert pos[1].tolist() == [2 + i // 8 for i in range(64)], "h row runs 2..2+gh-1"
    assert pos[2].tolist() == [2 + i % 8 for _ in range(8) for i in range(8)], "w row cycles 2..2+gw-1"


def test_pack_leaves_deepstack_out_when_the_vision_config_disables_it():
    """``disable_deepstack`` empties every item's ``deepstack`` list. The packer must
    then pass neither a mask nor layer features, so the backbone does not try to
    ``torch.stack`` an empty list."""
    from veomni.models.seed_omni.modules.qwen3vl.llm.modeling import (
        pack_qwen3vl_conversations_for_forward as pack,
    )
    from veomni.models.seed_omni.utils.conversation import ConversationItem

    text = ConversationItem(type="text", value=torch.zeros(2, 64), role="user")
    image = ConversationItem(
        type="image", value=torch.zeros(4, 64), role="user", meta={"grid_thw": [1, 2, 2], "deepstack": []}
    )
    out = pack([[text, image]], torch.device("cpu"), 2)
    assert out["visual_pos_masks"] is None
    assert out["deepstack_visual_embeds"] is None

    # With layer features present, both come through.
    with_ds = ConversationItem(
        type="image",
        value=torch.zeros(4, 64),
        role="user",
        meta={"grid_thw": [1, 2, 2], "deepstack": [torch.zeros(4, 64), torch.zeros(4, 64)]},
    )
    out = pack([[text, with_ds]], torch.device("cpu"), 2)
    assert out["visual_pos_masks"] is not None
    assert len(out["deepstack_visual_embeds"]) == 2


def test_vision_preprocessor_injects_one_dummy_per_sample_and_tags_token_len():
    """A sample with no media gets its own ``is_dummy`` placeholder, tagged with
    the merged-token count ``dyn_bsz`` counts (one per sample, not one per batch)."""
    from veomni.models.seed_omni.modules.qwen3vl.vision.processing import Qwen3VLVisionPreprocessor
    from veomni.models.seed_omni.utils.conversation import (
        _IMG_TAG_KEY,
        _OMNI_TOKEN_LEN,
        ConversationItem,
        iter_desired_items,
    )

    vc = SimpleNamespace(spatial_merge_size=2, in_channels=3, temporal_patch_size=2, patch_size=16)
    config = SimpleNamespace(vision_config=vc)
    pre = Qwen3VLVisionPreprocessor(image_processor=lambda **_: {}, video_processor=lambda **_: {})
    pre.bind_dummy_inputs(config, dtype=torch.bfloat16)

    text_only = [[ConversationItem(type="text", value="a", role="user")] for _ in range(3)]
    pre.preprocess_conversations(text_only, inference=False)

    for sample in text_only:
        dummies = [it for it in iter_desired_items([sample], types=["image"], roles=["user"]) if it.is_dummy]
        assert len(dummies) == 1, "exactly one placeholder per sample"
        dummy = dummies[0]
        assert dummy.meta[_IMG_TAG_KEY] == "und"
        # grid [1, 2*merge, 2*merge] = [1, 4, 4] patches -> 16 // merge**2 merged tokens.
        assert dummy.meta[_OMNI_TOKEN_LEN] == 16 // vc.spatial_merge_size**2

    # Inference never injects: the FSM skips the edge instead.
    inference_only = [[ConversationItem(type="text", value="a", role="user")] for _ in range(2)]
    pre.preprocess_conversations(inference_only, inference=True)
    assert all(len(sample) == 1 for sample in inference_only)


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    TextEncoder = OMNI_MODEL_REGISTRY["qwen3vl_text_encoder"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()

    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=True))
    _save_fake_fast_tokenizer(save_as_omni(tmp_path, "qwen3vl_text_encoder", te))

    rcfg, te2 = load_from_omni(tmp_path, "qwen3vl_text_encoder")
    assert rcfg.model_type == "qwen3vl_text_encoder"
    assert isinstance(te2, TextEncoder)
    assert te2.config.vocab_size == 64
    assert te2.config.hidden_size == 16
