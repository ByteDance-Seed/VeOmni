"""Smoke tests for Qwen3-VL SeedOmni V2 modules (registry + M-RoPE helper)."""

from pathlib import Path
from types import SimpleNamespace

import torch

from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY
from veomni.models.seed_omni.modules.qwen3vl.llm.modulemixin import Qwen3VLLlmModuleMixin


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
    import veomni.models.seed_omni.modules.qwen3vl.vision.modeling as vision_modeling

    monkeypatch.setattr(vision_modeling, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=False))

    Enc = OMNI_MODEL_REGISTRY["qwen3vl_vision"]()
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
    enc = Enc(Cfg(vision_config=vc)).eval()

    g = [1, 4, 4]  # two dummy placeholders
    pixel_values = torch.zeros(2 * 1 * 4 * 4, 3 * 2 * 16 * 16)
    grid_thw = torch.tensor([g, g], dtype=torch.long)
    vit_metadata = enc._build_vit_metadata([g, g])

    real_emb, real_deep = enc._encode(pixel_values, grid_thw, vit_metadata)
    out = enc.forward(pixel_values=pixel_values, image_grid_thw=grid_thw, vit_metadata=vit_metadata, is_dummy=True)

    assert out["image_embeds"] is not None
    assert out["image_embeds"].shape == real_emb.shape
    assert out["image_embeds"].abs().sum().item() == 0.0
    assert len(out["deepstack_features"]) == len(real_deep)
    assert out["deepstack_features"][0].shape == real_deep[0].shape


def test_vision_dummy_forward_skips_vit_in_eval_even_under_fsdp(monkeypatch):
    """Inference (eval) needs no gradient anchor, so the dummy vision forward
    fabricates zeros even with FSDP enabled — the real ViT must not run."""
    import veomni.models.seed_omni.modules.qwen3vl.vision.modeling as vision_modeling

    monkeypatch.setattr(vision_modeling, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=True))

    Enc = OMNI_MODEL_REGISTRY["qwen3vl_vision"]()
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
    enc = Enc(Cfg(vision_config=vc)).eval()

    def _boom(*_a, **_k):
        raise AssertionError("ViT must not run for a dummy in eval mode")

    monkeypatch.setattr(enc, "_encode", _boom)
    g = [1, 4, 4]
    out = enc.forward(
        pixel_values=torch.zeros(1 * 4 * 4, 3 * 2 * 16 * 16),
        image_grid_thw=torch.tensor([g], dtype=torch.long),
        vit_metadata=enc._build_vit_metadata([g]),
        is_dummy=True,
    )
    assert out["image_embeds"].abs().sum().item() == 0.0
    assert len(out["deepstack_features"]) == 1


def test_vision_position_ids_layout():
    # grid (t=1, h=4, w=6), merge=2 -> llm grid (1, 2, 3); start offset 5.
    pos = Qwen3VLLlmModuleMixin._vision_position_ids(5, torch.tensor([1, 4, 6]), merge=2)
    assert pos.shape == (3, 6)
    # temporal: single frame -> all equal to start
    assert pos[0].tolist() == [5, 5, 5, 5, 5, 5]
    # height: 0,0,0,1,1,1 + start
    assert pos[1].tolist() == [5, 5, 5, 6, 6, 6]
    # width: 0,1,2,0,1,2 + start
    assert pos[2].tolist() == [5, 6, 7, 5, 6, 7]


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    TextEncoder = OMNI_MODEL_REGISTRY["qwen3vl_text_encoder"]()
    TextEncoderConfig = OMNI_CONFIG_REGISTRY["qwen3vl_text_encoder"]()

    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=True))
    te.save_pretrained(tmp_path)

    rcfg = TextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "qwen3vl_text_encoder"

    te2 = TextEncoder.from_pretrained(tmp_path)
    assert isinstance(te2, TextEncoder)
    assert te2.config.vocab_size == 64
    assert te2.config.hidden_size == 16
