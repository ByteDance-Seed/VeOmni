"""Smoke tests for SeedOmni V2 module mixins + per-module CheckpointCallback.

These tests exercise the mixin lifecycle (construct, save, reload) without
requiring real Janus weights — small synthetic configs only.  The intent
is to catch regressions in:

* :data:`MODULE_MIXIN_REGISTRY` membership.
* HuggingFace ``AutoConfig`` / ``AutoModel`` registration round-trip.
* :class:`OmniModule` mixin protocol (``forward`` / ``encode`` / ``decode``
  return shapes / loss key contract).
* :class:`OmniModuleCheckpointCallback` writes the expected layout and the
  module reloads successfully.

Step 2 (``OmniTrainer`` integration) will add an end-to-end test that
combines these modules with the V2 graph runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from veomni.models.seed_omni import (
    MODULE_MIXIN_REGISTRY,
    MODULE_PROCESSOR_REGISTRY,
    OmniModule,
    OmniModuleCheckpointCallback,
)
from veomni.models.seed_omni.modules import (
    JanusLlama,
    JanusLlamaConfig,
    JanusSiglip,
    JanusSiglipConfig,
    JanusTextEmbed,
    JanusTextEmbedConfig,
    JanusVqvae,
    JanusVqvaeConfig,
    TextEmbed,
    TextEmbedConfig,
)


# ── Tiny configs used everywhere ──────────────────────────────────────────────


def _tiny_text_cfg() -> dict:
    """Small LlamaConfig dict that fits in <100k params."""
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )


def _tiny_vision_cfg() -> dict:
    return dict(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=64,
        patch_size=16,
        projection_dim=64,  # output of aligner = janus_llama hidden_size
    )


def _tiny_vq_cfg() -> dict:
    """Tiny VQVAE config with shapes lined up to match the tiny LLM (hidden=64).

    * ``embed_dim``              — codebook entry dim (input to aligner.fc1).
    * ``projection_dim``         — output of ``generation_aligner``; must
                                    equal ``janus_llama.hidden_size`` so the
                                    aligner can feed the backbone.
    * ``image_token_embed_dim``  — input to ``generation_head.proj_out``;
                                    must equal ``janus_llama.hidden_size``
                                    so the head can read the LLM's hidden.
    * ``num_embeddings``         — codebook size.  The CE in
                                    :meth:`JanusVqvae.decode` projects to
                                    this many logits.
    """
    return dict(
        embed_dim=8,
        num_embeddings=64,
        num_hidden_layers=2,
        projection_dim=64,
        image_token_embed_dim=64,
        in_channels=3,
        out_channels=3,
    )


# ── Registry assertions ───────────────────────────────────────────────────────


def test_mixin_registry_contains_all_v2_modules():
    expected = {"text_embed", "janus_siglip", "janus_vqvae", "janus_llama", "janus_text_embed"}
    assert expected.issubset(set(MODULE_MIXIN_REGISTRY))
    assert MODULE_MIXIN_REGISTRY["text_embed"] is TextEmbed
    assert MODULE_MIXIN_REGISTRY["janus_siglip"] is JanusSiglip
    assert MODULE_MIXIN_REGISTRY["janus_vqvae"] is JanusVqvae
    assert MODULE_MIXIN_REGISTRY["janus_llama"] is JanusLlama
    assert MODULE_MIXIN_REGISTRY["janus_text_embed"] is JanusTextEmbed


def test_processor_registry_only_for_vision_modules():
    """janus_llama / text_embed have no per-module asset."""
    assert set(MODULE_PROCESSOR_REGISTRY) == {"janus_siglip", "janus_vqvae"}


def test_all_registered_classes_are_omnimodule_mixins():
    for name, cls in MODULE_MIXIN_REGISTRY.items():
        assert issubclass(cls, OmniModule), f"{name} must inherit OmniModule"


# ── HF AutoConfig / AutoModel round-trip ──────────────────────────────────────


def test_text_embed_save_reload_via_hf_auto(tmp_path: Path):
    """Verifies AutoConfig.register / AutoModel.register wired correctly."""
    from transformers import AutoConfig, AutoModel

    te = TextEmbed(TextEmbedConfig(vocab_size=128, hidden_size=64, tie_word_embeddings=True))
    te.save_pretrained(tmp_path)

    cfg = AutoConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "text_embed"

    te2 = AutoModel.from_pretrained(tmp_path)
    assert isinstance(te2, TextEmbed)
    assert te2.config.vocab_size == 128


def test_janus_llama_save_reload_via_hf_auto(tmp_path: Path):
    from transformers import AutoConfig, AutoModel

    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    jl.save_pretrained(tmp_path)

    cfg = AutoConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "janus_llama"
    # HF Janus 1.3B uses 100581 for the image token (understanding+generation).
    assert cfg.image_token_id == 100581

    jl2 = AutoModel.from_pretrained(tmp_path)
    assert isinstance(jl2, JanusLlama)
    # embed_tokens dropped via Identity — reloaded module also has Identity.
    from torch.nn import Identity

    assert isinstance(jl2.language_model.get_input_embeddings(), Identity)


def test_janus_text_embed_save_reload_via_hf_auto(tmp_path: Path):
    from transformers import AutoConfig, AutoModel

    cfg = JanusTextEmbedConfig(
        vocab_size=128,
        hidden_size=64,
        tie_word_embeddings=True,
        begin_of_image_token_id=12345,
        end_of_image_token_id=67890,
    )
    jte = JanusTextEmbed(cfg)
    jte.save_pretrained(tmp_path)

    rcfg = AutoConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "janus_text_embed"
    assert rcfg.begin_of_image_token_id == 12345
    assert rcfg.end_of_image_token_id == 67890

    jte2 = AutoModel.from_pretrained(tmp_path)
    assert isinstance(jte2, JanusTextEmbed)
    assert jte2.config.vocab_size == 128


def test_janus_text_embed_emit_methods_return_expected_shapes():
    """``emit_image_start`` / ``emit_image_end`` produce the boundary token."""
    cfg = JanusTextEmbedConfig(
        vocab_size=128,
        hidden_size=16,
        tie_word_embeddings=True,
        begin_of_image_token_id=42,
        end_of_image_token_id=43,
    )
    jte = JanusTextEmbed(cfg)

    # No batch_size hint in ctx → defaults to 1.
    out = jte.emit_image_start()
    assert out["input_ids"].tolist() == [[42]]
    assert out["last_token_id"].tolist() == [42]
    assert out["inputs_embeds"].shape == (1, 1, 16)

    # batch_size inferred from ctx['input_ids'].
    ctx_in = torch.zeros(3, 5, dtype=torch.long)
    out = jte.emit_image_end(input_ids=ctx_in)
    assert out["input_ids"].shape == (3, 1)
    assert (out["input_ids"] == 43).all()
    assert out["last_token_id"].shape == (3,)
    assert out["inputs_embeds"].shape == (3, 1, 16)


# ── Mixin call-site contracts (loss key, shapes) ──────────────────────────────


def test_text_embed_decode_returns_single_loss_key():
    """V2 single-loss protocol: only the ``_loss`` key is allowed."""
    te = TextEmbed(TextEmbedConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    labels = torch.randint(0, 64, (2, 4))
    out = te.decode(hidden_states=h, labels=labels)
    assert "_loss" in out and out["_loss"].dim() == 0
    # The pre-V2 ``lm_loss`` alias must not appear.
    assert "lm_loss" not in out


def test_text_embed_encode_inference_loop_keys():
    """Inference path returns the exact keys the FSM body expects."""
    te = TextEmbed(TextEmbedConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    out = te.decode(hidden_states=h)
    assert set(out.keys()) >= {"logits", "last_token_id", "input_ids"}
    assert out["last_token_id"].shape == (2,)
    assert out["input_ids"].shape == (2, 1)


def test_janus_vqvae_decode_branches():
    """Three input-driven dispatch paths in JanusVqvae.decode."""
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg(), freeze_vqvae=True))

    # Path 1: training (hidden + gt_token_ids → _loss).
    h = torch.randn(1, 4, 64)  # janus_llama hidden_size
    gt = torch.randint(0, 64, (1, 4))
    out = jv.decode(hidden_states=h, gt_token_ids=gt)
    assert set(out.keys()) == {"_loss"}
    assert out["_loss"].dim() == 0

    # Path 2: inference sample (hidden → vq_token_id + embed).
    out = jv.decode(hidden_states=h)
    assert set(out.keys()) == {"vq_token_id", "embed"}
    assert out["vq_token_id"].shape == (1,)
    assert out["embed"].shape == (1, 1, 64)

    # Path 3: pre-sampled lookup (token_id → embed).
    tok = torch.tensor([5])
    out = jv.decode(token_id=tok)
    assert set(out.keys()) == {"embed"}
    assert out["embed"].shape == (1, 1, 64)


def test_janus_siglip_forward_passes_dummy_pixel_path():
    """Empty pixel_values → empty dict (text-only batch)."""
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    out = js(pixel_values=None)
    assert out == {}


def test_janus_llama_forward_requires_inputs_embeds():
    """The backbone must receive `inputs_embeds` produced upstream."""
    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    with pytest.raises(ValueError, match="inputs_embeds"):
        jl(inputs_embeds=None)


# ── Per-module CheckpointCallback ─────────────────────────────────────────────


def test_callback_writes_module_subfolder(tmp_path: Path):
    te = TextEmbed(TextEmbedConfig(vocab_size=64, hidden_size=16))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_embed", processor=None, is_rank_0=True)
    cb.save(str(tmp_path))

    out = tmp_path / "text_embed"
    assert out.is_dir()
    files = sorted(p.name for p in out.iterdir())
    assert "config.json" in files
    assert "model.safetensors" in files
    # No processor → no preprocessor_config.json should appear.
    assert "preprocessor_config.json" not in files


def test_callback_with_processor_writes_asset(tmp_path: Path):
    """Verifies the per-module asset (vision processor) is saved alongside."""
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))

    proc_cls = MODULE_PROCESSOR_REGISTRY["janus_siglip"]
    proc = proc_cls()  # default-constructed; tests only that save/load round-trips

    cb = OmniModuleCheckpointCallback(module=js, module_name="janus_siglip", processor=proc, is_rank_0=True)
    cb.save(str(tmp_path))

    out = tmp_path / "janus_siglip"
    files = sorted(p.name for p in out.iterdir())
    assert "config.json" in files
    assert "model.safetensors" in files
    assert "preprocessor_config.json" in files


def test_callback_round_trip_via_auto(tmp_path: Path):
    """Save with callback → reload via HF AutoModel → still a TextEmbed."""
    from transformers import AutoModel

    te = TextEmbed(TextEmbedConfig(vocab_size=64, hidden_size=16))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_embed", is_rank_0=True)
    cb.save(str(tmp_path))

    reloaded = AutoModel.from_pretrained(str(tmp_path / "text_embed"))
    assert isinstance(reloaded, TextEmbed)
    assert reloaded.config.vocab_size == 64
    assert reloaded.config.hidden_size == 16


def test_callback_non_rank_0_is_noop(tmp_path: Path):
    te = TextEmbed(TextEmbedConfig(vocab_size=32, hidden_size=8))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_embed", is_rank_0=False)
    cb.save(str(tmp_path))
    # Non-rank-0 must not write anything.
    assert not (tmp_path / "text_embed").exists()


# ── _no_split_modules preservation ────────────────────────────────────────────


def test_fsdp_no_split_modules_preserved():
    """The FSDP unit boundary list must survive the mixin reshuffle."""
    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    assert "LlamaDecoderLayer" in (jl._no_split_modules or set())

    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    assert "JanusVisionEncoderLayer" in (js._no_split_modules or set())


# ── janus_1.3b/{train,infer_*}.yaml smoke load ────────────────────────────────


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "janus_1.3b"


def test_janus_train_yaml_loads_with_v2_module_names():
    """The shipped training YAML must round-trip through ``OmniConfig.from_yamls``."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    cfg = OmniConfig.from_yamls(_janus_cfg_dir() / "train_joint.yaml")

    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "text_embed"}
    assert cfg.tokenizer_path is not None
    # Sanity: every training-graph edge is declared in the edges pool.
    edge_names = set(cfg.edges)
    for e in cfg.training_edges:
        assert e in edge_names
    # Inference-only nodes / edges live in the pool but are NOT in training_edges.
    assert "emit_image_start" in cfg.nodes
    assert "emit_image_end" in cfg.nodes
    assert "emit_start_to_ar" in cfg.edges
    assert "emit_end_to_ar" in cfg.edges
    assert "emit_start_to_ar" not in cfg.training_edges


@pytest.mark.parametrize("infer_yaml", ["infer_interleave.yaml", "infer_t2i.yaml", "infer_understanding.yaml"])
def test_janus_train_plus_infer_merges_generation_graph(infer_yaml: str):
    """Two-file load: training vocabulary + inference scenario merge cleanly."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    cfg = OmniConfig.from_yamls(
        _janus_cfg_dir() / "train_joint.yaml",
        _janus_cfg_dir() / infer_yaml,
    )
    # Training vocabulary still present.
    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "text_embed"}
    # Generation graph painted on top.
    assert cfg.has_generation_graph()
    assert "states" in cfg.generation_graph
    assert "done" in cfg.generation_graph["states"]
    # Each inference body should reference only edges that exist in the pool.
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert e in cfg.edges, f"state '{state_name}' body edge '{e}' not in pool"


def test_from_yamls_deep_merge_overrides_specific_keys(tmp_path: Path):
    """Override semantics: dict merge, scalar replace, ``None`` no-op."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    base = tmp_path / "base.yaml"
    over = tmp_path / "over.yaml"
    base.write_text(
        "tokenizer_path: /a\n"
        "modules:\n"
        "  foo: {weights_path: /x, micro_batch_size: 4}\n"
        "  bar: {weights_path: /y}\n"
        "nodes:\n"
        "  enc: {module: foo}\n"
        "edges:\n"
        "  e1: {from: enc, to: end}\n"
        "training_graph:\n"
        "  edges: [e1]\n"
    )
    over.write_text(
        "tokenizer_path: /b\n"
        "modules:\n"
        "  foo: {micro_batch_size: 8}\n"  # only override one knob; weights_path kept
        "generation_graph:\n"
        "  initial: s\n"
        "  states: {s: {body: [], token_length: {type: variable}, transitions: []}}\n"
    )
    cfg = OmniConfig.from_yamls(base, over)

    assert cfg.tokenizer_path == "/b"  # scalar replaced
    assert cfg.modules["foo"]["weights_path"] == "/x"  # base survived
    assert cfg.modules["foo"]["micro_batch_size"] == 8  # override applied
    assert cfg.modules["bar"]["weights_path"] == "/y"  # untouched module preserved
    assert cfg.has_generation_graph()  # painted on
    assert cfg.training_edges == ["e1"]  # not stomped
