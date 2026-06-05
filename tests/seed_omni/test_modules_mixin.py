"""Smoke tests for SeedOmni V2 module mixins + per-module CheckpointCallback.

These tests exercise the mixin lifecycle (construct, save, reload) without
requiring real Janus weights — small synthetic configs only.  The intent
is to catch regressions in:

* :data:`OMNI_MODEL_REGISTRY` membership.
* OMNI registry ``from_pretrained`` round-trip.
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
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    OmniModule,
    OmniModuleCheckpointCallback,
)
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY


def _config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def _model_cls(model_type: str):
    return OMNI_MODEL_REGISTRY[model_type]()


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
    expected = {"text_encoder", "janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    assert expected.issubset(set(OMNI_MODEL_REGISTRY.valid_keys()))
    assert _model_cls("text_encoder").__name__ == "TextEncoder"
    assert _model_cls("janus_siglip").__name__ == "JanusSiglip"
    assert _model_cls("janus_vqvae").__name__ == "JanusVqvae"
    assert _model_cls("janus_llama").__name__ == "JanusLlama"
    assert _model_cls("janus_text_encoder").__name__ == "JanusTextEncoder"


def test_processor_registry_only_for_vision_modules():
    """janus_llama / text_encoder have no per-module asset."""
    assert set(OMNI_PROCESSOR_REGISTRY.valid_keys()) == {"janus_siglip", "janus_vqvae"}


def test_all_registered_classes_are_omnimodule_mixins():
    for name in OMNI_MODEL_REGISTRY.valid_keys():
        cls = OMNI_MODEL_REGISTRY[name]()
        assert issubclass(cls, OmniModule), f"{name} must inherit OmniModule"


# ── save / reload via OMNI registry ───────────────────────────────────────────


def test_text_encoder_save_reload_via_registry(tmp_path: Path):
    """Verifies config/model round-trip through OMNI registry classes."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")

    te = TextEncoder(TextEncoderConfig(vocab_size=128, hidden_size=64, tie_word_embeddings=True))
    te.save_pretrained(tmp_path)

    cfg = TextEncoderConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "text_encoder"

    te2 = TextEncoder.from_pretrained(tmp_path)
    assert isinstance(te2, TextEncoder)
    assert te2.config.vocab_size == 128


def test_janus_llama_save_reload_via_registry(tmp_path: Path):
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")

    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    jl.save_pretrained(tmp_path)

    cfg = JanusLlamaConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "janus_llama"

    jl2 = JanusLlama.from_pretrained(tmp_path)
    assert isinstance(jl2, JanusLlama)
    # embed_tokens dropped via Identity — reloaded module also has Identity.
    from torch.nn import Identity

    assert isinstance(jl2.language_model.get_input_embeddings(), Identity)


def test_janus_text_encoder_save_reload_via_registry(tmp_path: Path):
    JanusTextEncoder = _model_cls("janus_text_encoder")
    JanusTextEncoderConfig = _config_cls("janus_text_encoder")

    cfg = JanusTextEncoderConfig(
        vocab_size=128,
        hidden_size=64,
        tie_word_embeddings=True,
        begin_of_image_token_id=12345,
        end_of_image_token_id=67890,
    )
    jte = JanusTextEncoder(cfg)
    jte.save_pretrained(tmp_path)

    rcfg = JanusTextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "janus_text_encoder"
    assert rcfg.begin_of_image_token_id == 12345
    assert rcfg.end_of_image_token_id == 67890

    jte2 = JanusTextEncoder.from_pretrained(tmp_path)
    assert isinstance(jte2, JanusTextEncoder)
    assert jte2.config.vocab_size == 128


def test_janus_text_encoder_emit_methods_return_expected_shapes():
    """``emit_image_start`` / ``emit_image_end`` produce the boundary token."""
    JanusTextEncoder = _model_cls("janus_text_encoder")
    JanusTextEncoderConfig = _config_cls("janus_text_encoder")

    cfg = JanusTextEncoderConfig(vocab_size=128, hidden_size=16, tie_word_embeddings=True)
    jte = JanusTextEncoder(cfg)

    class _MockTokenizer:
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 3
        boi_token_id = 42
        eoi_token_id = 43
        image_token_id = 99

    jte.tokenizer = _MockTokenizer()

    # No batch_size hint in ctx → defaults to 1.
    out = jte.emit_image_start()
    assert out["input_ids"].tolist() == [[42]]
    assert out["inputs_embeds"].shape == (1, 1, 16)

    # batch_size inferred from ctx['input_ids'].
    ctx_in = torch.zeros(3, 5, dtype=torch.long)
    out = jte.emit_image_end(input_ids=ctx_in)
    assert out["input_ids"].shape == (3, 1)
    assert (out["input_ids"] == 43).all()
    assert out["inputs_embeds"].shape == (3, 1, 16)


# ── Mixin call-site contracts (loss key, shapes) ──────────────────────────────


def test_text_encoder_decode_returns_single_loss_key():
    """V2 single-loss protocol: ``post_forward`` maps ``loss`` → ``_loss``."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    labels = torch.randint(0, 64, (2, 4))
    out = te.decode(hidden_states=h, labels=labels)
    assert out.loss is not None and out.loss.dim() == 0
    graph_out = te.post_forward("decode", **out.to_dict())
    assert "_loss" in graph_out and graph_out["_loss"].dim() == 0
    assert "lm_loss" not in graph_out


def test_text_encoder_encode_uses_last_token_with_kv_cache():
    """With KV cache, ``encode`` embeds only ``input_ids[:, -1:]``."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    full = torch.tensor([[1, 2, 42]], dtype=torch.long)
    out = te.encode(input_ids=full, past_key_values=())
    assert out["inputs_embeds"].shape == (1, 1, 16)
    assert torch.equal(out["inputs_embeds"], te.embed_tokens(torch.tensor([[42]])))


def test_text_encoder_decode_inference_returns_logits_only():
    """Base ``TextEncoder.decode`` without labels returns logits only."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    out = te.decode(hidden_states=h)
    assert out.logits is not None and out.logits.shape == (2, 4, 64)
    assert out.loss is None


def test_janus_vqvae_decode_training_loss():
    """Training ``decode``: hidden_states + labels → scalar loss."""
    JanusVqvae = _model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))

    h = torch.randn(1, 4, 64)  # janus_llama hidden_size
    labels = torch.randint(0, 64, (1, 4))
    out = jv.decode(hidden_states=h, labels=labels)
    assert set(out.keys()) == {"loss"}
    assert out["loss"].dim() == 0


def test_janus_siglip_forward_passes_dummy_pixel_path():
    """Empty pixel_values → empty dict (text-only batch)."""
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    out = js(pixel_values=None)
    assert out == {}


def test_janus_llama_forward_requires_inputs_embeds():
    """The backbone must receive `inputs_embeds` produced upstream."""
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    with pytest.raises(ValueError, match="inputs_embeds"):
        jl(inputs_embeds=None)


# ── Per-module CheckpointCallback ─────────────────────────────────────────────


def test_callback_writes_module_subfolder(tmp_path: Path):
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_encoder", processor=None, is_rank_0=True)
    cb.save(str(tmp_path))

    out = tmp_path / "text_encoder"
    assert out.is_dir()
    files = sorted(p.name for p in out.iterdir())
    assert "config.json" in files
    assert "model.safetensors" in files
    # No processor → no preprocessor_config.json should appear.
    assert "preprocessor_config.json" not in files


def test_callback_with_processor_writes_asset(tmp_path: Path):
    """Verifies the per-module asset (vision processor) is saved alongside."""
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))

    proc_cls = OMNI_PROCESSOR_REGISTRY["janus_siglip"]()
    proc = proc_cls()  # default-constructed; tests only that save/load round-trips

    cb = OmniModuleCheckpointCallback(module=js, module_name="janus_siglip", processor=proc, is_rank_0=True)
    cb.save(str(tmp_path))

    out = tmp_path / "janus_siglip"
    files = sorted(p.name for p in out.iterdir())
    assert "config.json" in files
    assert "model.safetensors" in files
    assert "preprocessor_config.json" in files


def test_callback_round_trip_via_registry(tmp_path: Path):
    """Save with callback → reload via OMNI registry class → still a TextEncoder."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")

    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_encoder", is_rank_0=True)
    cb.save(str(tmp_path))

    reloaded = TextEncoder.from_pretrained(str(tmp_path / "text_encoder"))
    assert isinstance(reloaded, TextEncoder)
    assert reloaded.config.vocab_size == 64
    assert reloaded.config.hidden_size == 16


def test_callback_non_rank_0_is_noop(tmp_path: Path):
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=32, hidden_size=8))
    cb = OmniModuleCheckpointCallback(module=te, module_name="text_encoder", is_rank_0=False)
    cb.save(str(tmp_path))
    # Non-rank-0 must not write anything.
    assert not (tmp_path / "text_encoder").exists()


# ── _no_split_modules preservation ────────────────────────────────────────────


def test_fsdp_no_split_modules_preserved():
    """The FSDP unit boundary list must survive the mixin reshuffle."""
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")

    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    assert "LlamaDecoderLayer" in (jl._no_split_modules or set())

    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    assert "JanusVisionEncoderLayer" in (js._no_split_modules or set())


# ── janus_1.3b/{train,infer_*}.yaml smoke load ────────────────────────────────


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "janus_1.3b"


def test_janus_train_yaml_loads_with_v2_module_names():
    """The shipped training YAML must round-trip through ``OmniConfig.from_paths``."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    cfg = OmniConfig.from_paths(
        model_path="",
        train_yaml_path=_janus_cfg_dir() / "train.yaml",
    )

    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    assert cfg.modules["janus_siglip"]["model"]["weights_path"] == "janus_siglip"
    # Sanity: every training-graph edge is declared in the edges pool.
    edge_names = set(cfg.edges)
    for e in cfg.training_edges:
        assert e in edge_names
    # Inference-only nodes / edges live in the pool but are NOT in training_edges.
    assert "emit_image_start" in cfg.nodes
    assert "emit_image_end" in cfg.nodes
    assert "emit_start_to_janus_llama" in cfg.edges
    assert "emit_end_to_janus_llama" in cfg.edges
    assert "emit_start_to_janus_llama" not in cfg.training_edges


@pytest.mark.parametrize("infer_yaml", ["infer_interleave.yaml", "infer_gen.yaml", "infer_und.yaml"])
def test_janus_train_plus_infer_merges_generation_graph(infer_yaml: str):
    """Two-file load: training vocabulary + inference scenario merge cleanly."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    cfg = OmniConfig.from_paths(
        model_path="",
        train_yaml_path=_janus_cfg_dir() / "train.yaml",
        infer_yaml_path=_janus_cfg_dir() / infer_yaml,
    )
    # Training vocabulary still present.
    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    # Generation graph painted on top.
    assert cfg.has_generation_graph()
    assert "states" in cfg.generation_graph
    # `done` is framework-injected — must NOT be authored in YAML.
    assert "done" not in cfg.generation_graph["states"], (
        f"`done` should be auto-injected by GenerationGraph, not declared in {infer_yaml}. "
        "Remove the `done:` block from the inference YAML."
    )
    assert "done_state" not in cfg.generation_graph, (
        "`done_state` is no longer configurable — the terminal state name is hardcoded to 'done'."
    )
    # At least one transition must funnel into the built-in `done` state — otherwise
    # the FSM has no way to terminate via condition.
    assert any(
        t.get("next_state") == "done"
        for state in cfg.generation_graph["states"].values()
        for t in state.get("transitions", [])
    ), f"{infer_yaml} has no transition to `done` — the FSM cannot terminate."
    # Each inference body should reference only edges that exist in the pool.
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert e in cfg.edges, f"state '{state_name}' body edge '{e}' not in pool"


def test_from_paths_deep_merges_infer_module_overrides():
    """Infer YAML ``modules:`` patches train YAML per module without restating ``weights_path``."""
    import yaml

    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    train_yaml = _janus_cfg_dir() / "train.yaml"
    infer_yaml = _janus_cfg_dir() / "infer_gen.yaml"
    with open(infer_yaml, encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)
    infer_cfg["modules"] = {
        "janus_siglip": {
            "train": {
                "accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2", "full_shard": False}},
            },
            "model": {"model_config": {"freeze": True}},
        }
    }

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(infer_cfg, tmp)
        patched_infer = tmp.name

    try:
        cfg = OmniConfig.from_paths(
            model_path="/tmp/janus",
            train_yaml_path=train_yaml,
            infer_yaml_path=patched_infer,
        )
    finally:
        Path(patched_infer).unlink(missing_ok=True)

    siglip = cfg.modules["janus_siglip"]
    assert siglip["model"]["weights_path"] == "janus_siglip"
    assert "model_path" not in siglip["model"]
    assert siglip["train"]["accelerator"]["fsdp_config"]["fsdp_mode"] == "fsdp2"
    assert siglip["train"]["accelerator"]["fsdp_config"]["full_shard"] is False
    assert siglip["model"]["model_config"]["freeze"] is True
    # Unaffected modules keep the train YAML block only.
    assert "train" not in cfg.modules["janus_llama"]


def test_build_module_args_resolves_relative_weights_path():
    """Relative ``model.weights_path`` joins the launcher ``model.model_path``."""
    from veomni.arguments import DataArguments
    from veomni.trainer.omni_trainer import OmniModelArguments, VeOmniOmniArguments, build_module_args

    root = "seed_omni/janus_1.3b"
    global_args = VeOmniOmniArguments(
        model=OmniModelArguments(model_path=root),
        data=DataArguments(train_path="/tmp/unused"),
    )
    mod_cfg = {"model": {"weights_path": "janus_siglip"}}
    module_args = build_module_args(global_args, mod_cfg)
    assert module_args.model.model_path == f"{root}/janus_siglip"
    assert module_args.model.config_path == f"{root}/janus_siglip"


def test_from_paths_keeps_relative_module_weights_in_omni_config():
    """``OmniConfig`` stores raw module blocks; path join happens in ``build_module_args``."""
    from veomni.models.seed_omni.configuration_seed_omni import OmniConfig

    root = "seed_omni/janus_1.3b"
    cfg = OmniConfig.from_paths(
        model_path=root,
        train_yaml_path=_janus_cfg_dir() / "train.yaml",
        infer_yaml_path=_janus_cfg_dir() / "infer_gen.yaml",
    )

    assert cfg.modules["janus_siglip"]["model"]["weights_path"] == "janus_siglip"
    assert cfg.modules["janus_text_encoder"]["model"]["weights_path"] == "janus_text_encoder"
    assert "model_path" not in cfg.modules["janus_siglip"]["model"]
    assert cfg.has_generation_graph()
    assert cfg.generation_graph["initial"] == "prompt_encode"
