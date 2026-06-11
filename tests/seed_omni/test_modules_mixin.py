"""Smoke tests for SeedOmni V2 ``*ModuleMixin`` classes and checkpoint layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from veomni.arguments.arguments_types import DataArguments
from veomni.models.seed_omni import (
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    ModuleMixin,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.conversation import ConversationItem
from veomni.models.seed_omni.generation_graph import FSM_SIGNAL_KEY
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.trainer.omni_trainer import OmniModelArguments, VeOmniOmniArguments


def _config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def _model_cls(model_type: str):
    return OMNI_MODEL_REGISTRY[model_type]()


def _load_omni_config(
    *,
    model_path: str = "",
    train_yaml_path: Path,
    infer_yaml_path: Path | None = None,
) -> OmniConfig:
    model_args = OmniModelArguments(model_path=model_path, config_path=model_path or ".")
    global_args = VeOmniOmniArguments(
        model=model_args,
        data=DataArguments(train_path=""),
    )._to_base_args()
    return OmniConfig._init(
        global_args=global_args,
        model_path=global_args.model.model_path,
        train_yaml_path=train_yaml_path,
        infer_yaml_path=infer_yaml_path,
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


def _tiny_qwen3_cfg() -> dict:
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
    )


def _tiny_bagel_qwen2_cfg() -> dict:
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
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
    expected = {
        "text_encoder",
        "janus_siglip",
        "janus_vqvae",
        "janus_llama",
        "janus_text_encoder",
        "qwen3_llm",
        "qwen3_text_encoder",
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert expected.issubset(set(OMNI_MODEL_REGISTRY.valid_keys()))
    assert _model_cls("text_encoder").__name__ == "TextEncoder"
    assert _model_cls("janus_siglip").__name__ == "JanusSiglip"
    assert _model_cls("janus_vqvae").__name__ == "JanusVqvae"
    assert _model_cls("janus_llama").__name__ == "JanusLlama"
    assert _model_cls("janus_text_encoder").__name__ == "JanusTextEncoder"
    assert _model_cls("qwen3_llm").__name__ == "Qwen3Llm"
    assert _model_cls("qwen3_text_encoder").__name__ == "Qwen3TextEncoder"
    assert _model_cls("bagel_text_encoder").__name__ == "BagelTextEncoder"
    assert _model_cls("bagel_siglip_navit").__name__ == "BagelSiglipNavit"
    assert _model_cls("bagel_qwen2_mot").__name__ == "BagelQwen2MoT"
    assert _model_cls("bagel_flow_connector").__name__ == "BagelFlowConnector"
    assert _model_cls("bagel_vae").__name__ == "BagelVAE"


def test_processor_registry_only_for_vision_modules():
    """janus_llama / text_encoder have no per-module asset."""
    assert set(OMNI_PROCESSOR_REGISTRY.valid_keys()) == {"janus_siglip", "janus_vqvae"}


def test_all_registered_classes_are_module_mixins():
    for name in OMNI_MODEL_REGISTRY.valid_keys():
        cls = OMNI_MODEL_REGISTRY[name]()
        assert issubclass(cls, ModuleMixin), f"{name} must inherit ModuleMixin"


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


def test_janus_text_encoder_emit_image_start_replaces_output_tail():
    JanusTextEncoder = _model_cls("janus_text_encoder")
    JanusTextEncoderConfig = _config_cls("janus_text_encoder")

    cfg = JanusTextEncoderConfig(vocab_size=128, hidden_size=16, tie_word_embeddings=True)
    jte = JanusTextEncoder(cfg)
    jte._boi_token_id = 42
    jte._eoi_token_id = 43

    conv = [ConversationItem(type="output", value=torch.randn(1, 1, 16), role="assistant")]
    out = jte.emit_image_start(conversation_list=conv, generation_kwargs={})
    assert out["conversation_list"][-1].value.shape == (1, 16)


# ── Mixin call-site contracts (loss key, shapes) ──────────────────────────────


def test_text_encoder_decode_returns_single_loss_key():
    """V2 single-loss protocol: ``post_forward`` maps ``loss`` → ``_loss``."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    labels = torch.randint(0, 64, (2, 4))
    out = te.decode(hidden_states=h, labels=labels)
    assert out["loss"] is not None and out["loss"].dim() == 0
    graph_out = te.post_forward("decode", **out)
    assert "_loss" in graph_out and graph_out["_loss"].dim() == 0
    assert "lm_loss" not in graph_out


def test_text_encoder_decode_inference_returns_logits_only():
    """Base ``TextEncoder.decode`` without labels returns logits only."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16))
    h = torch.randn(2, 4, 16)
    out = te.decode(hidden_states=h)
    assert out["logits"] is not None and out["logits"].shape == (2, 4, 64)
    assert out["loss"] is None


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


def test_janus_vqvae_dummy_decode_keeps_generation_head_in_graph():
    """FSDP2 regression: the dummy decode path must route through
    ``generation_head`` so its grad/reduce_scatter fires on every rank (ranks
    with no assistant image would otherwise skip it and dead-lock NCCL)."""
    JanusVqvae = _model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))

    # generation_head must be trainable (only the inner vqmodel is frozen).
    jv.freeze_model()
    assert all(p.requires_grad for p in jv.generation_head.parameters())

    h = torch.randn(1, 4, 64, requires_grad=True)
    labels = torch.randint(0, 64, (1, 4))
    out = jv.decode(hidden_states=h, labels=labels, is_dummy=True)

    # Zero loss contribution, but the head's params must be in the graph.
    assert out["loss"].dim() == 0
    assert out["loss"].detach().item() == 0.0

    out["loss"].backward()
    head_grads = [p.grad for p in jv.generation_head.parameters() if p.grad is not None]
    assert head_grads, "dummy decode must produce a gradient path through generation_head"


def test_janus_siglip_forward_returns_image_embeds():
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    pixels = torch.randn(1, 3, 64, 64)
    out = js(pixel_values=pixels)
    assert "image_embeds" in out and out["image_embeds"].dim() >= 2


def test_janus_llama_forward_returns_hidden_states():
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    embeds = torch.randn(1, 4, 64)
    out = jl(inputs_embeds=embeds)
    assert out["hidden_states"].shape == (1, 4, 64)


def test_qwen3_llm_save_reload_via_registry(tmp_path: Path):
    Qwen3Llm = _model_cls("qwen3_llm")
    Qwen3LlmConfig = _config_cls("qwen3_llm")

    llm = Qwen3Llm(Qwen3LlmConfig(text_config=_tiny_qwen3_cfg()))
    llm.save_pretrained(tmp_path)

    cfg = Qwen3LlmConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "qwen3_llm"

    llm2 = Qwen3Llm.from_pretrained(tmp_path)
    assert isinstance(llm2, Qwen3Llm)
    from torch.nn import Identity

    assert isinstance(llm2.language_model.get_input_embeddings(), Identity)


def test_qwen3_text_encoder_save_reload_via_registry(tmp_path: Path):
    Qwen3TextEncoder = _model_cls("qwen3_text_encoder")
    Qwen3TextEncoderConfig = _config_cls("qwen3_text_encoder")

    te = Qwen3TextEncoder(Qwen3TextEncoderConfig(vocab_size=128, hidden_size=64, tie_word_embeddings=True))
    te.save_pretrained(tmp_path)

    rcfg = Qwen3TextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "qwen3_text_encoder"

    te2 = Qwen3TextEncoder.from_pretrained(tmp_path)
    assert isinstance(te2, Qwen3TextEncoder)
    assert te2.config.vocab_size == 128


def test_bagel_qwen2_mot_gen_mode_requires_token_indexes():
    BagelQwen2MoT = _model_cls("bagel_qwen2_mot")
    BagelQwen2MoTConfig = _config_cls("bagel_qwen2_mot")
    model = BagelQwen2MoT(BagelQwen2MoTConfig(**_tiny_bagel_qwen2_cfg()))

    with pytest.raises(ValueError, match="mode='gen' requires"):
        model._forward_packed_inference(
            packed_query_sequence=torch.randn(1, 64),
            query_lens=torch.tensor([1], dtype=torch.int32),
            packed_query_position_ids=torch.tensor([0], dtype=torch.long),
            packed_query_indexes=torch.tensor([0], dtype=torch.long),
            mode="gen",
        )


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
    cfg = _load_omni_config(train_yaml_path=_janus_cfg_dir() / "train.yaml")

    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    assert cfg.modules["janus_siglip"]["model"]["model_path"] == "janus_siglip"
    # training_graph is a flat list of `{from, to}` edges; endpoints are
    # self-describing `module[.method]` strings.
    assert isinstance(cfg.training_graph, list) and cfg.training_graph
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "janus_siglip" in endpoints
    assert "janus_vqvae.encode" in endpoints
    assert "janus_text_encoder.encode" in endpoints
    assert "end" in endpoints
    # Inference-only call-sites (emit_image_*) are NOT in the training graph.
    assert not any("emit_image" in e["from"] or "emit_image" in e["to"] for e in cfg.training_graph)


@pytest.mark.parametrize("infer_yaml", ["infer_interleave.yaml", "infer_gen.yaml", "infer_und.yaml"])
def test_janus_train_plus_infer_merges_generation_graph(infer_yaml: str):
    cfg = _load_omni_config(
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
    # Each inference body is a list of inline `{from, to}` edge dicts.
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert isinstance(e, dict) and "from" in e and "to" in e, (
                f"state '{state_name}' body item must be a `{{from, to}}` dict: {e!r}"
            )


def test_init_deep_merges_infer_module_overrides():
    """Infer YAML ``modules:`` patches train YAML per module."""
    import tempfile

    import yaml

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

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(infer_cfg, tmp)
        patched_infer = tmp.name

    try:
        cfg = _load_omni_config(
            model_path="/tmp/janus",
            train_yaml_path=train_yaml,
            infer_yaml_path=Path(patched_infer),
        )
    finally:
        Path(patched_infer).unlink(missing_ok=True)

    siglip = cfg.modules["janus_siglip"]
    assert siglip["model"]["model_path"] == "/tmp/janus/janus_siglip"
    assert siglip["train"]["accelerator"]["fsdp_config"]["fsdp_mode"] == "fsdp2"
    assert siglip["train"]["accelerator"]["fsdp_config"]["full_shard"] is False
    assert siglip["model"]["model_config"]["freeze"] is True
    llama_train = cfg.modules["janus_llama"]["train"]["accelerator"]["fsdp_config"]
    assert llama_train["fsdp_mode"] == "eager"


def test_init_resolves_relative_module_paths():
    root = "seed_omni/janus_1.3b"
    cfg = _load_omni_config(
        model_path=root,
        train_yaml_path=_janus_cfg_dir() / "train.yaml",
        infer_yaml_path=_janus_cfg_dir() / "infer_gen.yaml",
    )

    assert cfg.modules["janus_siglip"]["model"]["model_path"] == f"{root}/janus_siglip"
    assert cfg.modules["janus_text_encoder"]["model"]["model_path"] == f"{root}/janus_text_encoder"
    assert cfg.has_generation_graph()
    assert cfg.generation_graph["initial"] == "prompt_encode"


def _qwen3_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "qwen3_0.6b"


def test_qwen3_train_yaml_loads_with_v2_module_names():
    cfg = _load_omni_config(train_yaml_path=_qwen3_cfg_dir() / "train.yaml")

    assert set(cfg.modules) == {"qwen3_text_encoder", "qwen3_llm"}
    assert cfg.modules["qwen3_text_encoder"]["model"]["model_path"] == "qwen3_text_encoder"
    assert isinstance(cfg.training_graph, list) and cfg.training_graph
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "qwen3_text_encoder.encode" in endpoints and "qwen3_llm" in endpoints


def test_qwen3_train_plus_infer_merges_generation_graph():
    cfg = _load_omni_config(
        train_yaml_path=_qwen3_cfg_dir() / "train.yaml",
        infer_yaml_path=_qwen3_cfg_dir() / "infer_text.yaml",
    )
    assert set(cfg.modules) == {"qwen3_text_encoder", "qwen3_llm"}
    assert cfg.has_generation_graph()
    assert cfg.generation_graph["initial"] == "text_ar"
    assert "done" not in cfg.generation_graph["states"]


def _bagel_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "bagel_7b_mot"


def test_bagel_train_yaml_loads_with_v2_module_names():
    cfg = _load_omni_config(train_yaml_path=_bagel_cfg_dir() / "train.yaml")

    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert isinstance(cfg.training_graph, list) and cfg.training_graph
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "bagel_text_encoder.encode" in endpoints
    assert "bagel_siglip_navit" in endpoints
    assert "bagel_vae.encode" in endpoints
    assert "bagel_flow_connector.embed_latent" in endpoints
    assert "bagel_text_encoder.decode" in endpoints
    assert "bagel_flow_connector.decode_velocity" in endpoints
    assert "end" in endpoints


@pytest.mark.parametrize("infer_yaml", ["infer_und.yaml", "infer_gen.yaml", "infer_interleave.yaml"])
def test_bagel_train_plus_infer_merges_generation_graph(infer_yaml: str):
    cfg = _load_omni_config(
        train_yaml_path=_bagel_cfg_dir() / "train.yaml",
        infer_yaml_path=_bagel_cfg_dir() / infer_yaml,
    )
    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert cfg.has_generation_graph()
    assert cfg.generation_graph["initial"] == "prompt_encode"
    assert "done" not in cfg.generation_graph["states"]
    assert any(
        t.get("next_state") == "done"
        for state in cfg.generation_graph["states"].values()
        for t in state.get("transitions", [])
    ), f"{infer_yaml} has no transition to `done`."
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert isinstance(e, dict) and "from" in e and "to" in e, (
                f"state '{state_name}' body item must be a `{{from, to}}` dict: {e!r}"
            )


class _BagelInterleaveTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    _token_ids = {
        "<|im_start|>": 1,
        "<|vision_start|>": 5,
        "<|vision_end|>": 6,
        "prompt": 7,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._token_ids.get(part, 8) for part in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {1, 2, 5, 6} if skip_special_tokens else set()
        return " ".join(str(token_id) for token_id in token_ids if token_id not in special)


def test_bagel_raw_image_preprocessing_builds_official_metadata():
    BagelSiglip = _model_cls("bagel_siglip_navit")
    BagelSiglipConfig = _config_cls("bagel_siglip_navit")
    siglip = BagelSiglip(
        BagelSiglipConfig(
            hidden_size=8,
            output_size=8,
            image_size=28,
            min_image_size=14,
            max_pixels=28 * 14,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=14,
            vit_max_num_patch_per_side=2,
        )
    )
    image_item = ConversationItem(
        type="image",
        value=Image.new("RGB", (20, 10), color=(255, 0, 0)),
        role="user",
    )
    packed_pixels, position_ids, vit_token_lens = siglip._prepare_image_item(image_item)

    assert packed_pixels.shape == (2, 14 * 14 * 3)
    assert position_ids.tolist() == [0, 1]
    assert vit_token_lens.tolist() == [2]
    assert image_item.meta["preprocessed_image_size"] == [28, 14]
    assert image_item.meta["image_text_indexes"].tolist() == [0, 3]
    assert image_item.meta["vit_token_indexes"].tolist() == [1, 2]
    assert image_item.meta["sequence_indexes"].tolist() == [0, 1, 2, 3]
    assert image_item.meta["key_value_lens_after"].tolist() == [4]
    assert image_item.meta["rope_after"].tolist() == [1]

    BagelTextEncoder = _model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = _config_cls("bagel_text_encoder")
    text_encoder = BagelTextEncoder(BagelTextEncoderConfig(vocab_size=16, hidden_size=8))
    text_encoder.tokenizer = _BagelInterleaveTokenizer()
    image_item.value = torch.zeros(2, 8)
    image_item.meta["image_embeds"] = image_item.value
    image_item.meta["image_embeds_ready"] = True
    text_item = ConversationItem(type="text", value="prompt", role="user")
    text_encoder.generate(conversation_list=[image_item, text_item])

    assert text_item.meta["key_value_lens_before"].tolist() == [4]
    assert text_item.meta["position_ids"].tolist() == [1, 2, 3]
    assert text_item.meta["sequence_indexes"].tolist() == [4, 5, 6]


class _NoopBagelSiglip(nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}


class _InterleaveBagelQwen(nn.Module):
    def generate(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        active_image = next(
            (
                item
                for item in conversation_list
                if item.meta.get("bagel_role") == "image_gen_latent" and not item.meta.get("decoded_image_ready")
            ),
            None,
        )
        if active_image is not None and active_image.meta.get("flow_packed_sequence_ready"):
            active_image.value = torch.zeros(2, 8)
            active_image.meta["flow_hidden_ready"] = True
            return {"conversation_list": conversation_list, "bagel_last_hidden_state": active_image.value}

        tail = conversation_list[-1]
        assert tail.type == "output"
        hidden = torch.zeros(1, 8)
        if any(item.meta.get("decoded_image_ready") for item in conversation_list):
            hidden[0, 1] = 1.0  # eos
        else:
            hidden[0, 0] = 1.0  # vision_start
        tail.value = hidden
        return {"conversation_list": conversation_list, "bagel_last_hidden_state": hidden}


class _InterleaveBagelFlow(nn.Module):
    def embed_latent(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = next(
            (item for item in conversation_list if item.meta.get("bagel_role") == "image_gen_latent"),
            None,
        )
        if item is None:
            return {"conversation_list": conversation_list}
        item.value = torch.zeros(2, 8)
        item.meta["flow_packed_sequence_ready"] = True
        return {"conversation_list": conversation_list}

    def decode_velocity(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = next(item for item in conversation_list if item.meta.get("bagel_role") == "image_gen_latent")
        item.value = torch.zeros(2, 8)
        item.meta.pop("flow_packed_sequence_ready", None)
        item.meta.pop("flow_hidden_ready", None)
        return {"conversation_list": conversation_list, FSM_SIGNAL_KEY: "image_complete"}


class _InterleaveBagelVAE(nn.Module):
    def encode(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        return {"conversation_list": conversation_list}

    def decode(self, conversation_list: list[ConversationItem] | None = None, **kwargs):
        del kwargs
        assert conversation_list is not None
        item = next(item for item in conversation_list if item.meta.get("bagel_role") == "image_gen_latent")
        item.value = "decoded-image"
        item.meta["decoded_image_ready"] = True
        return {
            "conversation_list": conversation_list,
            "generated": {"type": "image", "value": item.value, "meta": {}},
        }


def test_bagel_interleave_image_branch_signal_smoke():
    BagelTextEncoder = _model_cls("bagel_text_encoder")
    BagelTextEncoderConfig = _config_cls("bagel_text_encoder")
    text_encoder = BagelTextEncoder(BagelTextEncoderConfig(vocab_size=16, hidden_size=8))
    text_encoder.tokenizer = _BagelInterleaveTokenizer()
    with torch.no_grad():
        text_encoder.lm_head.weight.zero_()
        text_encoder.lm_head.weight[5, 0] = 1.0  # <|vision_start|>
        text_encoder.lm_head.weight[2, 1] = 1.0  # eos

    cfg = _load_omni_config(
        train_yaml_path=_bagel_cfg_dir() / "train.yaml",
        infer_yaml_path=_bagel_cfg_dir() / "infer_interleave.yaml",
    )
    model = OmniModel(
        cfg,
        {
            "bagel_text_encoder": text_encoder,
            "bagel_siglip_navit": _NoopBagelSiglip(),
            "bagel_qwen2_mot": _InterleaveBagelQwen(),
            "bagel_flow_connector": _InterleaveBagelFlow(),
            "bagel_vae": _InterleaveBagelVAE(),
        },
    ).eval()
    trace: list[str] = []
    ctx = model.generate(
        {"conversation_list": [ConversationItem(type="text", value="prompt", role="user")]},
        trace=trace,
        generation_kwargs={
            "max_new_tokens": 8,
            "do_sample": False,
            "image_height": 64,
            "image_width": 64,
        },
    )

    assert any("transition: prompt_encode -> image_flow" in entry for entry in trace)
    assert any("transition: image_flow -> image_decode" in entry for entry in trace)
    assert any("transition: image_decode -> text_ar" in entry for entry in trace)
    assert any("transition: text_ar -> done" in entry for entry in trace)
    assert any(item["type"] == "image" for item in model.generated)
    assert ctx["conversation_list"][-1].type == "text"
