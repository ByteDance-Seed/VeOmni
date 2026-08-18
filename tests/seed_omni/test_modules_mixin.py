"""Smoke tests for SeedOmni V2 ``*BaseMixin`` classes and checkpoint layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from veomni.arguments import OmniArguments, OmniDataArguments
from veomni.models.seed_omni import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
    BaseMixin,
)
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY
from veomni.models.seed_omni.utils.conversation import ConversationItem
from veomni.omni_arguments import OmniModelRuntimeArguments, build_omni_model_runtime


def _config_cls(model_type: str):
    return OMNI_CONFIG_REGISTRY[model_type]()


def _model_cls(model_type: str):
    return OMNI_MODEL_REGISTRY[model_type]()


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

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<s>": 2,
        "</s>": 3,
        "<begin_of_image>": 4,
        "<end_of_image>": 5,
        "a": 6,
        "b": 7,
    }
    backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        # Janus-specific chat-template markers — harmless extra attrs for other families.
        boi_token="<begin_of_image>",
        eoi_token="<end_of_image>",
    )
    tokenizer.save_pretrained(module_path)


def _patch_parallel_state(monkeypatch, **attrs):
    import veomni.distributed.parallel_state as ps_utils

    monkeypatch.setattr(ps_utils, "get_parallel_state", lambda: SimpleNamespace(**attrs))
    return ps_utils


def _patch_local_loss_reducer(monkeypatch):
    """Keep module-only tests out of the distributed loss collective."""
    import veomni.distributed.sequence_parallel as sp_utils

    def _local_reduce(loss, num_valid_tokens, group=None):
        del group
        return torch.where(num_valid_tokens > 0, loss, torch.zeros_like(loss))

    monkeypatch.setattr(sp_utils, "reduce_sequence_parallel_loss", _local_reduce)
    return sp_utils


def _omni_base_args(*, model_path: str = ""):
    return OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path or ".",
            model_config={
                "modules": str(_janus_cfg_dir() / "modules_train.yaml"),
                "train_graph": str(_janus_cfg_dir() / "graph_train.yaml"),
            },
        ),
        data=OmniDataArguments(train_path=""),
    )._to_module_global_args()


def _load_omni_config(
    *,
    model_path: str = "",
    modules_path: Path,
    train_graph_path: Path | None = None,
    infer_graph_path: Path | None = None,
    generation_kwargs: dict | None = None,
) -> OmniConfig:
    model_path = model_path or "."
    model_config = {"modules": str(modules_path)}
    if train_graph_path is not None:
        model_config["train_graph"] = str(train_graph_path)
    base = OmniArguments(
        model=OmniModelRuntimeArguments(
            model_path=model_path,
            model_config=model_config,
        ),
        data=OmniDataArguments(train_path="."),
    )._to_module_global_args()
    return build_omni_model_runtime(
        global_args=base,
        model_path=model_path,
        train_modules=str(modules_path),
        train_graph=str(train_graph_path) if train_graph_path else None,
        infer_graph=str(infer_graph_path) if infer_graph_path else None,
        generation_kwargs=generation_kwargs,
    ).to_hf_config()


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


def test_processor_registry_only_for_modules_with_processor_assets():
    """janus_llama / text_encoder have no per-module asset."""
    assert set(OMNI_PROCESSOR_REGISTRY.valid_keys()) == {
        "bagel_siglip_navit",
        "bagel_vae",
        "janus_siglip",
        "janus_vqvae",
        "qwen3vl_vision",
    }


def _accelerated_model_cls(model_type: str):
    return OMNI_ACCELERATED_MODEL_REGISTRY[model_type]()


def test_all_registered_classes_are_module_mixins():
    from veomni.models.seed_omni.omni_pretrained_model import OmniPreTrainedModel

    accelerated_keys = set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys())
    for name in OMNI_MODEL_REGISTRY.valid_keys():
        native_cls = OMNI_MODEL_REGISTRY[name]()
        assert name in accelerated_keys, f"{name} must register OMNI_ACCELERATED_MODEL_REGISTRY"
        assert issubclass(native_cls, OmniPreTrainedModel), f"{name} native must inherit OmniPreTrainedModel"
        assert not issubclass(native_cls, BaseMixin), f"{name} native must not inherit BaseMixin"
        accelerated_cls = OMNI_ACCELERATED_MODEL_REGISTRY[name]()
        assert issubclass(accelerated_cls, BaseMixin), f"{name} accelerated must inherit BaseMixin"


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
    _save_fake_fast_tokenizer(tmp_path)

    rcfg = JanusTextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "janus_text_encoder"
    assert rcfg.begin_of_image_token_id == 12345
    assert rcfg.end_of_image_token_id == 67890

    jte2 = JanusTextEncoder.from_pretrained(tmp_path)
    assert isinstance(jte2, JanusTextEncoder)
    assert jte2.config.vocab_size == 128


def test_janus_text_encoder_emit_image_start_replaces_output_tail():
    JanusTextEncoder = _accelerated_model_cls("janus_text_encoder")
    JanusTextEncoderConfig = _config_cls("janus_text_encoder")

    cfg = JanusTextEncoderConfig(vocab_size=128, hidden_size=16, tie_word_embeddings=True)
    jte = JanusTextEncoder(cfg)
    # emit_image_start reads the boi/eoi ids off the chat template; stub it (no tokenizer).
    jte._chat_template = SimpleNamespace(boi_token_id=42, eoi_token_id=43)

    conv = [ConversationItem(type="output", value=torch.randn(1, 1, 16), role="assistant")]
    out = jte.emit_image_start(conversation_list=conv, generation_kwargs={})
    assert out["conversation_list"][-1].value.shape == (1, 16)


# ── Mixin call-site contracts (loss key, shapes) ──────────────────────────────


def test_text_encoder_decode_returns_single_loss_key():
    """V2 single-loss protocol: ``post_forward`` maps ``loss`` → ``_loss``."""
    TextEncoder = _accelerated_model_cls("text_encoder")
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


def test_text_encoder_decode_dispatches_pre_shifted_loss(monkeypatch):
    """Untied, bias-free heads delegate loss policy to the configured ops wrapper.

    Asserted on the ACCELERATED class: it overrides ``decode``, so the fused
    dispatch has to live there to be what training actually runs.
    """
    import veomni.models.seed_omni.modules.base.text_encoder.accelerated as text_encoder_accelerated

    reduction_group = object()
    captured = {}
    monkeypatch.setattr(
        text_encoder_accelerated,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=reduction_group),
    )

    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

    def fake_loss_function(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"].sum() * 0, None, None

    te.loss_function = fake_loss_function
    hidden_states = torch.randn(5, 16)
    shift_labels = torch.tensor([1, 2, -100, 3, -100])
    out = te.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    assert out["loss"].dim() == 0
    assert out["logits"] is None
    assert captured["hidden_states"] is hidden_states
    assert captured["shift_labels"] is shift_labels
    assert captured["labels"] is shift_labels
    assert captured["weights"] is te.lm_head.weight
    assert captured["loss_reduction_group"] is reduction_group


def test_text_encoder_tied_head_uses_explicit_eager_loss(monkeypatch):
    """Tied heads must not enter a globally selected fused-linear loss."""
    _patch_local_loss_reducer(monkeypatch)
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=True))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("tied head must not call the globally selected loss function")

    te.loss_function = unexpected_loss_function
    hidden_states = torch.randn(5, 16)
    shift_labels = torch.tensor([1, 2, -100, 3, -100])
    out = te.decode(hidden_states=hidden_states, shift_labels=shift_labels)

    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (5, 64)


def test_text_encoder_decode_all_masked_span_scores_zero(monkeypatch):
    """An unsupervised span scores 0.0, never NaN (constraint 7b clamped denominator).

    The fused kernels normalize by their own supervised-token count with no way to
    clamp it, so a span whose labels are all ``-100`` must NOT reach them.
    """
    _patch_local_loss_reducer(monkeypatch)
    TextEncoder = _accelerated_model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    te = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))

    def unexpected_loss_function(**kwargs):
        raise AssertionError("an all-masked span must not enter the fused loss (0/0)")

    te.loss_function = unexpected_loss_function
    out = te.decode(hidden_states=torch.randn(5, 16), shift_labels=torch.full((5,), -100))

    assert float(out["loss"].detach()) == 0.0
    # Still graph-connected, so the head's grads (zeros) exist on every rank.
    assert out["loss"].requires_grad


def test_janus_vqvae_decode_training_loss(monkeypatch):
    """Training ``decode``: hidden_states + labels → scalar loss."""
    _patch_local_loss_reducer(monkeypatch)

    JanusVqvae = _model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))

    h = torch.randn(1, 4, 64)  # janus_llama hidden_size
    labels = torch.randint(0, 64, (1, 4))
    out = jv.decode(hidden_states=h, labels=labels)
    assert set(out.keys()) == {"loss"}
    assert out["loss"].dim() == 0


def test_janus_vqvae_dummy_decode_keeps_generation_head_in_graph(monkeypatch):
    """FSDP2 regression: under FSDP the dummy decode path must route through
    ``generation_head`` so its grad/reduce_scatter fires on every rank (ranks
    with no assistant image would otherwise skip it and dead-lock NCCL)."""
    _patch_local_loss_reducer(monkeypatch)
    _patch_parallel_state(monkeypatch, fsdp_enabled=True, fsdp_group=None)

    JanusVqvae = _accelerated_model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))

    # generation_head must be trainable (only the inner vqmodel is frozen).
    jv.freeze_model()
    assert all(p.requires_grad for p in jv.generation_head.parameters())

    h = torch.randn(1, 4, 64, requires_grad=True)
    labels = torch.full((1, 4), -100)
    out = jv.decode(hidden_states=h, labels=labels, is_dummy=True)

    # Zero loss contribution, but the head's params must be in the graph.
    assert out["loss"].dim() == 0
    assert out["loss"].detach().item() == 0.0

    out["loss"].backward()
    head_grads = [p.grad for p in jv.generation_head.parameters() if p.grad is not None]
    assert head_grads, "dummy decode must produce a gradient path through generation_head"
    assert all(torch.count_nonzero(grad) == 0 for grad in head_grads)


def test_janus_vqvae_dummy_encode_emits_real_shaped_zeros_without_fsdp(monkeypatch):
    """Off-FSDP the dummy encode skips the codec forward but must still emit
    zeros shaped exactly like a real encode (same batch + token count, no
    ``None``), so the pre/post hooks never special-case the dummy."""
    _patch_parallel_state(monkeypatch, fsdp_enabled=False)

    JanusVqvae = _accelerated_model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))

    pixel_values = torch.zeros(3, 3, 32, 32)  # batch of 3 dummy placeholders
    real = jv._encode_pixels(pixel_values)
    out = jv.encode(pixel_values=pixel_values, is_dummy=True)

    assert out["image_embeds"] is not None and out["vq_token_ids"] is not None
    assert out["image_embeds"].shape == real["image_embeds"].shape
    assert out["vq_token_ids"].shape == real["vq_token_ids"].shape
    assert out["vq_token_ids"].dtype == real["vq_token_ids"].dtype
    assert out["image_embeds"].abs().sum().item() == 0.0


def test_janus_vqvae_dummy_encode_skips_codec_in_eval_even_under_fsdp(monkeypatch):
    """Inference (eval) needs no gradient anchor, so the dummy encode fabricates
    zeros even with FSDP enabled — the real codec must not run."""
    _patch_parallel_state(monkeypatch, fsdp_enabled=True)

    JanusVqvae = _accelerated_model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg())).eval()

    def _boom(*_a, **_k):
        raise AssertionError("codec must not run for a dummy in eval mode")

    monkeypatch.setattr(jv, "_encode_pixels", _boom)
    out = jv.encode(pixel_values=torch.zeros(2, 3, 32, 32), is_dummy=True)
    assert out["image_embeds"].abs().sum().item() == 0.0


def test_janus_vqvae_dummy_decode_keeps_generation_head_in_graph_without_fsdp(monkeypatch):
    """The non-distributed dummy path follows the same decode contract: ignored
    labels contribute 0.0 while ``generation_head`` remains in the graph."""
    _patch_local_loss_reducer(monkeypatch)
    _patch_parallel_state(monkeypatch, fsdp_enabled=False, fsdp_group=None)

    JanusVqvae = _accelerated_model_cls("janus_vqvae")
    JanusVqvaeConfig = _config_cls("janus_vqvae")
    jv = JanusVqvae(JanusVqvaeConfig(vq_config=_tiny_vq_cfg()))
    jv.freeze_model()

    h = torch.randn(1, 4, 64, requires_grad=True)
    labels = torch.full((1, 4), -100)
    out = jv.decode(hidden_states=h, labels=labels, is_dummy=True)

    assert out["loss"].detach().item() == 0.0
    out["loss"].backward()
    head_grads = [p.grad for p in jv.generation_head.parameters() if p.grad is not None]
    assert head_grads, "dummy decode must preserve the uniform generation_head path"
    assert all(torch.count_nonzero(grad) == 0 for grad in head_grads)


def test_janus_siglip_forward_returns_image_embeds():
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    pixels = torch.randn(1, 3, 64, 64)
    out = js(pixel_values=pixels)
    assert "image_embeds" in out and out["image_embeds"].dim() >= 2


def test_janus_siglip_dummy_forward_emits_real_shaped_zeros_without_fsdp(monkeypatch):
    """Off-FSDP the dummy forward skips the ViT but must still emit zeros shaped
    exactly like a real encode (no ``None``), so forward_post never branches."""
    import veomni.models.seed_omni.modules.janus.siglip.accelerated as siglip_accelerated

    monkeypatch.setattr(siglip_accelerated, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=False))

    JanusSiglipAccelerated = _accelerated_model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglipAccelerated(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    pixels = torch.zeros(3, 3, 64, 64)
    real = js._encode_pixel_values(pixels)
    out = js.forward(pixel_values=pixels, is_dummy=True)

    assert out["image_embeds"] is not None
    assert out["image_embeds"].shape == real.shape
    assert out["image_embeds"].abs().sum().item() == 0.0


def test_janus_siglip_dummy_forward_skips_vit_in_eval_even_under_fsdp(monkeypatch):
    """Inference (eval) needs no gradient anchor, so the dummy forward fabricates
    zeros even with FSDP enabled — the real ViT must not run."""
    import veomni.models.seed_omni.modules.janus.siglip.accelerated as siglip_accelerated

    monkeypatch.setattr(siglip_accelerated, "get_parallel_state", lambda: SimpleNamespace(fsdp_enabled=True))

    JanusSiglipAccelerated = _accelerated_model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")
    js = JanusSiglipAccelerated(JanusSiglipConfig(vision_config=_tiny_vision_cfg())).eval()

    def _boom(*_a, **_k):
        raise AssertionError("ViT must not run for a dummy in eval mode")

    monkeypatch.setattr(js, "_encode_pixel_values", _boom)
    out = js.forward(pixel_values=torch.zeros(3, 3, 64, 64), is_dummy=True)
    assert out["image_embeds"].abs().sum().item() == 0.0


def test_janus_llama_forward_returns_hidden_states():
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    embeds = torch.randn(1, 4, 64)
    out = jl(inputs_embeds=embeds)
    assert out["hidden_states"].shape == (1, 4, 64)


def test_accelerated_forward_delegates_to_native_modeling():
    JanusLlamaAccelerated = _accelerated_model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    model = JanusLlamaAccelerated(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    embeds = torch.randn(1, 4, 64)
    out = model.forward(inputs_embeds=embeds)
    assert out["hidden_states"].shape == (1, 4, 64)


def test_training_module_mixin_forward_requires_override_without_native_impl():
    from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin

    class CodecOnlyAccelerated(TrainingModuleMixin, BaseMixin):
        pass

    with pytest.raises(NotImplementedError, match="forward\\(\\*\\*kwargs\\) is not implemented"):
        CodecOnlyAccelerated().forward()


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
    _save_fake_fast_tokenizer(tmp_path)

    rcfg = Qwen3TextEncoderConfig.from_pretrained(tmp_path)
    assert rcfg.model_type == "qwen3_text_encoder"

    te2 = Qwen3TextEncoder.from_pretrained(tmp_path)
    assert isinstance(te2, Qwen3TextEncoder)
    assert te2.config.vocab_size == 128


# ── _no_split_modules preservation ────────────────────────────────────────────


def test_fsdp_no_split_modules_preserved():
    """The FSDP unit boundary list must survive the mixin reshuffle."""
    TextEncoder = _model_cls("text_encoder")
    TextEncoderConfig = _config_cls("text_encoder")
    JanusLlama = _model_cls("janus_llama")
    JanusLlamaConfig = _config_cls("janus_llama")
    JanusSiglip = _model_cls("janus_siglip")
    JanusSiglipConfig = _config_cls("janus_siglip")

    text_encoder = TextEncoder(TextEncoderConfig(vocab_size=64, hidden_size=16, tie_word_embeddings=False))
    assert text_encoder._no_split_modules == {"Embedding"}

    jl = JanusLlama(JanusLlamaConfig(text_config=_tiny_text_cfg()))
    assert "LlamaDecoderLayer" in (jl._no_split_modules or set())

    js = JanusSiglip(JanusSiglipConfig(vision_config=_tiny_vision_cfg()))
    assert "JanusVisionEncoderLayer" in (js._no_split_modules or set())


# ── janus_1.3b/{train,infer_*}.yaml smoke load ────────────────────────────────


def _janus_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Janus" / "janus_1.3b"


def test_janus_train_yaml_loads_with_v2_module_names():
    cfg = _load_omni_config(
        modules_path=_janus_cfg_dir() / "modules_train.yaml",
        train_graph_path=_janus_cfg_dir() / "graph_train.yaml",
        infer_graph_path=_janus_cfg_dir() / "graph_infer_gen.yaml",
    )

    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    assert cfg.modules["janus_siglip"]["subfolder"] == "janus_siglip"
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


@pytest.mark.parametrize(
    "infer_graph", ["graph_infer_interleave.yaml", "graph_infer_gen.yaml", "graph_infer_und.yaml"]
)
def test_janus_train_plus_infer_merges_generation_graph(infer_graph: str):
    cfg = _load_omni_config(
        modules_path=_janus_cfg_dir() / "modules_train.yaml",
        train_graph_path=_janus_cfg_dir() / "graph_train.yaml",
        infer_graph_path=_janus_cfg_dir() / infer_graph,
    )
    # Training vocabulary still present.
    assert set(cfg.modules) == {"janus_siglip", "janus_vqvae", "janus_llama", "janus_text_encoder"}
    # Generation graph painted on top.
    assert cfg.generation_graph is not None
    assert "states" in cfg.generation_graph
    # `done` is framework-injected — must NOT be authored in YAML.
    assert "done" not in cfg.generation_graph["states"], (
        f"`done` should be auto-injected by GenerationGraph, not declared in {infer_graph}. "
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
    ), f"{infer_graph} has no transition to `done` — the FSM cannot terminate."
    # Each inference body is a list of inline `{from, to}` edge dicts.
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert isinstance(e, dict) and "from" in e and "to" in e, (
                f"state '{state_name}' body item must be a `{{from, to}}` dict: {e!r}"
            )


def test_init_applies_eager_defaults_for_inference():
    """Inference applies all-eager defaults on top of ``model.model_config.modules``."""
    from veomni.omni_arguments.arguments_types import build_module_runtime_args

    runtime_args = build_module_runtime_args(
        global_args=_omni_base_args(model_path="/tmp/janus"),
        model_path="/tmp/janus",
        modules={
            "janus_siglip": {
                "model_path": "janus_siglip",
                "accelerator": {"fsdp_config": {"fsdp_mode": "fsdp2", "full_shard": False}},
            },
            "janus_llama": {"model_path": "janus_llama"},
        },
        for_inference=True,
    )

    siglip = runtime_args["janus_siglip"]
    assert siglip.model_path == "/tmp/janus/janus_siglip"
    assert siglip.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert runtime_args["janus_llama"].accelerator.fsdp_config.fsdp_mode == "eager"


def test_init_resolves_relative_module_paths():
    root = "seed_omni/janus_1.3b"
    cfg = _load_omni_config(
        model_path=root,
        modules_path=_janus_cfg_dir() / "modules_train.yaml",
        train_graph_path=_janus_cfg_dir() / "graph_train.yaml",
        infer_graph_path=_janus_cfg_dir() / "graph_infer_gen.yaml",
    )

    assert cfg.modules["janus_siglip"]["subfolder"] == "janus_siglip"
    assert cfg.modules["janus_text_encoder"]["subfolder"] == "janus_text_encoder"
    assert cfg.generation_graph is not None
    assert cfg.generation_graph["initial"] == "prompt_encode"

    from veomni.omni_arguments.arguments_types import build_module_runtime_args

    runtime_args = build_module_runtime_args(
        global_args=_omni_base_args(model_path=root),
        model_path=root,
        modules=str(_janus_cfg_dir() / "modules_infer_fsdp.yaml"),
        for_inference=True,
    )
    assert runtime_args["janus_siglip"].model_path == f"{root}/janus_siglip"
    assert runtime_args["janus_text_encoder"].model_path == f"{root}/janus_text_encoder"


def _qwen3_cfg_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "seed_omni" / "Qwen" / "qwen3_0.6b"


def test_qwen3_train_yaml_loads_with_v2_module_names():
    cfg = _load_omni_config(
        modules_path=_qwen3_cfg_dir() / "modules_train.yaml",
        train_graph_path=_qwen3_cfg_dir() / "graph_train.yaml",
        infer_graph_path=_qwen3_cfg_dir() / "graph_infer.yaml",
    )

    assert set(cfg.modules) == {"qwen3_text_encoder", "qwen3_llm"}
    assert cfg.modules["qwen3_text_encoder"]["subfolder"] == "qwen3_text_encoder"
    assert isinstance(cfg.training_graph, list) and cfg.training_graph
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "qwen3_text_encoder.encode" in endpoints and "qwen3_llm" in endpoints


def test_qwen3_train_plus_infer_merges_generation_graph():
    cfg = _load_omni_config(
        modules_path=_qwen3_cfg_dir() / "modules_train.yaml",
        train_graph_path=_qwen3_cfg_dir() / "graph_train.yaml",
        infer_graph_path=_qwen3_cfg_dir() / "graph_infer.yaml",
    )
    assert set(cfg.modules) == {"qwen3_text_encoder", "qwen3_llm"}
    assert cfg.generation_graph is not None
    assert cfg.generation_graph["initial"] == "text_ar"
    assert "done" not in cfg.generation_graph["states"]
