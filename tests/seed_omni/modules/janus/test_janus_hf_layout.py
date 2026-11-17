"""Running the ``janus`` SeedOmni family straight off its upstream HF checkpoint.

A tiny ``JanusForConditionalGeneration`` checkpoint stands in for the upstream
one. What this file pins is the layout contract: every HF key routes onto exactly
one module and back, each module config comes from the upstream config, and
``OmniModel.from_pretrained`` reads an HF root as if it were a split checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import JanusConfig, JanusVisionConfig, JanusVQVAEConfig, LlamaConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.janus.hf_layout import JANUS_HF_LAYOUT
from veomni.models.seed_omni.utils.hf_layout import read_hf_source, resolve_omni_checkpoint_root


SIGLIP = "janus_siglip"
VQVAE = "janus_vqvae"
TEXT_ENCODER = "janus_text_encoder"
LLAMA = "janus_llama"

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _janus_config(*, tied: bool = False) -> JanusConfig:
    """A Janus config small enough to build on CPU.

    Janus-1.3B ships untied, so that is the default here.
    """
    text_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
    )
    vision_config = JanusVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
        projection_dim=16,
    )
    vq_config = JanusVQVAEConfig(
        embed_dim=8,
        num_embeddings=16,
        projection_dim=16,
        depth=1,
        image_token_embed_dim=16,
    )
    config = JanusConfig(text_config=text_config, vision_config=vision_config, vq_config=vq_config)
    config.tie_word_embeddings = tied
    return config


def _source_tensors(config: JanusConfig) -> dict[str, torch.Tensor]:
    from transformers import JanusForConditionalGeneration

    torch.manual_seed(0)
    model = JanusForConditionalGeneration(config).to(torch.bfloat16)
    return {key: value.detach().clone().contiguous() for key, value in model.state_dict().items()}


def _save_tokenizer(path: Path) -> None:
    """A byte-level BPE tokenizer carrying Janus' boundary tokens.

    ``extra_special_tokens`` is what makes ``tokenizer.boi_token`` / ``eoi_token``
    / ``image_token`` survive a save/load round trip; the chat template reads
    them by name. Byte-level (not Metaspace) because the layout rejects a
    Metaspace tokenizer as a mislabelled export.
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
        "<image_placeholder>": 6,
        "a": 7,
        "b": 8,
    }
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        extra_special_tokens={
            "image_token": "<image_placeholder>",
            "boi_token": "<begin_of_image>",
            "eoi_token": "<end_of_image>",
        },
    ).save_pretrained(path)


def _write_hf_checkpoint(root: Path, *, tied: bool = False) -> dict[str, torch.Tensor]:
    """Write a tiny upstream Janus root (single-file safetensors)."""
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    config = _janus_config(tied=tied)
    root.mkdir(parents=True)
    config.save_pretrained(root)
    JanusImageProcessor(image_size=8, patch_size=4).save_pretrained(root)
    _save_tokenizer(root)
    tensors = _source_tensors(config)
    if tied:
        # A real tied export omits ``lm_head.weight``; HF's ``state_dict()`` still
        # lists it (aliasing the embedding).
        tensors.pop("lm_head.weight", None)
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return tensors


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    root = tmp_path / "Janus-tiny"
    return root, _write_hf_checkpoint(root)


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = JANUS_HF_LAYOUT
    assert layout.route("model.vision_model.embeddings.patch_embedding.weight") == (
        SIGLIP,
        "vision_model.embeddings.patch_embedding.weight",
    )
    assert layout.route("model.aligner.weight") == (SIGLIP, "aligner.weight")
    assert layout.route("model.vqmodel.encoder.conv_in.weight") == (VQVAE, "vqmodel.encoder.conv_in.weight")
    assert layout.route("model.generation_head.weight") == (VQVAE, "generation_head.weight")
    assert layout.route("model.language_model.embed_tokens.weight") == (
        TEXT_ENCODER,
        "embed_tokens.weight",
    )
    assert layout.route("lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    assert layout.route("model.language_model.layers.0.mlp.up_proj.weight") == (
        LLAMA,
        "language_model.layers.0.mlp.up_proj.weight",
    )
    # Not a Janus key, so no module claims it.
    assert layout.route("model.visual.patch_embed.weight") is None

    assert layout.source_key(SIGLIP, "aligner.weight") == "model.aligner.weight"
    assert layout.source_key(LLAMA, "language_model.norm.weight") == "model.language_model.norm.weight"
    # Routes back to the text encoder, so the backbone does not own it.
    assert layout.source_key(LLAMA, "language_model.embed_tokens.weight") is None


def test_the_layout_claims_every_upstream_key():
    """Every tensor an upstream Janus checkpoint stores lands on exactly one module."""
    tensors = _source_tensors(_janus_config())
    unrouted = [key for key in tensors if JANUS_HF_LAYOUT.route(key) is None]
    assert unrouted == []
    for key in tensors:
        assert JANUS_HF_LAYOUT.source_key(*JANUS_HF_LAYOUT.route(key)) == key, key


def test_build_config_maps_the_upstream_config_per_module():
    """Each module's config comes from the upstream one, including the tie flag.

    ``tie_word_embeddings_if_requested`` ANDs the outer and nested flags, so the
    layout does too — the same predicate the module then acts on.
    """
    config = _janus_config()
    modules = JANUS_HF_LAYOUT.modules
    assert modules[SIGLIP].build_config(config).model_type == "janus_siglip"
    assert modules[VQVAE].build_config(config).model_type == "janus_vqvae"
    assert modules[TEXT_ENCODER].build_config(config).model_type == "janus_text_encoder"
    assert modules[LLAMA].build_config(config).model_type == "janus_llama"

    # The backbone keeps the upstream understanding-image placeholder.
    assert modules[LLAMA].build_config(config).image_token_id == config.image_token_id

    text_encoder = modules[TEXT_ENCODER]
    assert text_encoder.build_config(_janus_config(tied=True)).tie_word_embeddings is True
    assert text_encoder.build_config(_janus_config(tied=False)).tie_word_embeddings is False
    # Outer ``True`` beside an untied ``text_config`` resolves to untied — the
    # AND the framework applies.
    disagreeing = _janus_config(tied=False)
    disagreeing.tie_word_embeddings = True
    assert text_encoder.build_config(disagreeing).tie_word_embeddings is False


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    assert view != str(root)
    assert resolve_omni_checkpoint_root(root) == view
    assert resolve_omni_checkpoint_root(view) == view

    source = read_hf_source(view)
    assert source.model_type == "janus"
    assert json.loads(Path(view, "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    for name in (SIGLIP, VQVAE, TEXT_ENCODER, LLAMA):
        assert Path(view, name, "config.json").is_file()
    # Each tower binds its own image processor; only the text encoder binds a tokenizer.
    assert Path(view, SIGLIP, "preprocessor_config.json").is_file()
    assert Path(view, VQVAE, "preprocessor_config.json").is_file()
    assert Path(view, TEXT_ENCODER, "tokenizer.json").is_file()
    assert not list(Path(view).rglob("*.safetensors"))

    from veomni.models.seed_omni.configuration_omni import OmniConfig

    assert set(OmniConfig.from_pretrained(str(root)).module_names) == {SIGLIP, VQVAE, TEXT_ENCODER, LLAMA}


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_omni_model_loads_every_module_weight_from_the_hf_root(tmp_path, tied):
    """An HF root loads as if it were a split checkpoint, in both tie modes.

    Janus-1.3B is untied (the file stores its own distinct head); a tied export
    omits ``lm_head.weight`` and the text encoder must project through the
    embedding instead. Both are accepted by ``from_pretrained``, so both have to
    load here.
    """
    root = tmp_path / "Janus-tiny"
    tensors = _write_hf_checkpoint(root, tied=tied)
    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    layout = model.config._hf_source.layout

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            if source_key in tensors:
                assert torch.equal(value, tensors[source_key]), (name, key)
                compared.add(source_key)
    assert compared == set(tensors)

    text_encoder = model.modules_dict[TEXT_ENCODER]
    assert text_encoder._tokenizer is not None
    assert (text_encoder.lm_head is not None) is not tied
    # The backbone owns no wte / lm_head; the text encoder does.
    assert not any("embed_tokens" in key for key in model.modules_dict[LLAMA].state_dict())
