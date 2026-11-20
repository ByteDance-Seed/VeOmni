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


def _janus_config(*, tied: bool = False, nested_tied: bool | None = None) -> JanusConfig:
    """A Janus config small enough to build on CPU.

    Janus-1.3B ships untied, so that is the default here. ``nested_tied`` sets
    the ``text_config`` flag independently: upstream expands the tied mapping
    from the *outer* flag alone, so the two can disagree.
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
        tie_word_embeddings=tied if nested_tied is None else nested_tied,
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


def _write_hf_checkpoint(
    root: Path,
    *,
    tied: bool = False,
    nested_tied: bool | None = None,
    head: str = "auto",
    serialization: str = "safetensors",
) -> dict[str, torch.Tensor]:
    """Write a tiny upstream Janus root.

    ``head`` says what the checkpoint stores for ``lm_head.weight``:

    * ``"auto"`` — omit it when tied (what a tied export does), else store a
      distinct head (what an untied export does);
    * ``"omit"`` — never store it, even when the flags say untied;
    * ``"equal"`` — store it byte-identical to the embedding;
    * ``"different"`` — store it with its own values.

    ``serialization="bin"`` writes ``pytorch_model.bin`` instead, so a rule that
    consults the stored tensors is held to both formats `from_pretrained` takes.
    """
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    config = _janus_config(tied=tied, nested_tied=nested_tied)
    root.mkdir(parents=True)
    config.save_pretrained(root)
    JanusImageProcessor(image_size=8, patch_size=4).save_pretrained(root)
    _save_tokenizer(root)
    tensors = _source_tensors(config)

    if head == "auto":
        if tied:
            # A real tied export omits ``lm_head.weight``; HF's ``state_dict()``
            # still lists it (aliasing the embedding).
            tensors.pop("lm_head.weight", None)
    elif head == "omit":
        tensors.pop("lm_head.weight", None)
    elif head == "equal":
        tensors["lm_head.weight"] = tensors["model.language_model.embed_tokens.weight"].clone()
    elif head == "different":
        tensors["lm_head.weight"] = tensors["lm_head.weight"] + 1.0
    else:  # pragma: no cover - guards the call sites
        raise ValueError(f"unknown head mode {head!r}")

    if serialization == "bin":
        torch.save(tensors, root / "pytorch_model.bin")
    elif serialization == "safetensors":
        save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    else:  # pragma: no cover - guards the call sites
        raise ValueError(f"unknown serialization {serialization!r}")
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

    The tie decision follows upstream ``from_pretrained``: the flag decides, so
    it is read off the top-level ``tie_word_embeddings`` (upstream expands the
    tied mapping from that flag alone — the nested ``text_config`` value does not
    take part).
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
    # Upstream ties from the outer flag, so an untied ``text_config`` beside it
    # does not veto the tie (verified against transformers 5.16.1).
    disagreeing = _janus_config(tied=False)
    disagreeing.tie_word_embeddings = True
    assert text_encoder.build_config(disagreeing).tie_word_embeddings is True


@pytest.mark.parametrize("serialization", ["safetensors", "bin"])
def test_stored_head_uses_the_same_weight_file_as_the_loader(tmp_path, serialization):
    """A stale shard index cannot override the single file selected by the loader."""
    from veomni.models.checkpoint.weights import _load_state_dict
    from veomni.models.seed_omni.utils.hf_layout import read_source_weight_files

    embed_key = "model.language_model.embed_tokens.weight"
    tensors = {embed_key: torch.zeros(2, 2), "lm_head.weight": torch.ones(2, 2)}
    stale = {embed_key: torch.zeros(2, 2)}
    if serialization == "safetensors":
        single, shard, index = "model.safetensors", "model-00001-of-00001.safetensors", "model.safetensors.index.json"
        save_file(tensors, tmp_path / single)
        save_file(stale, tmp_path / shard)
    else:
        single, shard, index = "pytorch_model.bin", "pytorch_model-00001-of-00001.bin", "pytorch_model.bin.index.json"
        torch.save(tensors, tmp_path / single)
        torch.save(stale, tmp_path / shard)
    (tmp_path / index).write_text(json.dumps({"weight_map": {embed_key: shard}}))

    files = read_source_weight_files(str(tmp_path))
    assert files == dict.fromkeys(tensors, single)
    loaded = {key: value for iterator in _load_state_dict(str(tmp_path)) for key, value in iterator}
    assert torch.equal(loaded["lm_head.weight"], tensors["lm_head.weight"])
    config = JANUS_HF_LAYOUT.modules[TEXT_ENCODER].build_config(_janus_config(tied=False), str(tmp_path))
    assert config.tie_word_embeddings is False


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


@pytest.mark.parametrize(
    ("tied", "nested_tied", "head", "expect_head"),
    [
        # F015: upstream's own default (outer True, nested False) with the
        # duplicate omitted. The old AND-based rule built an independent head
        # with nothing to load into it — i.e. random weights on a real model.
        (True, False, "omit", False),
        # F022: both flags say tied, but the checkpoint stores a head that
        # differs from the embedding. Upstream warns and keeps them separate, so
        # the stored head must load rather than be replaced by the embedding.
        (True, True, "different", True),
        # A tied export that wrote the duplicate out keeps the tie.
        (True, True, "equal", False),
        # Janus-1.3B: untied flags and a real head of its own.
        (False, False, "different", True),
    ],
    ids=["omitted-head-nested-untied", "stored-head-differs", "stored-head-equal", "plain-untied"],
)
def test_the_stored_head_decides_tying_the_way_from_pretrained_does(tmp_path, tied, nested_tied, head, expect_head):
    """The tie rule has to match upstream, in both directions.

    Upstream's ``from_pretrained`` consults the stored tensors: an omitted head
    is tied regardless of the flags, and a stored head with different values is
    kept independent regardless of them. Only an equal pair (and an unreadable
    checkpoint) falls back to the config.
    """
    root = tmp_path / "Janus-tiny"
    tensors = _write_hf_checkpoint(root, tied=tied, nested_tied=nested_tied, head=head)

    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    text_encoder = model.modules_dict[TEXT_ENCODER]
    assert (text_encoder.lm_head is not None) is expect_head

    if expect_head:
        assert torch.equal(text_encoder.lm_head.weight, tensors["lm_head.weight"])
    else:
        assert torch.equal(
            text_encoder.embed_tokens.weight,
            tensors["model.language_model.embed_tokens.weight"],
        )


@pytest.mark.parametrize("head", ["different", "equal", "omit"], ids=["head-differs", "head-equal", "head-omitted"])
def test_a_bin_root_resolves_tying_from_its_stored_head(tmp_path, head):
    """``.bin`` must reach the same verdict as safetensors, not fall back to the flag.

    The config says tied, and only the stored head disagrees. A rule that read
    safetensors and treated ``.bin`` as unreadable would silently skip a trained
    head here — the same data loss as F022, on the other serialization.
    """
    root = tmp_path / "Janus-bin"
    tensors = _write_hf_checkpoint(root, tied=True, nested_tied=True, head=head, serialization="bin")
    assert (root / "pytorch_model.bin").is_file()
    assert not (root / "model.safetensors").exists()

    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    text_encoder = model.modules_dict[TEXT_ENCODER]

    if head == "different":
        # The stored head wins over the config: it must load, not be replaced.
        assert text_encoder.lm_head is not None
        assert torch.equal(text_encoder.lm_head.weight, tensors["lm_head.weight"])
    else:
        # Equal or omitted: the tie is what upstream keeps.
        assert text_encoder.lm_head is None
        assert torch.equal(
            text_encoder.embed_tokens.weight,
            tensors["model.language_model.embed_tokens.weight"],
        )
