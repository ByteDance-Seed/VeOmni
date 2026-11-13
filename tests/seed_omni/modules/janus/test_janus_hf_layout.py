"""Running the ``janus`` SeedOmni family straight off its upstream HF checkpoint.

A tiny ``JanusForConditionalGeneration`` checkpoint stands in for the upstream
one. The tests check that the ``janus`` :class:`OmniHFLayout` routes every HF key
onto the module that owns it, that ``OmniModel.from_pretrained`` reads the HF
root as if it were a split checkpoint, and that
:func:`save_hf_source_checkpoint` writes the source's own keys back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import JanusConfig, JanusVisionConfig, JanusVQVAEConfig, LlamaConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.janus.hf_layout import JANUS_GENERATION_KWARGS, JANUS_HF_LAYOUT
from veomni.models.seed_omni.utils.hf_layout import (
    read_hf_source,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


SIGLIP = "janus_siglip"
VQVAE = "janus_vqvae"
TEXT_ENCODER = "janus_text_encoder"
LLAMA = "janus_llama"

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _janus_config(*, tied: bool = False, nested_tied: bool | None = None) -> JanusConfig:
    """A Janus config small enough to build on CPU.

    ``tied`` follows the top-level field, which is what
    ``JanusForConditionalGeneration.tie_weights`` reads — Janus-1.3B ships
    untied, so that is the default here.
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
    them by name. Byte-level (not Metaspace) because the family converter
    rejects a Metaspace tokenizer as a mislabelled export.
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
    sharded: bool = False,
    tied: bool = False,
    store_tied_head: bool = False,
    nested_tied: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Write a tiny upstream Janus root.

    ``tied`` sets the flag on both the top-level and the nested text config, the
    shape :meth:`JanusForConditionalGeneration.tie_weights` reads. HF still lists
    ``lm_head.weight`` in ``state_dict()`` (it aliases the embedding), but a real
    tied export omits the file, so ``store_tied_head=False`` drops it.
    """
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    config = _janus_config(tied=tied, nested_tied=nested_tied)
    root.mkdir(parents=True)
    config.save_pretrained(root)
    JanusImageProcessor(image_size=8, patch_size=4).save_pretrained(root)
    _save_tokenizer(root)
    tensors = _source_tensors(config)
    if tied and not store_tied_head:
        tensors.pop("lm_head.weight", None)
    if not sharded:
        save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
        return tensors
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map = {key: shards[0] if ".layers.0." in key else shards[1] for key in tensors}
    for shard in shards:
        save_file(
            {key: value for key, value in tensors.items() if weight_map[key] == shard},
            root / shard,
            metadata={"format": "pt"},
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return tensors


def _read_safetensors(root: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")
    tensors = {}
    for filename in sorted(set(weight_map.values())):
        with safe_open(root / filename, framework="pt") as f:
            tensors.update({key: f.get_tensor(key) for key in f.keys()})
    return tensors, weight_map


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

    ``post_process_after_weight_loading`` ANDs the outer and nested flags, so the
    layout does too: guessing from one loses a real ``lm_head.weight`` in one
    direction and drops a genuinely tied embedding in the other.
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
    # A stale top-level flag beside an untied text_config means the file stores a
    # head: build it rather than substituting the embedding.
    disagreeing = _janus_config(tied=False)
    disagreeing.tie_word_embeddings = True
    assert text_encoder.build_config(disagreeing).tie_word_embeddings is False


def test_the_backbone_never_ties_itself():
    """``janus_llama`` holds no wte / lm_head, so a tie there crashes the post-load step.

    Its input embedding is ``nn.Identity``; the shared tie does
    ``input_embeddings._parameters["weight"]`` and raises
    ``RuntimeError('Failed to tie input/output embeddings')``.
    """
    llm = JANUS_HF_LAYOUT.modules[LLAMA].build_config(_janus_config(tied=True))
    assert llm.tie_word_embeddings is False
    assert llm.text_config.tie_word_embeddings is False


@pytest.mark.parametrize(
    ("tied", "nested_tied", "store_tied_head", "expect_head", "equal_weights"),
    [
        # Janus-1.3B: untied config, the file stores its own distinct head.
        (False, None, False, True, False),
        # A genuinely tied export omits lm_head.weight.
        (True, None, False, False, False),
        # Tied config and the duplicate stored with *equal* values: transformers
        # keeps the tie there, so the head must stay tied rather than become an
        # independent copy that SFT would drift away from the embedding.
        (True, None, True, False, True),
        # Equal stored tensors outrank an untied config: the pair is shared in the
        # source model, so splitting it would let SFT update the two apart.
        (False, None, True, False, True),
        # Outer ``True`` beside an untied ``text_config`` with a stored, distinct
        # head: a stale top-level flag must not drop it.
        (True, False, True, True, False),
        # Same flags with NO stored head: there is no tensor to load into an
        # independent head, so tying is the only option — building one anyway left
        # it randomly initialised, corrupting every logit.
        (True, False, False, False, False),
    ],
    ids=[
        "untied",
        "tied",
        "tied-with-duplicate",
        "equal-weights-outrank-config",
        "outer-tied-nested-untied-with-head",
        "outer-tied-nested-untied-no-head",
    ],
)
def test_the_layout_decides_tying_from_the_config_and_the_stored_weights(
    tmp_path, tied, nested_tied, store_tied_head, expect_head, equal_weights
):
    """A tie needs the config's permission and weights that support it.

    The backbone used to crash outright on a tied checkpoint (its input embedding
    is ``nn.Identity``, so the shared post-load tie did ``_parameters["weight"]``
    on it), and reading only one of the two signals lost a real head or coupled a
    tied pair. This pins every combination end to end.
    """
    root = tmp_path / "Janus-tiny"
    tensors = _write_hf_checkpoint(root, tied=tied, nested_tied=nested_tied, store_tied_head=store_tied_head)

    if equal_weights:
        # Same values as the embedding: the tied pair HF keeps together.
        tensors["lm_head.weight"] = tensors["model.language_model.embed_tokens.weight"].clone()
    elif "lm_head.weight" in tensors:
        # Stored separately with its own values: a genuinely independent head.
        tensors["lm_head.weight"] = tensors["lm_head.weight"] + 1.0
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    text_encoder = model.modules_dict[TEXT_ENCODER]
    assert (text_encoder.lm_head is not None) is expect_head

    embedding = tensors["model.language_model.embed_tokens.weight"]
    if expect_head:
        assert torch.equal(text_encoder.lm_head.weight, tensors["lm_head.weight"])
    else:
        # No head of its own, so ``decode`` projects through the embedding.
        assert torch.equal(text_encoder.embed_tokens.weight, embedding)


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


def test_omni_model_loads_every_module_weight_from_the_hf_root(hf_root):
    root, tensors = hf_root
    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    layout = model.config._hf_source.layout

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            if source_key in tensors:
                assert torch.equal(value, tensors[source_key]), (name, key)
                compared.add(source_key)
    # The untied source's lm_head is loaded by the text encoder, not skipped.
    assert "lm_head.weight" in compared
    assert compared == set(tensors)

    assert model.modules_dict[TEXT_ENCODER].lm_head is not None
    assert model.modules_dict[TEXT_ENCODER]._tokenizer is not None
    # The backbone owns no wte / lm_head; the text encoder does.
    assert not any("embed_tokens" in key for key in model.modules_dict[LLAMA].state_dict())
    assert "infer_interleave" in model.config.generation_graphs


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_export_writes_back_the_source_layout(tmp_path, sharded):
    root = tmp_path / "Janus-tiny"
    tensors = _write_hf_checkpoint(root, sharded=sharded)
    model = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    text_encoder = model.modules_dict[TEXT_ENCODER]
    with torch.no_grad():
        text_encoder.embed_tokens.weight.add_(1.0)
        text_encoder.lm_head.weight.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {TEXT_ENCODER: (text_encoder, None)}, str(out))

    exported, weight_map = _read_safetensors(root=out)
    _, source_map = _read_safetensors(root=root)
    # Same shard files as the source, so the export loads wherever that does.
    assert weight_map == source_map
    assert sorted(path.name for path in out.iterdir()) == sorted(path.name for path in root.iterdir())
    assert set(exported) == set(tensors)
    for key, value in tensors.items():
        assert exported[key].dtype == value.dtype, key

    trained_embed = tensors["model.language_model.embed_tokens.weight"] + 1.0
    trained_head = tensors["lm_head.weight"] + 1.0
    assert torch.equal(exported["model.language_model.embed_tokens.weight"], trained_embed)
    assert torch.equal(exported["lm_head.weight"], trained_head)
    # Modules not handed in are carried from the source unchanged.
    for key, value in tensors.items():
        if key not in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            assert torch.equal(exported[key], value), key


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_offline_convert_matches_the_runtime_load(tmp_path, tied):
    """The offline convert and the direct load must agree.

    ``convert_checkpoint`` now routes a standard HF root through the layout, so
    this pins that the split it writes is what the runtime load reads back — and
    that the defaults the layout declares (``generation_kwargs``, ``infer_type``)
    survive into the split root's config.
    """
    from safetensors.torch import load_file

    from veomni.models.seed_omni.utils.convert_registry import _save_converted_omni, convert_checkpoint
    from veomni.models.seed_omni.utils.hf_layout import convert_with_hf_layout

    root = tmp_path / "Janus-tiny"
    _write_hf_checkpoint(root, tied=tied)

    split = tmp_path / "split"
    convert_checkpoint(str(root), str(split), attn_implementation="eager")
    root_config = json.loads((split / "config.json").read_text(encoding="utf-8"))
    assert root_config["model_type"] == "omni"
    assert read_hf_source(split) is None
    # The layout's decode defaults reach the split root, not just the offline path.
    assert root_config["generation_kwargs"] == JANUS_GENERATION_KWARGS
    assert root_config["infer_type"] == "infer_interleave"

    from_layout = tmp_path / "from_layout"
    _save_converted_omni(str(from_layout), **convert_with_hf_layout(str(root), attn_implementation="eager"))

    for name in (SIGLIP, VQVAE, TEXT_ENCODER, LLAMA):
        converted = load_file(split / name / "model.safetensors")
        loaded = load_file(from_layout / name / "model.safetensors")
        assert converted.keys() == loaded.keys(), name
        for key in converted:
            assert torch.equal(converted[key], loaded[key]), (name, key)

    # The direct load of the HF root agrees with both.
    runtime = OmniModel.from_pretrained(str(root), dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
    for name in (SIGLIP, VQVAE, TEXT_ENCODER, LLAMA):
        expected = load_file(split / name / "model.safetensors")
        actual = runtime.modules_dict[name].state_dict()
        assert expected.keys() == actual.keys(), name
        for key in expected:
            assert torch.equal(expected[key], actual[key]), (name, key)


def test_declared_graphs_and_infer_type_match_the_family_converter():
    layout = JANUS_HF_LAYOUT
    # The interleave FSM is the family default on both the layout and converter paths.
    assert layout.infer_type == "infer_interleave"
    assert set(layout.generation_graphs) == {"infer_gen", "infer_und", "infer_interleave"}
    assert layout.training_graphs == "train/graph_train.yaml"


def test_a_bin_only_root_is_refused_with_a_usable_message(tmp_path):
    """The layout reads safetensors; a ``.bin`` root gets told so, up front.

    ``source_weight_files`` is safetensors-only, so a ``.bin`` checkpoint cannot be
    loaded through the layout at all. Answering the tie question from its config
    flags would move the failure into ``safe_open``, which raises a
    ``SafetensorError`` while the module is being built.
    """
    root = tmp_path / "Janus-bin"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "janus"}), encoding="utf-8")
    torch.save({"lm_head.weight": torch.zeros(4, 4)}, root / "pytorch_model.bin")
    (root / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "pytorch_model.bin"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reads safetensors only"):
        JANUS_HF_LAYOUT.read_hf_config(str(root))
