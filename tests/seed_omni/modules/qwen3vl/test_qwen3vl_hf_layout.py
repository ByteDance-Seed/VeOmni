"""Running the ``qwen3_vl`` SeedOmni family straight off its upstream HF checkpoint.

A tiny Qwen3-VL checkpoint stands in for the upstream one. The tests check that
the ``qwen3_vl`` :class:`OmniHFLayout` routes every HF key onto the module that
owns it, that ``OmniModel.from_pretrained`` reads the HF root as if it were a
split checkpoint, and that :func:`save_hf_source_checkpoint` writes the source's
own keys back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3VLConfig

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments, _validate_hf_view_modules
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3vl.hf_layout import QWEN3VL_HF_LAYOUT
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint
from veomni.models.seed_omni.utils.hf_layout import (
    has_hf_layout,
    read_hf_source,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


VISION = "qwen3vl_vision"
TEXT_ENCODER = "qwen3vl_text_encoder"
LLM = "qwen3vl_llm"

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _qwen3vl_config(*, tied: bool = True) -> Qwen3VLConfig:
    """A checkpoint small enough to build in a test, sharing the real key names."""
    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            dtype="bfloat16",
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0],
        ),
        tie_word_embeddings=tied,
    )


def _source_tensors(config: Qwen3VLConfig) -> dict[str, torch.Tensor]:
    """Every weight the upstream checkpoint stores, including small-model tied head copies."""
    from transformers import Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    tensors = {key: value.detach().clone().contiguous() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        # 2B keeps the duplicate on disk although the config ties the two.
        tensors["lm_head.weight"] = tensors["model.language_model.embed_tokens.weight"].clone()
    return tensors


def _save_assets(root: Path) -> None:
    """Tokenizer + both vision processors, as an upstream Qwen3-VL root ships them."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLVideoProcessor

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(root)
    Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16 * 16 * 4,
        max_pixels=16 * 16 * 64,
        size={"shortest_edge": 64, "longest_edge": 128},
    ).save_pretrained(root)
    Qwen3VLVideoProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16 * 16 * 4,
        max_pixels=16 * 16 * 64,
    ).save_pretrained(root)


def _write_hf_checkpoint(root: Path, *, tied: bool = True) -> dict[str, torch.Tensor]:
    config = _qwen3vl_config(tied=tied)
    root.mkdir(parents=True)
    config.save_pretrained(root)
    _save_assets(root)
    tensors = _source_tensors(config)
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
    return tensors


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    root = tmp_path / "Qwen3-VL-tiny"
    return root, _write_hf_checkpoint(root)


def _read_safetensors(root: Path) -> dict[str, torch.Tensor]:
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
    return tensors


def test_qwen3_vl_registers_a_layout_beside_its_converter():
    """The family keeps both: the converter for the offline split, the layout for a direct load."""
    from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY

    assert has_hf_layout("qwen3_vl")
    assert "qwen3_vl" in OMNI_CONVERT_REGISTRY.valid_keys()


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = QWEN3VL_HF_LAYOUT
    assert layout.route("model.visual.blocks.0.attn.qkv.weight") == (VISION, "visual.blocks.0.attn.qkv.weight")
    assert layout.route("model.language_model.embed_tokens.weight") == (TEXT_ENCODER, "embed_tokens.weight")
    assert layout.route("lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    # The text-encoder prefix is longer, so the backbone does not claim the embedding.
    assert layout.route("model.language_model.layers.3.mlp.up_proj.weight") == (
        LLM,
        "language_model.layers.3.mlp.up_proj.weight",
    )
    assert layout.route("model.language_model.norm.weight") == (LLM, "language_model.norm.weight")
    assert layout.route("encoder.proj.weight") is None

    assert layout.source_key(VISION, "visual.merger.linear_fc1.weight") == "model.visual.merger.linear_fc1.weight"
    assert layout.source_key(TEXT_ENCODER, "embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert layout.source_key(LLM, "language_model.norm.weight") == "model.language_model.norm.weight"
    # Routes back to the text encoder, so it is not an llm key.
    assert layout.source_key(LLM, "language_model.embed_tokens.weight") is None


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    assert view != str(root)
    assert resolve_omni_checkpoint_root(root) == view
    assert resolve_omni_checkpoint_root(view) == view

    source = read_hf_source(view)
    assert source.model_type == "qwen3_vl" and source.path == os.path.realpath(root)
    assert json.loads(Path(view, "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    for name in (VISION, TEXT_ENCODER, LLM):
        assert Path(view, name, "config.json").is_file()
    # Both processors land in the vision module, the tokenizer in the text encoder.
    assert Path(view, VISION, "preprocessor_config.json").is_file()
    assert Path(view, VISION, "video_preprocessor_config.json").is_file()
    assert Path(view, TEXT_ENCODER, "tokenizer.json").is_file()
    assert not list(Path(view).rglob("*.safetensors"))
    assert set(OmniConfig.from_pretrained(str(root)).module_names) == {VISION, TEXT_ENCODER, LLM}
    assert OmniConfig.from_pretrained(str(root)).infer_type == "vision_understanding"


def test_omni_model_loads_every_module_weight_from_the_hf_root(hf_root):
    root, tensors = hf_root
    model = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    layout = model.config._hf_source.layout

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            if source_key in tensors:
                assert torch.equal(value, tensors[source_key]), (name, key)
                compared.add(source_key)
    # Every source tensor is read by exactly one module, bar the tied head duplicate.
    assert compared == set(tensors) - {"lm_head.weight"}
    assert model.modules_dict[TEXT_ENCODER].lm_head is None
    assert model.modules_dict[VISION]._image_processor is not None
    assert model.modules_dict[VISION]._video_processor is not None
    assert model.modules_dict[TEXT_ENCODER]._tokenizer is not None
    assert "vision_understanding" in model.config.generation_graphs


def test_the_backbone_drops_the_embedding_it_shares_with_the_text_encoder(hf_root):
    """The split's one real transform: the backbone swaps `embed_tokens` for `nn.Identity`."""
    root, tensors = hf_root
    model = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    backbone = model.modules_dict[LLM]
    assert isinstance(backbone.language_model.embed_tokens, torch.nn.Identity)
    assert not [key for key in backbone.state_dict() if "embed_tokens" in key]
    assert torch.equal(
        model.modules_dict[TEXT_ENCODER].embed_tokens.weight,
        tensors["model.language_model.embed_tokens.weight"],
    )


def test_an_untied_head_loads(tmp_path):
    """An untied checkpoint must give the text encoder a real `lm_head` weight."""
    root = tmp_path / "Qwen3-VL-untied"
    tensors = _write_hf_checkpoint(root, tied=False)
    assert "lm_head.weight" in tensors

    model = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    encoder = model.modules_dict[TEXT_ENCODER]
    assert encoder.lm_head is not None, "an untied checkpoint must not fall back to the tied path"
    assert torch.equal(encoder.lm_head.weight, tensors["lm_head.weight"])


def test_a_tied_head_stores_no_separate_projection(hf_root):
    """The 2B-style tied checkpoint reuses `embed_tokens` for the decode head."""
    root, tensors = hf_root
    model = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    assert model.modules_dict[TEXT_ENCODER].lm_head is None
    assert torch.equal(
        model.modules_dict[TEXT_ENCODER].embed_tokens.weight,
        tensors["model.language_model.embed_tokens.weight"],
    )


def test_export_writes_back_the_source_layout(hf_root, tmp_path):
    root, tensors = hf_root
    model = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    vision = model.modules_dict[VISION]
    with torch.no_grad():
        vision.visual.merger.linear_fc1.weight.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {VISION: (vision, None)}, str(out))

    exported = _read_safetensors(out)
    assert set(exported) == set(tensors)
    for key, value in tensors.items():
        assert exported[key].dtype == value.dtype, key
    trained = "model.visual.merger.linear_fc1.weight"
    assert torch.equal(exported[trained], tensors[trained] + 1.0)
    # Modules that were not handed in are carried from the source.
    for key, value in tensors.items():
        if key != trained:
            assert torch.equal(exported[key], value), key


def test_offline_convert_matches_the_runtime_load(hf_root, tmp_path):
    root, _ = hf_root
    out = tmp_path / "split"
    convert_checkpoint(str(root), str(out), attn_implementation="eager")
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    assert read_hf_source(out) is None

    split = OmniModel.from_pretrained(
        str(out), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    runtime = OmniModel.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    for name in (VISION, TEXT_ENCODER, LLM):
        expected = runtime.modules_dict[name].state_dict()
        actual = split.modules_dict[name].state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), (name, key)


def test_hf_view_modules_must_all_come_from_the_layout(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    ok = {
        name: OmniModuleRuntimeArguments(model_path=os.path.join(view, name)) for name in (VISION, TEXT_ENCODER, LLM)
    }
    _validate_hf_view_modules(view, ok)

    with pytest.raises(ValueError, match="not part of the `qwen3_vl` HF layout"):
        _validate_hf_view_modules(view, {**ok, "qwen3_llm": OmniModuleRuntimeArguments(model_path="/x")})
    with pytest.raises(ValueError, match="sets its own model_path"):
        _validate_hf_view_modules(view, {**ok, VISION: OmniModuleRuntimeArguments(model_path="/elsewhere/vision")})
