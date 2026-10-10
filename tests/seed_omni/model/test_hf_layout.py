"""Running a SeedOmni family straight off its upstream HF checkpoint.

A tiny Qwen3 checkpoint stands in for the upstream one. The tests check that
the ``qwen3`` :class:`OmniHFLayout` routes every HF key onto the module that
owns it, that ``OmniModel.from_pretrained`` reads the HF root as if it were a
split checkpoint, and that :func:`save_hf_source_checkpoint` writes the
source's own keys, dtypes and shards back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config

from veomni.arguments.omni_arguments_types import OmniModuleRuntimeArguments, _validate_hf_view_modules
from veomni.models.module_utils import _load_state_dict
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3.hf_layout import QWEN3_HF_LAYOUT
from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint
from veomni.models.seed_omni.utils.hf_layout import (
    OmniHFLayout,
    OmniHFModuleLayout,
    read_hf_source,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


TEXT_ENCODER = "qwen3_text_encoder"
LLM = "qwen3_llm"


def _qwen3_config(*, tied: bool = True) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        dtype="bfloat16",
    )


def _source_tensors(config: Qwen3Config) -> dict[str, torch.Tensor]:
    """Every weight a Qwen3 checkpoint stores, plus the tied ``lm_head`` copy small Qwen3s keep."""
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    tensors = {key: value.detach().clone().contiguous() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    return tensors


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(path)


def _write_hf_checkpoint(root: Path, *, sharded: bool, tied: bool = True) -> dict[str, torch.Tensor]:
    config = _qwen3_config(tied=tied)
    root.mkdir(parents=True)
    config.save_pretrained(root)
    _save_tokenizer(root)
    (root / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
    tensors = _source_tensors(config)
    if not sharded:
        save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})
        return tensors
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map = {key: shards[0] if key.startswith("model.layers.0.") else shards[1] for key in tensors}
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


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    root = tmp_path / "Qwen3-tiny"
    return root, _write_hf_checkpoint(root, sharded=False)


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


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = QWEN3_HF_LAYOUT
    assert layout.route("model.embed_tokens.weight") == (TEXT_ENCODER, "embed_tokens.weight")
    assert layout.route("lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    assert layout.route("model.layers.1.mlp.up_proj.weight") == (LLM, "language_model.layers.1.mlp.up_proj.weight")
    assert layout.route("model.norm.weight") == (LLM, "language_model.norm.weight")
    assert layout.route("visual.patch_embed.weight") is None

    assert layout.source_key(TEXT_ENCODER, "embed_tokens.weight") == "model.embed_tokens.weight"
    assert layout.source_key(LLM, "language_model.norm.weight") == "model.norm.weight"
    # Routes back to the text encoder, so it is not an llm key.
    assert layout.source_key(LLM, "language_model.embed_tokens.weight") is None
    assert layout.source_key(LLM, "rotary_emb.inv_freq") is None


def test_layout_rejects_a_source_prefix_claimed_by_two_modules():
    layout = OmniHFLayout(
        modules={
            "a": OmniHFModuleLayout(key_prefixes=(("model.", "a."),), build_config=lambda c: c),
            "b": OmniHFModuleLayout(key_prefixes=(("model.", "b."),), build_config=lambda c: c),
        }
    )
    with pytest.raises(ValueError, match="more than one module"):
        layout.route("model.x")


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    assert view != str(root)
    assert resolve_omni_checkpoint_root(root) == view
    assert resolve_omni_checkpoint_root(view) == view

    source = read_hf_source(view)
    assert source.model_type == "qwen3" and source.path == os.path.realpath(root)
    assert json.loads(Path(view, "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    for name in (TEXT_ENCODER, LLM):
        assert Path(view, name, "config.json").is_file()
    assert Path(view, TEXT_ENCODER, "tokenizer.json").is_file()
    assert not list(Path(view).rglob("*.safetensors"))
    assert set(OmniConfig.from_pretrained(str(root)).module_names) == {TEXT_ENCODER, LLM}


def test_split_roots_pass_through_and_unregistered_hf_types_fail(tmp_path):
    omni = tmp_path / "omni"
    omni.mkdir()
    (omni / "config.json").write_text(json.dumps({"model_type": "omni"}), encoding="utf-8")
    assert resolve_omni_checkpoint_root(omni) == omni
    assert read_hf_source(omni) is None

    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "not_a_registered_family"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no SeedOmni HF layout is registered"):
        resolve_omni_checkpoint_root(other)


def test_omni_model_loads_every_module_weight_from_the_hf_root(hf_root):
    root, tensors = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    layout = model.config._hf_source.layout

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            if source_key in tensors:
                assert torch.equal(value, tensors[source_key]), (name, key)
                compared.add(source_key)
    assert compared == set(tensors) - {"lm_head.weight"}
    assert model.modules_dict[TEXT_ENCODER].lm_head is None
    assert model.modules_dict[TEXT_ENCODER]._tokenizer is not None
    assert "infer_text" in model.config.generation_graphs


def test_an_untied_head_loads_even_through_a_wrapper(tmp_path):
    root = tmp_path / "Qwen3-untied"
    tensors = _write_hf_checkpoint(root, sharded=False, tied=False)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    text_encoder = model.modules_dict[TEXT_ENCODER]
    assert torch.equal(text_encoder.lm_head.weight, tensors["lm_head.weight"])

    # A LoRA wrapper prefixes every name; the head must still be read.
    wrapper = torch.nn.Module()
    wrapper.base_model = text_encoder
    converter = text_encoder._create_checkpoint_tensor_converter(wrapper)
    assert not converter.should_skip_without_loading("lm_head.weight")
    assert converter.should_skip_without_loading("model.norm.weight")


def test_auto_dtype_is_the_checkpoint_dtype(hf_root):
    root, _ = hf_root
    model = OmniModel.from_pretrained(str(root), torch_dtype="auto", device_map="cpu")
    for module in model.modules_dict.values():
        assert {param.dtype for param in module.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_export_writes_back_the_source_layout(tmp_path, sharded):
    root = tmp_path / "Qwen3-tiny"
    tensors = _write_hf_checkpoint(root, sharded=sharded)
    model = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    text_encoder = model.modules_dict[TEXT_ENCODER]
    with torch.no_grad():
        text_encoder.embed_tokens.weight.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {TEXT_ENCODER: (text_encoder, None)}, str(out))

    exported, weight_map = _read_safetensors(root=out)
    _, source_map = _read_safetensors(root=root)
    assert weight_map == source_map
    assert sorted(path.name for path in out.iterdir()) == sorted(path.name for path in root.iterdir())
    for key, value in tensors.items():
        assert exported[key].dtype == value.dtype, key
    trained = tensors["model.embed_tokens.weight"] + 1.0
    assert torch.equal(exported["model.embed_tokens.weight"], trained)
    # The tied duplicate follows its trained original, not the stale source copy.
    assert torch.equal(exported["lm_head.weight"], trained)
    # The llm was not handed in, so its weights are carried from the source.
    for key, value in tensors.items():
        if key not in ("model.embed_tokens.weight", "lm_head.weight"):
            assert torch.equal(exported[key], value), key


def test_offline_convert_matches_the_runtime_load(hf_root, tmp_path):
    root, tensors = hf_root
    out = tmp_path / "split"
    convert_checkpoint(str(root), str(out))
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    assert read_hf_source(out) is None

    split = OmniModel.from_pretrained(str(out), torch_dtype=torch.bfloat16, device_map="cpu")
    runtime = OmniModel.from_pretrained(str(root), torch_dtype=torch.bfloat16, device_map="cpu")
    for name in (TEXT_ENCODER, LLM):
        expected = runtime.modules_dict[name].state_dict()
        actual = split.modules_dict[name].state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), (name, key)


def test_state_dict_loader_skips_declared_keys_without_reading_them(hf_root):
    root, tensors = hf_root
    skip = lambda key: key.startswith("model.layers.")  # noqa: E731
    loaded = {}
    for iterator in _load_state_dict(str(root), skip_key=skip):
        for key, value in iterator:
            loaded[key] = value
    assert set(loaded) == {key for key in tensors if not skip(key)}


def test_hf_view_modules_must_all_come_from_the_layout(hf_root):
    root, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    ok = {name: OmniModuleRuntimeArguments(model_path=os.path.join(view, name)) for name in (TEXT_ENCODER, LLM)}
    _validate_hf_view_modules(view, ok)

    with pytest.raises(ValueError, match="not part of the `qwen3` HF layout"):
        _validate_hf_view_modules(view, {**ok, "qwen3vl_vision": OmniModuleRuntimeArguments(model_path="/x")})
    with pytest.raises(ValueError, match="sets its own model_path"):
        _validate_hf_view_modules(view, {**ok, LLM: OmniModuleRuntimeArguments(model_path="/elsewhere/qwen3_llm")})


def test_a_training_export_needs_safetensors_in_the_source(tmp_path):
    from veomni.arguments.omni_arguments_types import _check_hf_source_exportable

    root = tmp_path / "Qwen3-bin"
    tensors = _write_hf_checkpoint(root, sharded=False)
    (root / "model.safetensors").unlink()
    torch.save(tensors, root / "pytorch_model.bin")
    view = resolve_omni_checkpoint_root(root)

    with pytest.raises(ValueError, match="needs safetensors weights"):
        _check_hf_source_exportable(view)
