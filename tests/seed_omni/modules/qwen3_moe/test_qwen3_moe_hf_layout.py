"""Qwen3-MoE ``OMNI_HF_LAYOUT_REGISTRY["qwen3_moe"]``: direct load and export.

The layout lets ``model.model_path`` point straight at an upstream
``Qwen3MoeForCausalLM`` checkpoint, with no offline ``convert_model.py`` step.
Two directions are covered, because they are not symmetric:

* **Load** — source keys are renamed onto module parameters. The vocabulary half
  takes ``model.embed_tokens.`` / ``lm_head.``; the backbone claims the rest and
  renames ``model.`` → ``language_model.``.
* **Export** — module keys are renamed back and cast to the source dtype, and any
  source key no live module produced is carried from the source verbatim.

The expert layout is the part worth being careful about, and it differs between
the file and the process: a checkpoint stores experts **per expert** on disk
(``experts.{j}.{gate,up,down}_proj.weight``) while ``state_dict()`` fuses them
into ``experts.gate_up_proj`` / ``experts.down_proj``. So the backbone's attached
per-expert converter *does* fire when loading an upstream root — the opposite of
a split checkpoint written by ``convert_model.py``, which is already fused.
These tests write through ``save_pretrained`` so they see the on-disk layout a
real checkpoint has.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.qwen3_moe.hf_layout import QWEN3_MOE_HF_LAYOUT
from veomni.models.seed_omni.utils.hf_layout import (
    OMNI_HF_LAYOUT_REGISTRY,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


TEXT_ENCODER = "qwen3_text_encoder"
MOE_LLM = "qwen3_moe_llm"


def _qwen3_moe_config(*, tied: bool = False) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=tied,
        dtype="bfloat16",
    )


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "a": 4, "b": 5}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ).save_pretrained(path)


def _write_hf_checkpoint(root: Path, *, tied: bool = False) -> dict[str, torch.Tensor]:
    """A real upstream checkpoint, read back from disk so the expert layout is the file's."""
    config = _qwen3_moe_config(tied=tied)
    root.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(root)
    _save_tokenizer(root)
    (root / "generation_config.json").write_text(json.dumps({"do_sample": False}), encoding="utf-8")
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(root)
    return load_file(str(root / "model.safetensors"))


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    root = tmp_path / "Qwen3-MoE-tiny"
    return root, _write_hf_checkpoint(root)


def test_qwen3_moe_registers_a_layout_beside_its_converter():
    """Both entry points exist: an offline convert and a direct HF load."""
    assert "qwen3_moe" in set(OMNI_HF_LAYOUT_REGISTRY.valid_keys())
    assert set(QWEN3_MOE_HF_LAYOUT.modules) == {TEXT_ENCODER, MOE_LLM}
    assert QWEN3_MOE_HF_LAYOUT.infer_type == "infer_text"
    # The vocabulary half is shared with dense Qwen3, so its config comes from there.
    from veomni.models.seed_omni.modules.qwen3.text_encoder.configuration import Qwen3TextEncoderConfig

    assert isinstance(
        QWEN3_MOE_HF_LAYOUT.modules[TEXT_ENCODER].build_config(_qwen3_moe_config()), Qwen3TextEncoderConfig
    )


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = QWEN3_MOE_HF_LAYOUT

    assert layout.route("model.embed_tokens.weight") == (TEXT_ENCODER, "embed_tokens.weight")
    assert layout.route("lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    assert layout.route("model.layers.0.mlp.experts.0.gate_proj.weight") == (
        MOE_LLM,
        "language_model.layers.0.mlp.experts.0.gate_proj.weight",
    )
    assert layout.route("model.norm.weight") == (MOE_LLM, "language_model.norm.weight")

    # Every routed key must map back to itself, or an export would rename it wrong.
    for key in (
        "model.embed_tokens.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.norm.weight",
    ):
        module, module_key = layout.route(key)
        assert layout.source_key(module, module_key) == key


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _ = hf_root
    view = Path(resolve_omni_checkpoint_root(root))

    assert json.loads((view / "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    assert json.loads((view / "hf_source.json").read_text(encoding="utf-8"))["model_type"] == "qwen3_moe"
    for module in (TEXT_ENCODER, MOE_LLM):
        assert json.loads((view / module / "config.json").read_text(encoding="utf-8"))["model_type"] == module
    # A view holds configs and assets, never weights.
    assert not list(view.rglob("*.safetensors"))


def test_omni_model_loads_every_module_weight_from_the_hf_root(hf_root):
    """Load is per-expert on disk, fused in the module — the converter has to bridge it."""
    root, on_disk = hf_root
    assert any("experts.0.gate_proj" in key for key in on_disk), "the fixture must be per-expert on disk"

    model = OmniModel.from_pretrained(str(root), device_map="cpu", attn_implementation="eager")

    backbone = model.get_submodule(MOE_LLM)
    fused = backbone.language_model.layers[0].mlp.experts
    assert hasattr(fused, "gate_up_proj")

    # Fused values must equal the per-expert source stacked the way the converter does.
    hidden, inter = 16, 8
    expected_gate_up = torch.stack(
        [
            torch.cat(
                [
                    on_disk[f"model.layers.0.mlp.experts.{j}.gate_proj.weight"],
                    on_disk[f"model.layers.0.mlp.experts.{j}.up_proj.weight"],
                ],
                dim=0,
            )
            for j in range(4)
        ]
    )
    assert expected_gate_up.shape == (4, 2 * inter, hidden)
    assert torch.equal(fused.gate_up_proj.data, expected_gate_up)

    text_encoder = model.get_submodule(TEXT_ENCODER)
    assert torch.equal(text_encoder.embed_tokens.weight.data, on_disk["model.embed_tokens.weight"])


def test_an_untied_head_lands_in_the_text_encoder(hf_root):
    root, on_disk = hf_root
    assert "lm_head.weight" in on_disk

    model = OmniModel.from_pretrained(str(root), device_map="cpu", attn_implementation="eager")
    text_encoder = model.get_submodule(TEXT_ENCODER)

    assert torch.equal(text_encoder.lm_head.weight.data, on_disk["lm_head.weight"])


def test_a_tied_head_stores_no_separate_projection(tmp_path: Path):
    """A tied release may still write ``lm_head.weight``; the module keeps one tensor."""
    root = tmp_path / "Qwen3-MoE-tied"
    _write_hf_checkpoint(root, tied=True)

    model = OmniModel.from_pretrained(str(root), device_map="cpu", attn_implementation="eager")
    text_encoder = model.get_submodule(TEXT_ENCODER)

    assert text_encoder.lm_head is None or text_encoder.config.tie_word_embeddings


def test_export_writes_back_the_source_layout(hf_root, tmp_path):
    root, on_disk = hf_root
    model = OmniModel.from_pretrained(str(root), device_map="cpu", attn_implementation="eager")
    text_encoder = model.get_submodule(TEXT_ENCODER)
    with torch.no_grad():
        text_encoder.embed_tokens.weight.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {TEXT_ENCODER: (text_encoder, None)}, str(out))

    exported = load_file(str(out / "model.safetensors"))
    # Same keys as the source: the expert keys no live module produced are carried.
    assert set(exported) == set(on_disk)
    assert torch.equal(exported["model.embed_tokens.weight"], on_disk["model.embed_tokens.weight"] + 1.0)
    for key, value in on_disk.items():
        if key == "model.embed_tokens.weight":
            continue
        assert torch.equal(exported[key], value), key
        assert exported[key].dtype == value.dtype, key


def test_export_splits_fused_experts_back_to_per_expert_keys(hf_root, tmp_path):
    """A live backbone writes back in the source's per-expert layout.

    The loader fuses ``experts.{j}.{gate,up,down}_proj`` into
    ``experts.gate_up_proj`` / ``experts.down_proj``; export has to run that
    backwards or the fused tensor has no source key to land on. The round trip
    must also carry trained values — exporting the source's stale experts in place
    of updated ones would be worse than failing.
    """
    root, on_disk = hf_root
    model = OmniModel.from_pretrained(str(root), device_map="cpu", attn_implementation="eager")
    backbone = model.get_submodule(MOE_LLM)

    experts = backbone.language_model.layers[0].mlp.experts
    with torch.no_grad():
        experts.gate_up_proj.add_(1.0)

    out = tmp_path / "export"
    save_hf_source_checkpoint(model.config._hf_source, {MOE_LLM: (backbone, None)}, str(out))
    exported = load_file(str(out / "model.safetensors"))

    assert set(exported) == set(on_disk)
    num_experts = _qwen3_moe_config().num_experts
    for j in range(num_experts):
        for proj in ("gate_proj", "up_proj"):
            key = f"model.layers.0.mlp.experts.{j}.{proj}.weight"
            assert torch.allclose(exported[key], on_disk[key] + 1.0), key
        down = f"model.layers.0.mlp.experts.{j}.down_proj.weight"
        assert torch.equal(exported[down], on_disk[down]), down
    # Everything the module did not produce is carried verbatim.
    for key, value in on_disk.items():
        if "layers.0.mlp.experts" in key:
            continue
        assert torch.equal(exported[key], value), key


def test_offline_convert_honours_the_attention_backend_override(hf_root, tmp_path):
    """A CPU conversion host must be able to ask for ``eager``.

    Building a module instantiates the patched attention, and the config default is
    ``veomni_flash_attention_2`` — so on a host without flash-attn the split dies
    before any weights are copied. ``convert_model.py --extra attn_implementation=eager``
    is the escape hatch and ``convert_with_hf_layout`` must forward it.
    ``tests/seed_omni/conftest.py`` autouses the CPU attention-kernel guard, so a
    discarded override fails here rather than on CI.
    """
    from veomni.models.seed_omni.utils.convert_registry import convert_checkpoint

    root, _ = hf_root
    out = tmp_path / "split"
    convert_checkpoint(str(root), str(out), attn_implementation="eager")

    saved = json.loads((out / MOE_LLM / "config.json").read_text(encoding="utf-8"))
    # Recorded on the module config, which is what the split carries forward.
    assert saved["ops_implementation"]["attn_implementation"] == "eager"

    # And it is what the reloaded module selects, which is the point of recording it.
    from veomni.models.seed_omni import OMNI_MODEL_REGISTRY
    from veomni.models.seed_omni.modules.qwen3_moe.llm.configuration import Qwen3MoeLlmConfig

    reloaded = OMNI_MODEL_REGISTRY[MOE_LLM]()(Qwen3MoeLlmConfig.from_pretrained(out / MOE_LLM))
    assert reloaded.config.text_config._attn_implementation == "eager"
