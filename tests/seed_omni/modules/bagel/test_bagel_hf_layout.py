"""Running the ``bagel`` SeedOmni family straight off its upstream HF checkpoint.

A tiny BAGEL checkpoint with ``ema.safetensors`` (BF16) and ``ae.safetensors``
(FP32) stands in for the upstream one. The tests check that:
- :data:`BAGEL_HF_LAYOUT` routes every source key (including split Q/K/V,
  ``vit_pos_embed.pos_embed``, and ``latent_pos_embed.pos_embed``) onto the
  module that owns it and maps back cleanly;
- ``OmniModel.from_pretrained`` reads the upstream root whether or not
  ``model.safetensors.index.json`` is present, fuses Q/K/V on the accelerated
  MoT backbone, and preserves per-module dtypes (FP32 VAE, BF16 everything
  else);
- :func:`convert_checkpoint` produces a split checkpoint whose weights match
  the direct runtime load tensor-for-tensor; and
- :func:`save_hf_source_checkpoint` splits fused ``qkv_proj_{und,gen}`` back
  into upstream ``{q,k,v}_proj`` / ``{q,k,v}_proj_moe_gen`` keys and writes
  ``ae.safetensors`` and ``ema.safetensors`` back in the source layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.omni_arguments_types import (
    OmniModuleRuntimeArguments,
    _check_hf_source_exportable,
    _validate_hf_view_modules,
)
from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights
from veomni.models.seed_omni import OMNI_ACCELERATED_MODEL_REGISTRY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.flow_connector.modeling import BagelFlowConnector
from veomni.models.seed_omni.modules.bagel.hf_layout import (
    BAGEL_GENERATION_KWARGS,
    BAGEL_HF_LAYOUT,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import BagelQwen2MoT
from veomni.models.seed_omni.modules.bagel.siglip_navit.modeling import BagelSiglipNavit
from veomni.models.seed_omni.modules.bagel.text_encoder.modeling import BagelTextEncoder
from veomni.models.seed_omni.modules.bagel.vae.modeling import BagelVAE
from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY, convert_checkpoint
from veomni.models.seed_omni.utils.hf_layout import (
    attach_hf_source_converter,
    has_hf_layout,
    read_hf_source,
    resolve_omni_checkpoint_root,
    save_hf_source_checkpoint,
)


VAE = "bagel_vae"
SIGLIP = "bagel_siglip_navit"
TEXT_ENCODER = "bagel_text_encoder"
FLOW = "bagel_flow_connector"
MOT = "bagel_qwen2_mot"
ALL_MODULES = (VAE, SIGLIP, TEXT_ENCODER, FLOW, MOT)

pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _tiny_bagel_config_dict() -> dict:
    return {
        "architectures": ["BagelForConditionalGeneration"],
        "model_type": "bagel",
        "visual_gen": True,
        "visual_und": True,
        "llm_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "tie_word_embeddings": False,
            "qk_norm": True,
            "torch_dtype": "bfloat16",
        },
        "vit_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 16,
            "image_size": 28,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,  # 2 - 1 = 1 active layer
            "patch_size": 14,
            "num_channels": 3,
        },
        "vae_config": {
            "resolution": 32,
            "in_channels": 3,
            "downsample": 8,
            "ch": 32,
            "out_ch": 3,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "z_channels": 4,
            "max_image_size": 64,
            "min_image_size": 32,
            "image_stride": 16,
        },
        "latent_patch_size": 2,
        "max_latent_size": 4,
        "vit_max_num_patch_per_side": 2,
        "connector_act": "gelu_pytorch_tanh",
        "timestep_shift": 1.0,
        "torch_dtype": "bfloat16",
    }


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<s>": 2,
        "</s>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
        "<|vision_start|>": 6,
        "<|vision_end|>": 7,
        "a": 8,
        "b": 9,
    }
    for i in range(len(vocab), 32):
        vocab[f"tok_{i}"] = i
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"],
    ).save_pretrained(path)


def _write_bagel_hf_checkpoint(
    root: Path,
    *,
    with_index: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    root.mkdir(parents=True)
    cfg_dict = _tiny_bagel_config_dict()
    (root / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    (root / "llm_config.json").write_text(json.dumps(cfg_dict["llm_config"], indent=2), encoding="utf-8")
    (root / "vit_config.json").write_text(json.dumps(cfg_dict["vit_config"], indent=2), encoding="utf-8")
    _save_tokenizer(root)

    hf_config = BAGEL_HF_LAYOUT.read_hf_config(str(root))
    module_configs = {name: mod.build_config(hf_config) for name, mod in BAGEL_HF_LAYOUT.modules.items()}

    torch.manual_seed(0)
    vae = BagelVAE._from_config(module_configs[VAE], attn_implementation="eager").to(torch.float32)
    siglip = BagelSiglipNavit._from_config(module_configs[SIGLIP], attn_implementation="eager").to(torch.bfloat16)
    text_encoder = BagelTextEncoder._from_config(module_configs[TEXT_ENCODER], attn_implementation="eager").to(
        torch.bfloat16
    )
    flow = BagelFlowConnector._from_config(module_configs[FLOW], attn_implementation="eager").to(torch.bfloat16)
    mot = BagelQwen2MoT._from_config(module_configs[MOT], attn_implementation="eager").to(torch.bfloat16)

    ae_tensors: dict[str, torch.Tensor] = {}
    for key, value in vae.state_dict().items():
        source_key = BAGEL_HF_LAYOUT.source_key(VAE, key)
        assert source_key is not None
        ae_tensors[source_key] = value.detach().clone().contiguous()

    ema_tensors: dict[str, torch.Tensor] = {}
    for mod_name, mod in ((SIGLIP, siglip), (TEXT_ENCODER, text_encoder), (FLOW, flow), (MOT, mot)):
        for key, value in mod.state_dict().items():
            source_key = BAGEL_HF_LAYOUT.source_key(mod_name, key)
            assert source_key is not None, (mod_name, key)
            ema_tensors[source_key] = value.detach().clone().contiguous()

    save_file(ae_tensors, root / "ae.safetensors", metadata={"format": "pt"})
    save_file(ema_tensors, root / "ema.safetensors", metadata={"format": "pt"})

    if with_index:
        weight_map = {
            **dict.fromkeys(ae_tensors, "ae.safetensors"),
            **dict.fromkeys(ema_tensors, "ema.safetensors"),
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
    return ae_tensors, ema_tensors


@pytest.fixture
def hf_root(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    root = tmp_path / "BAGEL-tiny"
    ae_tensors, ema_tensors = _write_bagel_hf_checkpoint(root, with_index=True)
    return root, ae_tensors, ema_tensors


def test_bagel_registers_both_layout_and_converter():
    assert has_hf_layout("bagel")
    assert "bagel" in OMNI_CONVERT_REGISTRY.valid_keys()


def test_layout_routes_each_key_to_the_longest_claiming_prefix_and_back():
    layout = BAGEL_HF_LAYOUT
    assert layout.route("encoder.conv_in.weight") == (VAE, "encoder.conv_in.weight")
    assert layout.route("decoder.conv_out.weight") == (VAE, "decoder.conv_out.weight")
    assert layout.route("vit_model.vision_model.encoder.layers.0.mlp.fc1.weight") == (
        SIGLIP,
        "vision_model.encoder.layers.0.mlp.fc1.weight",
    )
    assert layout.route("connector.fc1.weight") == (SIGLIP, "connector.fc1.weight")
    assert layout.route("vit_pos_embed.pos_embed") == (SIGLIP, "vit_pos_embed.pos_embed")
    assert layout.route("language_model.model.embed_tokens.weight") == (TEXT_ENCODER, "embed_tokens.weight")
    assert layout.route("language_model.lm_head.weight") == (TEXT_ENCODER, "lm_head.weight")
    assert layout.route("language_model.model.layers.0.self_attn.q_proj.weight") == (
        MOT,
        "model.layers.0.self_attn.q_proj.weight",
    )
    assert layout.route("language_model.model.layers.0.self_attn.q_proj_moe_gen.weight") == (
        MOT,
        "model.layers.0.self_attn.q_proj_moe_gen.weight",
    )
    assert layout.route("vae2llm.weight") == (FLOW, "vae2llm.weight")
    assert layout.route("latent_pos_embed.pos_embed") == (FLOW, "latent_pos_embed.pos_embed")
    assert layout.route("unknown.weight") is None

    # Longest-prefix guard: backbone cannot claim text encoder keys.
    assert layout.source_key(MOT, "model.embed_tokens.weight") is None
    assert layout.source_key(MOT, "lm_head.weight") is None
    assert layout.source_key(TEXT_ENCODER, "embed_tokens.weight") == "language_model.model.embed_tokens.weight"
    assert (
        layout.source_key(MOT, "model.layers.0.self_attn.q_proj.weight")
        == "language_model.model.layers.0.self_attn.q_proj.weight"
    )


def test_an_hf_root_resolves_to_a_weight_free_split_view(hf_root):
    root, _, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    assert view != str(root)
    assert resolve_omni_checkpoint_root(root) == view
    assert resolve_omni_checkpoint_root(view) == view
    _check_hf_source_exportable(view)

    source = read_hf_source(view)
    assert source.model_type == "bagel" and source.path == os.path.realpath(root)
    root_cfg = json.loads(Path(view, "config.json").read_text(encoding="utf-8"))
    assert root_cfg["model_type"] == "omni"
    assert root_cfg["generation_kwargs"] == BAGEL_GENERATION_KWARGS
    for name in ALL_MODULES:
        assert Path(view, name, "config.json").is_file()
    assert Path(view, TEXT_ENCODER, "tokenizer.json").is_file()
    assert not list(Path(view).rglob("*.safetensors"))

    omni_cfg = OmniConfig.from_pretrained(str(root))
    assert set(omni_cfg.module_names) == set(ALL_MODULES)
    assert omni_cfg.infer_type == "infer_und"
    assert set(omni_cfg.generation_graphs) == {"infer_edit", "infer_gen", "infer_und"}


@pytest.mark.parametrize("with_index", [True, False], ids=["with_index", "without_index"])
def test_omni_model_loads_from_hf_root_with_or_without_index(tmp_path: Path, with_index: bool):
    root = tmp_path / f"BAGEL-{with_index}"
    ae_tensors, ema_tensors = _write_bagel_hf_checkpoint(root, with_index=with_index)
    all_source_tensors = {**ae_tensors, **ema_tensors}

    model = OmniModel.from_pretrained(str(root), torch_dtype="auto", device_map="cpu", attn_implementation="eager")
    layout = model.config._hf_source.layout

    # VAE stays FP32 while the other 4 modules load in BF16.
    assert {p.dtype for p in model.modules_dict[VAE].parameters()} == {torch.float32}
    for name in (SIGLIP, TEXT_ENCODER, FLOW, MOT):
        assert {p.dtype for p in model.modules_dict[name].parameters()} == {torch.bfloat16}

    compared = set()
    for name, module in model.modules_dict.items():
        for key, value in module.state_dict().items():
            source_key = layout.source_key(name, key)
            assert source_key in all_source_tensors, (name, key, source_key)
            assert torch.equal(value, all_source_tensors[source_key]), (name, key)
            compared.add(source_key)
    assert compared == set(all_source_tensors)


@pytest.mark.parametrize("with_index", [True, False], ids=["with_index", "without_index"])
def test_accelerated_mot_fuses_qkv_on_load_and_splits_back_on_export(tmp_path: Path, with_index: bool):
    root = tmp_path / f"BAGEL-export-{with_index}"
    ae_tensors, ema_tensors = _write_bagel_hf_checkpoint(root, with_index=with_index)

    view = resolve_omni_checkpoint_root(root)
    source = read_hf_source(view)
    omni_cfg = OmniConfig.from_pretrained(str(root))
    mot_cfg = omni_cfg._module_configs[MOT]
    mot_cfg.ops_implementation["attn_implementation"] = "eager"

    mot_accel_cls = OMNI_ACCELERATED_MODEL_REGISTRY[MOT]()
    with init_empty_weights():
        mot_accel = mot_accel_cls._from_config(mot_cfg, torch_dtype=torch.bfloat16, attn_implementation="eager")
    attach_hf_source_converter(mot_accel, source, MOT)
    load_model_weights(mot_accel, source.weights_path, init_device="cpu")

    # Verify fused qkv_proj_und and qkv_proj_gen match the concatenated source Q/K/V weights.
    expected_qkv_und = torch.cat(
        [
            ema_tensors["language_model.model.layers.0.self_attn.q_proj.weight"],
            ema_tensors["language_model.model.layers.0.self_attn.k_proj.weight"],
            ema_tensors["language_model.model.layers.0.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    assert torch.equal(mot_accel.model.layers[0].self_attn.qkv_proj_und.weight, expected_qkv_und)

    with torch.no_grad():
        mot_accel.model.layers[0].self_attn.qkv_proj_und.weight.add_(1.0)

    out = tmp_path / f"exported-{with_index}"
    save_hf_source_checkpoint(source, {MOT: (mot_accel, None)}, str(out))

    assert (out / "ae.safetensors").is_file()
    assert (out / "ema.safetensors").is_file()
    assert (out / "model.safetensors.index.json").is_file() == with_index

    with safe_open(out / "ae.safetensors", framework="pt") as f:
        exported_ae = {key: f.get_tensor(key) for key in f.keys()}
    with safe_open(out / "ema.safetensors", framework="pt") as f:
        exported_ema = {key: f.get_tensor(key) for key in f.keys()}

    assert set(exported_ae) == set(ae_tensors)
    assert set(exported_ema) == set(ema_tensors)
    for key, value in ae_tensors.items():
        assert exported_ae[key].dtype == torch.float32
        assert torch.equal(exported_ae[key], value), key

    modified_qkv_keys = {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
    }
    for key, value in ema_tensors.items():
        assert exported_ema[key].dtype == torch.bfloat16
        if key in modified_qkv_keys:
            assert torch.equal(exported_ema[key], value + 1.0), key
        else:
            assert torch.equal(exported_ema[key], value), key


def test_offline_convert_matches_the_runtime_load(hf_root, tmp_path):
    root, _, _ = hf_root
    out = tmp_path / "split"
    convert_checkpoint(str(root), str(out), attn_implementation="eager")
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["model_type"] == "omni"
    assert read_hf_source(out) is None

    split = OmniModel.from_pretrained(str(out), torch_dtype="auto", device_map="cpu", attn_implementation="eager")
    runtime = OmniModel.from_pretrained(str(root), torch_dtype="auto", device_map="cpu", attn_implementation="eager")
    for name in ALL_MODULES:
        expected = runtime.modules_dict[name].state_dict()
        actual = split.modules_dict[name].state_dict()
        assert expected.keys() == actual.keys(), name
        for key in expected:
            assert expected[key].dtype == actual[key].dtype, (name, key)
            assert torch.equal(expected[key], actual[key]), (name, key)


def test_hf_view_modules_must_all_come_from_the_layout(hf_root):
    root, _, _ = hf_root
    view = resolve_omni_checkpoint_root(root)
    ok = {name: OmniModuleRuntimeArguments(model_path=os.path.join(view, name)) for name in ALL_MODULES}
    _validate_hf_view_modules(view, ok)

    with pytest.raises(ValueError, match="not part of the `bagel` HF layout"):
        _validate_hf_view_modules(view, {**ok, "qwen3_llm": OmniModuleRuntimeArguments(model_path="/x")})
    with pytest.raises(ValueError, match="sets its own model_path"):
        _validate_hf_view_modules(view, {**ok, VAE: OmniModuleRuntimeArguments(model_path="/elsewhere/vae")})
