"""How a BAGEL checkpoint splits into SeedOmni modules.

Registered under ``OMNI_HF_LAYOUT_REGISTRY["bagel"]``. Point ``model.model_path``
at an upstream BAGEL checkpoint to train / infer from it directly, or split it
offline with ``scripts/seed_omni/convert_model.py``. Graphs come from
``configs/seed_omni/Bagel/bagel_7b_mot/``.

Modules::

    bagel_vae             # encoder.* + decoder.* (from ae.safetensors, float32) + image_processor
    bagel_siglip_navit    # vit_model.* + connector.* + vit_pos_embed.* + image_processor
    bagel_text_encoder    # language_model.model.embed_tokens.* + language_model.lm_head.* + tokenizer
    bagel_flow_connector  # vae2llm.* + llm2vae.* + time_embedder.* + latent_pos_embed.*
    bagel_qwen2_mot       # the rest of language_model.* (MoT decoder layers + norms)
"""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PretrainedConfig

from .....distributed.parallel_state import use_parallel_state
from ...utils.hf_layout import OMNI_HF_LAYOUT_REGISTRY, OmniHFLayout, OmniHFModuleLayout
from .flow_connector.configuration import BagelFlowConnectorConfig
from .qwen2_mot.checkpoint_conversion import BagelQwen2MoTCheckpointTensorConverter
from .qwen2_mot.configuration import BagelQwen2MoTConfig
from .siglip_navit.configuration import BagelSiglipNavitConfig
from .siglip_navit.processing import BagelSiglipNavitProcessor
from .text_encoder.configuration import BagelTextEncoderConfig
from .vae.configuration import BagelVAEConfig
from .vae.processing import BagelVAEProcessor


BAGEL_GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "image_height": 1024,
    "image_width": 1024,
    "latent_downsample": 16,
    "num_timesteps": 50,
    "timestep_shift": 3.0,
    "cfg_text_scale": 1.0,
    "cfg_img_scale": 1.0,
    "cfg_interval": [0.0, 1.0],
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
    "enable_taylorseer": False,
}

_BAGEL_SHARD_FILES = ("ae.safetensors", "ema.safetensors")
_STAGED_WEIGHTS_CACHE: dict[str, str] = {}

_VAE_CONFIG_KEYS = (
    "resolution",
    "in_channels",
    "downsample",
    "ch",
    "out_ch",
    "ch_mult",
    "num_res_blocks",
    "z_channels",
    "scale_factor",
    "shift_factor",
    "max_image_size",
    "min_image_size",
    "image_stride",
    "max_pixels",
    "image_mean",
    "image_std",
    "freeze",
)

_LLM_IGNORED_KEYS = frozenset(
    {
        "architectures",
        "dtype",
        "freeze_und",
        "layer_module",
        "model_type",
        "qk_norm",
        "torch_dtype",
    }
)


def _read_json_if_exists(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_bagel_hf_config(source_path: str) -> PretrainedConfig:
    """Load BAGEL root ``config.json``, merging ``llm_config.json`` / ``vit_config.json`` fallbacks."""
    config_dict, _ = PretrainedConfig.get_config_dict(source_path)
    if "llm_config" not in config_dict:
        llm_dict = _read_json_if_exists(os.path.join(source_path, "llm_config.json"))
        if llm_dict is not None:
            config_dict["llm_config"] = llm_dict
    if "vit_config" not in config_dict:
        vit_dict = _read_json_if_exists(os.path.join(source_path, "vit_config.json"))
        if vit_dict is not None:
            config_dict["vit_config"] = vit_dict

    # Official BAGEL checkpoints ship ``max_latent_size: 32`` in ``config.json`` while
    # ``ema.safetensors`` stores a 64x64 (4096-row) ``latent_pos_embed.pos_embed``.
    ema_path = os.path.join(source_path, "ema.safetensors")
    if os.path.isfile(ema_path):
        with safe_open(ema_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "latent_pos_embed.pos_embed" in keys:
                num_patches = int(f.get_slice("latent_pos_embed.pos_embed").get_shape()[0])
                config_dict["max_latent_size"] = math.isqrt(num_patches)
    else:
        config_dict.setdefault("max_latent_size", 64)

    return PretrainedConfig.from_dict(config_dict)


def _llm_dict(hf_config: Any) -> dict[str, Any]:
    llm = getattr(hf_config, "llm_config", None)
    return llm.to_dict() if hasattr(llm, "to_dict") else dict(llm or {})


def _vit_dict(hf_config: Any) -> dict[str, Any]:
    vit = getattr(hf_config, "vit_config", None)
    return vit.to_dict() if hasattr(vit, "to_dict") else dict(vit or {})


def _vae_dict(hf_config: Any) -> dict[str, Any]:
    vae = getattr(hf_config, "vae_config", None)
    return vae.to_dict() if hasattr(vae, "to_dict") else dict(vae or {})


def _vae_config(hf_config: Any) -> BagelVAEConfig:
    vae_raw = _vae_dict(hf_config)
    kwargs = {key: vae_raw[key] for key in _VAE_CONFIG_KEYS if key in vae_raw}
    return BagelVAEConfig(**kwargs, dtype="float32")


def _siglip_config(hf_config: Any) -> BagelSiglipNavitConfig:
    llm_raw = _llm_dict(hf_config)
    vit_raw = _vit_dict(hf_config)
    image_size = int(vit_raw["image_size"])
    patch_size = int(vit_raw["patch_size"])
    # Official BAGEL drops the final SigLIP encoder layer relative to vit_config.json.
    num_hidden_layers = int(vit_raw["num_hidden_layers"]) - 1
    vit_max_num_patch_per_side = int(getattr(hf_config, "vit_max_num_patch_per_side", image_size // patch_size))
    connector_act = str(getattr(hf_config, "connector_act", "gelu_pytorch_tanh"))
    return BagelSiglipNavitConfig(
        hidden_size=int(vit_raw["hidden_size"]),
        output_size=int(llm_raw["hidden_size"]),
        image_size=image_size,
        intermediate_size=int(vit_raw["intermediate_size"]),
        num_attention_heads=int(vit_raw["num_attention_heads"]),
        num_hidden_layers=num_hidden_layers,
        num_channels=int(vit_raw.get("num_channels", 3)),
        patch_size=patch_size,
        connector_act=connector_act,
        vit_max_num_patch_per_side=vit_max_num_patch_per_side,
        rope=bool(vit_raw.get("rope", False)),
        dtype="bfloat16",
    )


def _text_encoder_config(hf_config: Any) -> BagelTextEncoderConfig:
    llm_raw = _llm_dict(hf_config)
    return BagelTextEncoderConfig(
        vocab_size=int(llm_raw["vocab_size"]),
        hidden_size=int(llm_raw["hidden_size"]),
        tie_word_embeddings=bool(llm_raw.get("tie_word_embeddings", False)),
        lm_head_bias=False,
        dtype="bfloat16",
    )


def _flow_connector_config(hf_config: Any) -> BagelFlowConnectorConfig:
    llm_raw = _llm_dict(hf_config)
    vae_raw = _vae_dict(hf_config)
    latent_patch_size = int(getattr(hf_config, "latent_patch_size", 2))
    z_channels = int(vae_raw.get("z_channels", 16))
    max_latent_size = int(getattr(hf_config, "max_latent_size", 64))
    timestep_shift = float(getattr(hf_config, "timestep_shift", 1.0))
    return BagelFlowConnectorConfig(
        hidden_size=int(llm_raw["hidden_size"]),
        z_channels=z_channels,
        latent_patch_size=latent_patch_size,
        patch_latent_dim=latent_patch_size * latent_patch_size * z_channels,
        max_latent_size=max_latent_size,
        timestep_shift=timestep_shift,
        dtype="bfloat16",
    )


def _qwen2_mot_config(hf_config: Any) -> BagelQwen2MoTConfig:
    llm_raw = _llm_dict(hf_config)
    kwargs = {key: value for key, value in llm_raw.items() if key not in _LLM_IGNORED_KEYS}
    return BagelQwen2MoTConfig(
        **kwargs,
        qk_norm=bool(llm_raw.get("qk_norm", True)),
        layer_module=str(llm_raw.get("layer_module", "Qwen2MoTDecoderLayer")),
        freeze_und=bool(llm_raw.get("freeze_und", False)),
        dtype="bfloat16",
    )


def _text_encoder_assets(model_path: str) -> dict[str, Any]:
    return {"tokenizer": AutoTokenizer.from_pretrained(model_path)}


def _siglip_assets(model_path: str) -> dict[str, Any]:
    hf_config = _load_bagel_hf_config(model_path)
    return {"image_processor": BagelSiglipNavitProcessor.from_config(_siglip_config(hf_config))}


def _vae_assets(model_path: str) -> dict[str, Any]:
    hf_config = _load_bagel_hf_config(model_path)
    return {"image_processor": BagelVAEProcessor.from_config(_vae_config(hf_config))}


class BagelHFLayout(OmniHFLayout):
    """BAGEL HF checkpoint layout across ``ema.safetensors`` (BF16) and ``ae.safetensors`` (FP32)."""

    def source_weight_files(self, source_path: str) -> dict[str, str]:
        """Map every tensor key in ``ae.safetensors`` and ``ema.safetensors`` to its filename."""
        weight_map: dict[str, str] = {}
        for filename in _BAGEL_SHARD_FILES:
            shard_path = os.path.join(source_path, filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing BAGEL safetensors file: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        raise ValueError(
                            f"Duplicate BAGEL checkpoint key {key!r} in {weight_map[key]} and {filename}."
                        )
                    weight_map[key] = filename
        return weight_map

    def resolve_weights_path(self, source_path: str) -> str:
        """Stage a standard HF index view when ``source_path`` has no ``model.safetensors.index.json``."""
        key = os.path.realpath(source_path)
        index_path = os.path.join(key, "model.safetensors.index.json")
        single_path = os.path.join(key, "model.safetensors")
        if os.path.isfile(index_path) and not os.path.exists(single_path):
            return key

        staged = _STAGED_WEIGHTS_CACHE.get(key)
        if staged is None or not os.path.isdir(staged):
            staged = tempfile.mkdtemp(prefix="veomni_bagel_weights_")
            atexit.register(shutil.rmtree, staged, True)
            _STAGED_WEIGHTS_CACHE[key] = staged

        weight_map = self.source_weight_files(key)
        for filename in _BAGEL_SHARD_FILES:
            link_path = os.path.join(staged, filename)
            if os.path.lexists(link_path):
                os.remove(link_path)
            os.symlink(os.path.join(key, filename), link_path)
        with open(os.path.join(staged, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
            f.write("\n")
        return staged

    def extract_module_save_state(
        self,
        module_name: str,
        model: torch.nn.Module,
        parallel_state: Any = None,
    ) -> Mapping[str, torch.Tensor]:
        """Split fused ``qkv_proj_{und,gen}`` on ``bagel_qwen2_mot`` back to upstream Q/K/V keys."""
        if module_name != "bagel_qwen2_mot":
            return super().extract_module_save_state(module_name, model, parallel_state)
        ps_ctx = use_parallel_state(parallel_state) if parallel_state is not None else nullcontext()
        converter = BagelQwen2MoTCheckpointTensorConverter()
        with ps_ctx:
            return {name: tensor.detach().cpu() for name, tensor in converter.export_weights(model)}

    def build_fqn_to_index_mapping(self, weight_files: Mapping[str, str]) -> dict[str, int]:
        """Assign 1-based shard indices corresponding to ``_BAGEL_SHARD_FILES``."""
        index_by_name = {name: idx for idx, name in enumerate(_BAGEL_SHARD_FILES, start=1)}
        return {key: index_by_name[filename] for key, filename in weight_files.items()}

    def finalize_exported_shards(
        self,
        source_path: str,
        save_path: str,
        weight_files: Mapping[str, str],
    ) -> None:
        """Rename consolidated ``model-0000X-of-00002.safetensors`` back to ``ae.safetensors`` / ``ema.safetensors``."""
        del weight_files
        total = len(_BAGEL_SHARD_FILES)
        shard_to_target = {
            f"model-{idx:05d}-of-{total:05d}.safetensors": filename
            for idx, filename in enumerate(_BAGEL_SHARD_FILES, start=1)
        }
        for generated, target in shard_to_target.items():
            os.replace(os.path.join(save_path, generated), os.path.join(save_path, target))

        save_index = os.path.join(save_path, "model.safetensors.index.json")
        source_index = os.path.join(source_path, "model.safetensors.index.json")
        if os.path.isfile(source_index):
            with open(save_index, encoding="utf-8") as f:
                index_payload = json.load(f)
            index_payload["weight_map"] = {
                key: shard_to_target.get(filename, filename)
                for key, filename in index_payload.get("weight_map", {}).items()
            }
            with open(save_index, "w", encoding="utf-8") as f:
                json.dump(index_payload, f, indent=2)
                f.write("\n")
        elif os.path.isfile(save_index):
            os.remove(save_index)


BAGEL_HF_LAYOUT = BagelHFLayout(
    modules={
        "bagel_vae": OmniHFModuleLayout(
            key_prefixes=(("encoder.", "encoder."), ("decoder.", "decoder.")),
            build_config=_vae_config,
            build_assets=_vae_assets,
        ),
        "bagel_siglip_navit": OmniHFModuleLayout(
            key_prefixes=(
                ("vit_model.", ""),
                ("connector.", "connector."),
                ("vit_pos_embed.", "vit_pos_embed."),
            ),
            build_config=_siglip_config,
            build_assets=_siglip_assets,
        ),
        "bagel_text_encoder": OmniHFModuleLayout(
            key_prefixes=(
                ("language_model.model.embed_tokens.", "embed_tokens."),
                ("language_model.lm_head.", "lm_head."),
            ),
            build_config=_text_encoder_config,
            build_assets=_text_encoder_assets,
        ),
        "bagel_flow_connector": OmniHFModuleLayout(
            key_prefixes=(
                ("vae2llm.", "vae2llm."),
                ("llm2vae.", "llm2vae."),
                ("time_embedder.", "time_embedder."),
                ("latent_pos_embed.", "latent_pos_embed."),
            ),
            build_config=_flow_connector_config,
        ),
        "bagel_qwen2_mot": OmniHFModuleLayout(
            key_prefixes=(("language_model.", ""),),
            build_config=_qwen2_mot_config,
        ),
    },
    graph_dir="configs/seed_omni/Bagel/bagel_7b_mot",
    training_graphs="train/graph_train.yaml",
    generation_graphs={
        "infer_edit": "infer/graph_infer_edit.yaml",
        "infer_gen": "infer/graph_infer_gen.yaml",
        "infer_und": "infer/graph_infer_und.yaml",
    },
    infer_type="infer_und",
    generation_kwargs=deepcopy(BAGEL_GENERATION_KWARGS),
    load_hf_config=_load_bagel_hf_config,
)


@OMNI_HF_LAYOUT_REGISTRY.register("bagel")
def _register_bagel_hf_layout(source_path: str | None = None) -> BagelHFLayout:
    del source_path
    return BAGEL_HF_LAYOUT


__all__ = [
    "BAGEL_GENERATION_KWARGS",
    "BAGEL_HF_LAYOUT",
    "BagelHFLayout",
]
