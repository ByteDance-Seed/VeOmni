"""Split a BAGEL checkpoint into SeedOmni modules.

Registered under ``OMNI_CONVERT_REGISTRY["bagel"]``; :func:`convert_checkpoint`
writes the omni checkpoint. Graphs come from ``configs/seed_omni/Bagel/bagel_7b_mot/``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights
from veomni.models.seed_omni.utils.convert_registry import (
    OMNI_CONVERT_REGISTRY,
    attach_module_assets,
    load_family_graphs,
)


BAGEL_GENERATION_KWARGS = {
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prefixed_safetensors(
    path: Path,
    remap: Callable[[str], str | None],
    consumed_keys: set[str] | None = None,
) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            target_key = remap(key)
            if target_key is None:
                continue
            if consumed_keys is not None:
                consumed_keys.add(key)
            state_dict[target_key] = f.get_tensor(key)
    return state_dict


def _safetensor_keys(path: Path) -> set[str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        return set(f.keys())


def _assert_no_unhandled_keys(path: Path, consumed_keys: set[str], ignored_keys: set[str]) -> None:
    unhandled = sorted(_safetensor_keys(path) - consumed_keys - ignored_keys)
    if unhandled:
        preview = ", ".join(unhandled[:20])
        raise RuntimeError(f"Unhandled source keys in {path}: {preview}")


def _instantiate(model_type: str, config: Any):
    from veomni.models.seed_omni.modules import OMNI_MODEL_REGISTRY

    model_cls = OMNI_MODEL_REGISTRY[model_type]()
    with no_init_weights(), init_empty_weights():
        return model_cls._from_config(config)


def _materialize_allowed_missing_parameters(model: Any, missing: list[str]) -> None:
    for key in missing:
        if not key.endswith(".pos_embed"):
            continue
        module_name, _, param_name = key.rpartition(".")
        module = model.get_submodule(module_name)
        parameter = getattr(module, param_name)
        setattr(
            module,
            param_name,
            torch.nn.Parameter(torch.empty(tuple(parameter.shape)), requires_grad=parameter.requires_grad),
        )
        if not hasattr(module, "_init_weights"):
            raise RuntimeError(f"Cannot initialize deterministic missing parameter: {key}")
        module._init_weights()


def _build_module(
    model_type: str,
    config: Any,
    state_dict: dict[str, Any],
    *,
    allowed_missing: set[str] | None = None,
) -> Any:
    model = _instantiate(model_type, config)
    allowed_missing = set() if allowed_missing is None else allowed_missing
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = list(unexpected)
    missing = list(missing)
    unexpected_missing = [key for key in missing if key not in allowed_missing]
    if unexpected_missing or unexpected:
        raise RuntimeError(f"{model_type} load mismatch: missing={unexpected_missing}, unexpected={unexpected}")
    _materialize_allowed_missing_parameters(model, [key for key in missing if key in allowed_missing])
    print(f"[bagel] extracted {model_type}")
    return model


def convert_bagel_checkpoint(
    model_path: str,
    *,
    max_latent_size: int = 64,
    **kwargs: Any,
) -> dict[str, Any]:
    """Split an upstream BAGEL checkpoint into five SeedOmni modules."""
    training_graphs, generation_graphs = load_family_graphs(
        "configs/seed_omni/Bagel/bagel_7b_mot",
        training="train/graph_train.yaml",
        generation={
            "infer_edit": "infer/graph_infer_edit.yaml",
            "infer_gen": "infer/graph_infer_gen.yaml",
            "infer_und": "infer/graph_infer_und.yaml",
        },
        training_graph=kwargs.pop("training_graph", None),
        generation_graph=kwargs.pop("generation_graph", None),
    )
    del kwargs
    max_latent_size = int(max_latent_size)

    model_root = Path(model_path)

    ema_path = model_root / "ema.safetensors"
    ae_path = model_root / "ae.safetensors"
    if not ema_path.exists():
        raise FileNotFoundError(f"Missing BAGEL EMA weights: {ema_path}")
    if not ae_path.exists():
        raise FileNotFoundError(f"Missing BAGEL VAE weights: {ae_path}")

    consumed_ema_keys: set[str] = set()
    consumed_ae_keys: set[str] = set()

    llm_dict = _read_json(model_root / "llm_config.json")
    vit_dict = _read_json(model_root / "vit_config.json")
    hidden_size = int(llm_dict["hidden_size"])
    vit_hidden_size = int(vit_dict["hidden_size"])

    from veomni.models.seed_omni.modules import OMNI_CONFIG_REGISTRY

    text_cfg_cls = OMNI_CONFIG_REGISTRY["bagel_text_encoder"]()
    text_cfg = text_cfg_cls(
        vocab_size=int(llm_dict["vocab_size"]),
        hidden_size=hidden_size,
        tie_word_embeddings=False,
        lm_head_bias=False,
    )
    text_state = _load_prefixed_safetensors(
        ema_path,
        lambda key: {
            "language_model.model.embed_tokens.weight": "embed_tokens.weight",
            "language_model.lm_head.weight": "lm_head.weight",
        }.get(key),
        consumed_ema_keys,
    )
    text_encoder = _build_module("bagel_text_encoder", text_cfg, text_state)
    attach_module_assets(text_encoder, tokenizer=AutoTokenizer.from_pretrained(model_path))

    qwen_cfg_cls = OMNI_CONFIG_REGISTRY["bagel_qwen2_mot"]()
    qwen_cfg = qwen_cfg_cls(
        **llm_dict,
        qk_norm=True,
        layer_module="Qwen2MoTDecoderLayer",
        freeze_und=False,
    )
    qwen_state = _load_prefixed_safetensors(
        ema_path,
        lambda key: (
            key.removeprefix("language_model.")
            if key.startswith("language_model.")
            and not key.startswith("language_model.model.embed_tokens.")
            and not key.startswith("language_model.lm_head.")
            else None
        ),
        consumed_ema_keys,
    )
    qwen2_mot = _build_module("bagel_qwen2_mot", qwen_cfg, qwen_state)

    siglip_cfg_cls = OMNI_CONFIG_REGISTRY["bagel_siglip_navit"]()
    siglip_cfg = siglip_cfg_cls(
        hidden_size=vit_hidden_size,
        output_size=hidden_size,
        image_size=int(vit_dict["image_size"]),
        intermediate_size=int(vit_dict["intermediate_size"]),
        num_attention_heads=int(vit_dict["num_attention_heads"]),
        # Official inference/training uses one fewer active layer than vit_config.json.
        num_hidden_layers=int(vit_dict["num_hidden_layers"]) - 1,
        patch_size=int(vit_dict["patch_size"]),
        connector_act="gelu_pytorch_tanh",
        vit_max_num_patch_per_side=int(vit_dict["image_size"]) // int(vit_dict["patch_size"]),
        rope=False,
    )
    siglip_state = _load_prefixed_safetensors(
        ema_path,
        lambda key: (
            key.removeprefix("vit_model.")
            if key.startswith("vit_model.")
            else key
            if key.startswith("connector.")
            else None
        ),
        consumed_ema_keys,
    )
    siglip = _build_module(
        "bagel_siglip_navit",
        siglip_cfg,
        siglip_state,
        allowed_missing={"vit_pos_embed.pos_embed"},
    )
    from veomni.models.seed_omni.modules.bagel.siglip_navit.processing import BagelSiglipNavitProcessor

    attach_module_assets(siglip, image_processor=BagelSiglipNavitProcessor.from_config(siglip_cfg))

    flow_cfg_cls = OMNI_CONFIG_REGISTRY["bagel_flow_connector"]()
    patch_latent_dim = 2 * 2 * 16
    flow_cfg = flow_cfg_cls(
        hidden_size=hidden_size,
        patch_latent_dim=patch_latent_dim,
        max_latent_size=max_latent_size,
        timestep_shift=1.0,
    )
    flow_prefixes = ("vae2llm.", "llm2vae.", "time_embedder.")
    flow_state = _load_prefixed_safetensors(
        ema_path,
        lambda key: key if key.startswith(flow_prefixes) else None,
        consumed_ema_keys,
    )
    flow_connector = _build_module(
        "bagel_flow_connector",
        flow_cfg,
        flow_state,
        allowed_missing={"latent_pos_embed.pos_embed"},
    )

    vae_cfg_cls = OMNI_CONFIG_REGISTRY["bagel_vae"]()
    vae_cfg = vae_cfg_cls()
    vae_state = _load_prefixed_safetensors(ae_path, lambda key: key, consumed_ae_keys)
    vae = _build_module("bagel_vae", vae_cfg, vae_state)
    from veomni.models.seed_omni.modules.bagel.vae.processing import BagelVAEProcessor

    attach_module_assets(vae, image_processor=BagelVAEProcessor.from_config(vae_cfg))

    # Position embeddings are deterministic sin-cos buffers regenerated from config.
    ignored_ema_keys = {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
    _assert_no_unhandled_keys(ema_path, consumed_ema_keys, ignored_ema_keys)
    _assert_no_unhandled_keys(ae_path, consumed_ae_keys, set())

    return {
        "modules": {
            "bagel_vae": vae,
            "bagel_siglip_navit": siglip,
            "bagel_text_encoder": text_encoder,
            "bagel_flow_connector": flow_connector,
            "bagel_qwen2_mot": qwen2_mot,
        },
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": "infer_und",
        "generation_kwargs": deepcopy(BAGEL_GENERATION_KWARGS),
    }


@OMNI_CONVERT_REGISTRY.register("bagel")
def _register_bagel_convert():
    return convert_bagel_checkpoint


__all__ = ["convert_bagel_checkpoint"]
