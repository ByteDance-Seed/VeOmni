"""BAGEL official reference runtime assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tests.seed_omni.parity_suite.core import resolve_torch_dtype
from tests.seed_omni.parity_suite.reference.oracles.hf_model import empty_init_context

from .vendor.data.data_utils import add_special_tokens
from .vendor.modeling.autoencoder import load_ae
from .vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .vendor.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer


@dataclass(frozen=True)
class BagelReferenceBundle:
    """Loaded official BAGEL objects consumed by reference subjects."""

    model: Bagel
    tokenizer: Qwen2Tokenizer
    new_token_ids: Mapping[str, int]
    vae_model: nn.Module | None
    device: torch.device


@dataclass(frozen=True)
class BagelAssemblyPlan:
    """Model-owned plan for assembling a BAGEL reference runtime."""

    visual_gen: bool
    visual_und: bool
    load_ae: bool
    load_weights: bool
    move_model_to_device: bool

    @classmethod
    def full(cls) -> BagelAssemblyPlan:
        return cls(
            visual_gen=True,
            visual_und=True,
            load_ae=True,
            load_weights=True,
            move_model_to_device=True,
        )

    @classmethod
    def lazy(
        cls,
        *,
        visual_und: bool = False,
        language_model: bool = False,
        flow: bool = False,
        ae: bool = False,
        visual_gen: bool = False,
    ) -> BagelAssemblyPlan:
        """Build a partial official BAGEL runtime for module-tier capture."""

        visual_gen = visual_gen or flow
        load_ae = ae or visual_gen
        return cls(
            visual_gen=visual_gen,
            visual_und=visual_und,
            load_ae=load_ae,
            load_weights=visual_und or language_model or flow,
            move_model_to_device=language_model or flow or ae,
        )


def load_full_bagel_reference_bundle(
    *,
    checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    torch_dtype: torch.dtype | str | None = None,
    load_weights: bool = True,
    init_on_meta: bool = True,
) -> BagelReferenceBundle:
    """Assemble the full official BAGEL reference runtime.

    This mirrors ``bagel-official/app.py`` for full-model inference setup:
    configs are loaded first, VAE supplies the BAGEL VAE config, model modules
    are constructed under empty-init, then tokenizer/special-token assets and
    weights are loaded.
    """

    return load_bagel_reference_bundle(
        checkpoint=checkpoint,
        plan=BagelAssemblyPlan.full(),
        device=device,
        dtype=dtype,
        torch_dtype=torch_dtype,
        load_weights=load_weights,
        init_on_meta=init_on_meta,
    )


def load_bagel_reference_bundle(
    *,
    checkpoint: str | Path,
    plan: BagelAssemblyPlan,
    device: torch.device,
    dtype: torch.dtype,
    torch_dtype: torch.dtype | str | None = None,
    load_weights: bool | None = None,
    init_on_meta: bool = True,
) -> BagelReferenceBundle:
    """Assemble a BAGEL reference runtime from a model-owned assembly plan."""

    root = Path(checkpoint)
    target_dtype = resolve_torch_dtype(torch_dtype) if torch_dtype is not None else dtype
    should_load_weights = plan.load_weights if load_weights is None else bool(load_weights)

    _ensure_default_rope_init()
    llm_config = Qwen2Config.from_json_file(str(root / "llm_config.json"))
    _apply_official_llm_config_overrides(llm_config)

    vit_config = None
    if plan.visual_und:
        vit_config = SiglipVisionConfig.from_json_file(str(root / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

    vae_model = None
    vae_config = None
    if plan.load_ae:
        vae_model, vae_config = load_ae(local_path=str(root / "ae.safetensors"))
    elif plan.visual_gen:
        raise ValueError("BAGEL visual generation reference assembly requires load_ae=True for vae_config.")

    bagel_config = BagelConfig(
        visual_gen=plan.visual_gen,
        visual_und=plan.visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with empty_init_context() if init_on_meta else nullcontext():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config) if plan.visual_und else None
        model = Bagel(language_model, vit_model, bagel_config)
        if plan.visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=init_on_meta)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(root), local_files_only=True)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    if should_load_weights:
        model = _load_bagel_weights_like_official(
            model=model,
            checkpoint=root / "ema.safetensors",
            device=device,
            dtype=target_dtype,
        )
    if plan.move_model_to_device and not _has_accelerate_hooks(model):
        if should_load_weights or not init_on_meta:
            model.to(device=device)
        else:
            model.to_empty(device=device)
    model.eval()
    if vae_model is not None:
        vae_model = vae_model.eval()

    return BagelReferenceBundle(
        model=model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_model=vae_model,
        device=device,
    )


def _apply_official_llm_config_overrides(llm_config: Qwen2Config) -> None:
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.freeze_und = False
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "bos_token_id", 0)


def _ensure_default_rope_init() -> None:
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _compute_default_rope_parameters(config: Any, device: torch.device | None = None, **kwargs: Any):
    del kwargs
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _load_bagel_weights_like_official(
    *,
    model: Bagel,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Bagel:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Reference weights not found: {checkpoint}")

    if device.type == "cuda":
        device_map: dict[str, Any] | str = infer_auto_device_map(
            model,
            max_memory=dict.fromkeys(range(torch.cuda.device_count()), "80GiB"),
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        _co_locate_official_same_device_modules(device_map)
    else:
        device_map = {"": device}

    return load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="/tmp/open_veomni_bagel_reference_offload",
        dtype=dtype,
        force_hooks=True,
        strict=False,
    )


def _co_locate_official_same_device_modules(device_map: dict[str, Any]) -> None:
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = next((device_map[name] for name in same_device_modules if name in device_map), None)
    if first_device is None:
        return
    for module_name in same_device_modules:
        if module_name in device_map:
            device_map[module_name] = first_device


def _has_accelerate_hooks(model: nn.Module) -> bool:
    return any(hasattr(module, "_hf_hook") for module in model.modules())


__all__ = [
    "BagelAssemblyPlan",
    "BagelReferenceBundle",
    "load_bagel_reference_bundle",
    "load_full_bagel_reference_bundle",
]
