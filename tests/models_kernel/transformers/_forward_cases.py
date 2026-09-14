"""Shared accelerator model-forward cases for runtime integration gates."""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass, field
from pathlib import Path

import torch

from veomni.utils.device import empty_cache


REPO_ROOT = Path(__file__).resolve().parents[3]
DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class ForwardCase:
    case_id: str
    toy_config_dir: str
    architecture: str
    kind: str
    attn_implementation: str = "eager"
    dtype: str = "float32"
    forward_attr: str | None = None
    config_overrides: dict = field(default_factory=dict)


def _toy(name: str) -> str:
    return str(REPO_ROOT / "tests" / "toy_config" / name)


SYNC_FORWARD_CASES = (
    ForwardCase("qwen3_5-text-eager", _toy("qwen3_5_toy"), "Qwen3_5ForCausalLM", "qwen3_5_text"),
    ForwardCase(
        "qwen3_5-text-fa2",
        _toy("qwen3_5_toy"),
        "Qwen3_5ForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen3_5_moe-text-fa2-fused",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen3_5_vl-sdpa",
        _toy("qwen3_5_toy"),
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_vlm_full",
        attn_implementation="sdpa",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen3_vl-fa2",
        _toy("qwen3vl_toy"),
        "Qwen3VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen3_vl_moe-fa2-fused",
        _toy("qwen3vlmoe_toy"),
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen3_omni_moe-fa2-fused",
        _toy("qwen3omni_toy"),
        "Qwen3OmniMoeForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
    ForwardCase(
        "qwen2_vl-fa2",
        _toy("qwen2vl_toy"),
        "Qwen2VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen2_5_vl-fa2",
        _toy("qwen25vl_toy"),
        "Qwen2_5_VLForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    ForwardCase(
        "qwen2_5_omni-fa2",
        _toy("qwen25omni_toy"),
        "Qwen2_5OmniForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
)


def apply_determinism() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def release_device_memory() -> None:
    gc.collect()
    empty_cache()


def _apply_qwen3_5_text_overrides(text_config) -> None:
    if getattr(text_config, "layer_types", None) is not None:
        text_config.layer_types = ["full_attention"] * len(text_config.layer_types)
    text_config._experts_implementation = "eager"


def make_config(case: ForwardCase):
    from transformers import AutoConfig

    full_config = AutoConfig.from_pretrained(case.toy_config_dir)
    if case.kind == "qwen3_5_text":
        config = copy.deepcopy(full_config.text_config)
        _apply_qwen3_5_text_overrides(config)
    elif case.kind == "qwen3_5_vlm_full":
        config = full_config
        _apply_qwen3_5_text_overrides(config.text_config)
    else:
        config = full_config

    config.architectures = [case.architecture]
    for name, value in case.config_overrides.items():
        setattr(config, name, value)
    return config


def _vision_section(config):
    if not hasattr(config, "vision_config") and not hasattr(config, "thinker_config"):
        return None, None
    root = config.thinker_config if hasattr(config, "thinker_config") else config
    vision_config = getattr(root, "vision_config", None)
    if vision_config is None:
        return None, None
    image_token_id = getattr(root, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(root, "image_token_id", None)
    return vision_config, image_token_id


def make_inputs(case: ForwardCase, config, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, dict]:
    sequence_length = 32
    generator = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(32000, (1, sequence_length), device=device, dtype=torch.long, generator=generator)
    forward_kwargs: dict = {}

    if case.kind in ("qwen3_5_text", "qwen3_5_vlm_full"):
        forward_kwargs["cu_seq_lens_q"] = torch.tensor(
            [0, sequence_length],
            dtype=torch.int32,
            device=device,
        )

    vision_config, image_token_id = _vision_section(config)
    if vision_config is None or case.kind in ("causal_lm", "qwen3_5_text"):
        return input_ids, forward_kwargs

    patch_size = getattr(vision_config, "patch_size", 14)
    temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
    in_channels = getattr(vision_config, "in_channels", getattr(vision_config, "in_chans", 3))
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)

    grid_t, grid_h, grid_w = 1, spatial_merge_size, spatial_merge_size
    num_patches = grid_t * grid_h * grid_w
    num_tokens = num_patches // (spatial_merge_size**2)
    feature_dim = in_channels * temporal_patch_size * patch_size * patch_size

    pixel_generator = torch.Generator(device=device).manual_seed(1)
    pixel_values = torch.randn(
        num_patches,
        feature_dim,
        dtype=dtype,
        device=device,
        generator=pixel_generator,
    )
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=device)

    input_ids = input_ids.clone()
    input_ids[0, :num_tokens] = image_token_id
    image_mask = input_ids == image_token_id
    video_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    forward_kwargs.update(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_mask=image_mask,
        video_mask=video_mask,
        mm_token_type_ids=image_mask.int() + 2 * video_mask.int(),
    )
    if case.kind == "omni_thinker":
        forward_kwargs["audio_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids, forward_kwargs


def forward_target(model, case: ForwardCase):
    target = model
    if case.forward_attr is not None:
        for attribute in case.forward_attr.split("."):
            target = getattr(target, attribute)
    return target
