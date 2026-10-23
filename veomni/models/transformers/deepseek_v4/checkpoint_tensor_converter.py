# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runtime checkpoint tensor converter for DeepseekV4 MoE models.

DeepSeek-V4 upstream checkpoints ship per-expert split keys with the
``w1`` / ``w2`` / ``w3`` naming convention (HF's
``conversion_mapping.py`` recipe is ``MergeModulelist(dim=0) +
Concatenate(dim=1)`` for ``w1`` / ``w3`` -> ``gate_up_proj``, plus
``MergeModulelist(dim=0)`` for ``w2`` -> ``down_proj``). VeOmni's runtime
loader needs the v5 fused layout directly, so we stack + merge per-expert
tensors as they stream in from safetensors.

    HF checkpoint format (per-expert):
        model.layers.{i}.mlp.experts.{j}.w1.weight  [I, H]   # gate
        model.layers.{i}.mlp.experts.{j}.w2.weight  [H, I]   # down
        model.layers.{i}.mlp.experts.{j}.w3.weight  [I, H]   # up

    Target v5 format (direct layout, matches DeepseekV4Experts.__init__):
        model.layers.{i}.mlp.experts.gate_up_proj  [E, 2*I, H]
        model.layers.{i}.mlp.experts.down_proj     [E, H, I]

VeOmni's runtime loader reads safetensor keys directly and does not run HF's
``WeightConverter`` objects, so this converter accepts both the raw on-disk
``w1`` / ``w2`` / ``w3`` keys and the already-renamed
``gate_proj`` / ``down_proj`` / ``up_proj`` form.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Callable, Generator

import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming, rename_source_key

from veomni.models.checkpoint.convert import ConvertedCheckpointTensor, export_weights
from veomni.utils import logging


logger = logging.get_logger(__name__)

# Matches raw DeepSeek-V4 checkpoint keys (w1/w2/w3) and post-rename
# keys (gate_proj/up_proj/down_proj), e.g.:
#   model.layers.0.mlp.experts.3.w1.weight
#   model.layers.0.mlp.experts.3.gate_proj.weight
#   model.mtp.0.mlp.experts.3.w1.weight
_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)\.weight$")
_RAW_FP4_EXPERT_WEIGHT_PATTERN = re.compile(r"^(?:model\.)?(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
_MTP_HC_HEAD_PATTERN = re.compile(r"^mtp\.(\d+)\.hc_head_(fn|base|scale)$")
_MTP_MARKOV_W1_PATTERN = re.compile(r"^mtp\.(\d+)\.markov_head\.markov_w1\.(weight|scale)$")
_MTP_MARKOV_W2_PATTERN = re.compile(r"^mtp\.(\d+)\.markov_head\.markov_w2\.(weight|scale)$")
_MTP_STAGE_NORM_PATTERN = re.compile(r"^mtp\.(\d+)\.norm\.weight$")
_PROJ_NAME_ALIASES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_PROJ_NAME_TO_W_NAME = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}
_DEEPSEEK_V4_WEIGHT_RENAMINGS = [
    transform
    for transform in (get_checkpoint_conversion_mapping("deepseek_v4") or [])
    if isinstance(transform, WeightRenaming)
]
_DEEPSEEK_V4_WEIGHT_REVERSE_RENAMINGS = [
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.scorer\.weights_proj\.",
        target_patterns=r"layers.\1.attn.indexer.weights_proj.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.q_b_proj\.",
        target_patterns=r"layers.\1.attn.indexer.wq_b.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias$",
        target_patterns=r"layers.\1.attn.indexer.compressor.ape",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_proj\.",
        target_patterns=r"layers.\1.attn.indexer.compressor.wkv.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.gate_proj\.",
        target_patterns=r"layers.\1.attn.indexer.compressor.wgate.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_norm\.",
        target_patterns=r"layers.\1.attn.indexer.compressor.norm.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.kv_norm\.",
        target_patterns=r"layers.\1.attn.compressor.norm.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.kv_proj\.",
        target_patterns=r"layers.\1.attn.compressor.wkv.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.gate_proj\.",
        target_patterns=r"layers.\1.attn.compressor.wgate.",
    ),
    WeightRenaming(
        source_patterns=r"^layers\.(\d+)\.self_attn\.compressor\.position_bias$",
        target_patterns=r"layers.\1.attn.compressor.ape",
    ),
    *[
        transform.reverse_transform()
        for transform in reversed(_DEEPSEEK_V4_WEIGHT_RENAMINGS)
        if transform.source_patterns != [r"\.norm\."]
    ],
]
_QUANTIZED_RELEASE_WEIGHT_PATTERN = re.compile(
    r"^(?:layers|mtp)\.\d+\.(?:"
    r"attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)|"
    r"attn\.indexer\.wq_b|"
    r"ffn\.shared_experts\.w[123]|"
    r"ffn\.experts\.\d+\.w[123]|"
    r"main_proj"
    r")\.weight$"
)
_FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_BASE_MODEL_KEY_PREFIXES = ("embed_tokens.", "hc_head.", "layers.", "mtp.", "norm.")
_FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")
    )
    if dtype
)
_QUANTIZED_WEIGHT_DTYPES = _FLOAT8_DTYPES + (torch.int8,)


def _is_quantized_weight(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _QUANTIZED_WEIGHT_DTYPES


def _is_mtp_checkpoint_key(name: str) -> bool:
    return name.removeprefix("model.").startswith("mtp.")


def _is_mtp_expert_prefix(prefix: str) -> bool:
    return prefix.removeprefix("model.").startswith("mtp.")


def convert_deepseek_v4_checkpoint_key(
    name: str,
    *,
    target_model_prefix: str = "model.",
    enable_mtp: bool = False,
) -> str:
    """Convert DeepSeek's inference checkpoint names to Transformers v5 names."""
    if _is_mtp_checkpoint_key(name) and not enable_mtp:
        return name
    has_model_prefix = name.startswith("model.")
    source_name = name.removeprefix("model.") if has_model_prefix else name
    if enable_mtp and source_name.startswith("mtp."):
        hc_match = _MTP_HC_HEAD_PATTERN.match(source_name)
        if hc_match is not None:
            stage_idx, part = hc_match.groups()
            return f"{target_model_prefix}mtp.{stage_idx}.hc_head.hc_{part}"
        w1_match = _MTP_MARKOV_W1_PATTERN.match(source_name)
        if w1_match is not None:
            stage_idx, suffix = w1_match.groups()
            return f"{target_model_prefix}mtp.{stage_idx}.markov_head.embed.{suffix}"
        w2_match = _MTP_MARKOV_W2_PATTERN.match(source_name)
        if w2_match is not None:
            stage_idx, suffix = w2_match.groups()
            return f"{target_model_prefix}mtp.{stage_idx}.markov_head.head.{suffix}"
        # Guard ``mtp.{i}.norm.weight`` before HF's broad ``\.norm\. -> .kv_norm.`` rule
        # (which was written for ``layers.{i}.attn.compressor.norm.weight`` where top-level
        # ``norm.weight`` had no leading dot).
        if _MTP_STAGE_NORM_PATTERN.match(source_name):
            return f"{target_model_prefix}{source_name}"

    # Upgrade legacy VeOmni scorer keys before HF's broad inference-name
    # rewrite can add a second compressor prefix.
    if re.match(r"^layers\.\d+\.self_attn\.compressor\.indexer\.weights_proj\.", source_name):
        source_name = source_name.replace(".indexer.weights_proj.", ".indexer.scorer.weights_proj.")
        return f"{target_model_prefix}{source_name}"
    converted_name, _ = rename_source_key(source_name, _DEEPSEEK_V4_WEIGHT_RENAMINGS, [])
    if converted_name.startswith(_BASE_MODEL_KEY_PREFIXES):
        return f"{target_model_prefix}{converted_name}"
    if has_model_prefix and converted_name == source_name:
        return name
    return converted_name


_MTP_REVERSE_HC_HEAD_PATTERN = re.compile(r"^mtp\.(\d+)\.hc_head\.hc_(fn|base|scale)$")
_MTP_REVERSE_MARKOV_EMBED_PATTERN = re.compile(r"^mtp\.(\d+)\.markov_head\.embed\.(weight|scale)$")
_MTP_REVERSE_MARKOV_HEAD_PATTERN = re.compile(r"^mtp\.(\d+)\.markov_head\.head\.(weight|scale)$")


def _revert_deepseek_v4_checkpoint_key(
    name: str,
    *,
    target_model_prefix: str = "model.",
    enable_mtp: bool = False,
) -> str:
    """Convert Transformers v5 parameter names back to DeepSeek release checkpoint names."""
    if _is_mtp_checkpoint_key(name) and not enable_mtp:
        return name
    source_name = name.removeprefix(target_model_prefix) if target_model_prefix else name
    if source_name.startswith("model.") and source_name.removeprefix("model.").startswith(_BASE_MODEL_KEY_PREFIXES):
        source_name = source_name.removeprefix("model.")
    if enable_mtp and source_name.startswith("mtp."):
        hc_match = _MTP_REVERSE_HC_HEAD_PATTERN.match(source_name)
        if hc_match is not None:
            stage_idx, part = hc_match.groups()
            return f"mtp.{stage_idx}.hc_head_{part}"
        w1_match = _MTP_REVERSE_MARKOV_EMBED_PATTERN.match(source_name)
        if w1_match is not None:
            stage_idx, suffix = w1_match.groups()
            return f"mtp.{stage_idx}.markov_head.markov_w1.{suffix}"
        w2_match = _MTP_REVERSE_MARKOV_HEAD_PATTERN.match(source_name)
        if w2_match is not None:
            stage_idx, suffix = w2_match.groups()
            return f"mtp.{stage_idx}.markov_head.markov_w2.{suffix}"
        if _MTP_STAGE_NORM_PATTERN.match(source_name):
            return source_name
    if "mlp.experts." in source_name:
        fields = source_name.split(".")
        if len(fields) >= 2 and fields[-2] in _PROJ_NAME_TO_W_NAME:
            fields[-2] = _PROJ_NAME_TO_W_NAME[fields[-2]]
            source_name = ".".join(fields)
    reverted_name, _ = rename_source_key(source_name, _DEEPSEEK_V4_WEIGHT_REVERSE_RENAMINGS, [])
    return reverted_name


def _extract_quant_config(config) -> tuple[dict, int, str, bool]:
    """Extract normalized quantization settings ``(quant_cfg, fp8_block_size, expert_dtype, has_quant)`` from *config*."""
    quant_cfg = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant_cfg, dict):
        quant_cfg = quant_cfg.to_dict() if hasattr(quant_cfg, "to_dict") else vars(quant_cfg)
    weight_block_size = quant_cfg.get("weight_block_size") or (128, 128)
    fp8_block_size = int(weight_block_size[0])
    expert_dtype = getattr(config, "expert_dtype", None) or quant_cfg.get("expert_dtype") or "fp8"
    has_quant = bool(
        quant_cfg.get("quant_method") == "fp8"
        or getattr(config, "expert_dtype", None) in ("fp4", "fp8")
        or quant_cfg.get("expert_dtype") in ("fp4", "fp8")
    )
    return quant_cfg, fp8_block_size, expert_dtype, has_quant


def _build_default_deepseek_v4_fqn_to_index_mapping(
    model,
    *,
    target_model_prefix: str = "model.",
    enable_mtp: bool | None = None,
) -> dict[str, int]:
    """Synthesize a 1-shard release-format ``fqn_to_index_mapping`` from *model* when none was loaded."""
    config = getattr(model, "config", None)
    if enable_mtp is None:
        base_model = getattr(model, "model", model)
        mtp_modules = getattr(base_model, "mtp", None)
        enable_mtp = bool(mtp_modules is not None and len(mtp_modules) > 0)
    _quant_cfg, fp8_block_size, expert_dtype, has_quant = _extract_quant_config(config)
    state_dict_fn = getattr(model, "state_dict", None)
    if not callable(state_dict_fn):
        return {}

    state_dict = state_dict_fn()
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    num_experts_cfg = getattr(config, "n_routed_experts", None) or getattr(config, "num_local_experts", None)
    mtp_num_experts_cfg = getattr(config, "dspark_n_routed_experts", None) or num_experts_cfg

    mapping: dict[str, int] = {}

    def _add_entry(origin_key: str, shape: tuple[int, ...]) -> None:
        """Record ``origin_key`` in the shard map and add its ``.scale`` key when quantizable."""
        mapping[origin_key] = 1
        if not has_quant or not _QUANTIZED_RELEASE_WEIGHT_PATTERN.fullmatch(origin_key) or len(shape) != 2:
            return
        if "ffn.experts." in origin_key and expert_dtype == "fp4":
            can_quantize = shape[1] % 32 == 0
        else:
            can_quantize = shape[0] % fp8_block_size == 0 and shape[1] % fp8_block_size == 0
        if can_quantize:
            mapping[f"{origin_key.removesuffix('.weight')}.scale"] = 1

    for name, param in state_dict.items():
        if tie_word_embeddings and name == "lm_head.weight":
            continue
        if _is_mtp_checkpoint_key(name) and not enable_mtp:
            continue
        shape = tuple(param.shape)
        if name.endswith(".mlp.experts.gate_up_proj") and len(shape) == 3:
            prefix = name.removesuffix(".gate_up_proj")
            experts_cfg = mtp_num_experts_cfg if _is_mtp_expert_prefix(prefix) else num_experts_cfg
            num_experts = int(experts_cfg or shape[0])
            expert_shape = (shape[1] // 2, shape[2])
            for e in range(num_experts):
                for w_name in ("w1", "w3"):
                    origin_key = _revert_deepseek_v4_checkpoint_key(
                        f"{prefix}.{e}.{w_name}.weight",
                        target_model_prefix=target_model_prefix,
                        enable_mtp=enable_mtp,
                    )
                    _add_entry(origin_key, expert_shape)
        elif name.endswith(".mlp.experts.down_proj") and len(shape) == 3:
            prefix = name.removesuffix(".down_proj")
            experts_cfg = mtp_num_experts_cfg if _is_mtp_expert_prefix(prefix) else num_experts_cfg
            num_experts = int(experts_cfg or shape[0])
            expert_shape = (shape[1], shape[2])
            for e in range(num_experts):
                origin_key = _revert_deepseek_v4_checkpoint_key(
                    f"{prefix}.{e}.w2.weight",
                    target_model_prefix=target_model_prefix,
                    enable_mtp=enable_mtp,
                )
                _add_entry(origin_key, expert_shape)
        else:
            origin_key = _revert_deepseek_v4_checkpoint_key(
                name, target_model_prefix=target_model_prefix, enable_mtp=enable_mtp
            )
            _add_entry(origin_key, shape)

    return mapping


def validate_deepseek_v4_export_quant_support(
    fqn_to_index_mapping: Mapping[str, int] | None = None,
    *,
    model: torch.nn.Module | None = None,
    target_model_prefix: str = "model.",
    should_quantize: Callable[[str], bool] | None = None,
    enable_mtp: bool = False,
) -> None:
    """Validate that SM90+ TileLang quantization is available when ``.scale`` keys will be quantized on export."""
    if fqn_to_index_mapping is None and model is not None:
        fqn_to_index_mapping = getattr(model, "_veomni_fqn_to_index_mapping", None)
        if fqn_to_index_mapping is None:
            fqn_to_index_mapping = _build_default_deepseek_v4_fqn_to_index_mapping(
                model, target_model_prefix=target_model_prefix, enable_mtp=enable_mtp
            )
    if not fqn_to_index_mapping:
        return

    needs_quant = False
    for origin in fqn_to_index_mapping:
        if (_is_mtp_checkpoint_key(origin) and not enable_mtp) or not origin.endswith(".weight"):
            continue
        scale_name = f"{origin.removesuffix('.weight')}.scale"
        if scale_name not in fqn_to_index_mapping:
            continue
        if should_quantize is not None and not should_quantize(origin):
            continue
        needs_quant = True
        break

    if needs_quant:
        from veomni.models.transformers.deepseek_v4.export_quant import _require_tilelang_sm90

        _require_tilelang_sm90()


@lru_cache(maxsize=None)
def _get_fp4_e2m1_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=device)


def _is_block_scale_tensor(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 or tensor.numel() == 1


def _weight_name_from_scale_name(name: str) -> str | None:
    if name.endswith(".scale"):
        return f"{name.removesuffix('.scale')}.weight"
    if name.endswith(".weight_scale_inv"):
        return f"{name.removesuffix('.weight_scale_inv')}.weight"
    return None


def _dequantize_scaled_weight(weight: torch.Tensor, scale: torch.Tensor, *, packed_fp4: bool = False) -> torch.Tensor:
    """Apply DeepSeek block scales to an FP8 or packed E2M1 FP4 weight."""
    scale = scale.to(device=weight.device)
    if packed_fp4:
        if weight.dtype != torch.int8:
            raise TypeError(f"Packed E2M1 FP4 weights must use int8 storage, got {weight.dtype}.")
        # DeepSeek-V4 stores two E2M1 FP4 expert values in every int8 byte.
        # Preserve the bit pattern when splitting the low and high nibbles.
        packed = weight.contiguous().view(torch.uint8)
        table = _get_fp4_e2m1_table(weight.device)
        low = table[(packed & 0x0F).long()]
        high = table[((packed >> 4) & 0x0F).long()]
        weight = torch.stack((low, high), dim=-1).flatten(-2)
    else:
        weight = weight.float()

    if scale.numel() == 1:
        return weight * scale.float()
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "DeepseekV4 quantized checkpoint weight scales must be scalar or 2D block scales; "
            f"got weight shape {tuple(weight.shape)} and scale shape {tuple(scale.shape)}."
        )
    if weight.shape[0] % scale.shape[0] != 0 or weight.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            "DeepseekV4 quantized checkpoint weight shape must be divisible by scale shape; "
            f"got weight shape {tuple(weight.shape)} and scale shape {tuple(scale.shape)}."
        )

    block_rows = weight.shape[0] // scale.shape[0]
    block_cols = weight.shape[1] // scale.shape[1]
    return (
        weight.view(scale.shape[0], block_rows, scale.shape[1], block_cols)
        .mul(scale.float().view(scale.shape[0], 1, scale.shape[1], 1))
        .reshape(weight.shape)
    )


class DeepseekV4CheckpointTensorConverter:
    """Converts per-expert split checkpoint keys to stacked & merged v5 format.

    Buffers per-expert tensors as they stream from safetensor files, and emits
    merged tensors once all experts for a given (layer, projection) are collected.

    Implements ``fused_expert_target`` / ``for_expert_range`` so
    ``ep_sharded_stream_load`` can read only this rank's experts (including their
    FP4/FP8 scale companions) and stack the local ``[E/ep, ...]`` slice.

    Args:
        num_experts: Number of routed experts this converter stacks
            (``config.n_routed_experts`` for a full-rank converter — equal to
            ``config.num_local_experts`` via ``DeepseekV4Config.attribute_map``;
            ``E/ep`` after :meth:`for_expert_range`).
        expert_offset: Checkpoint index of the first expert it stacks; experts
            ``[expert_offset, expert_offset + num_experts)`` become rows
            ``0..num_experts-1``.
        target_model_prefix: Prefix prepended by :func:`convert_deepseek_v4_checkpoint_key`
            (``"model."`` for CausalLM, ``""`` for the bare Model).
        enable_mtp: Whether ``mtp.*`` DSpark draft stages are active on the live model.
        mtp_num_experts: Routed expert count for ``mtp.*`` stages (defaults to ``num_experts``).
    """

    def __init__(
        self,
        num_experts: int,
        *,
        expert_offset: int = 0,
        target_model_prefix: str = "model.",
        enable_mtp: bool = False,
        mtp_num_experts: int | None = None,
        mtp_expert_offset: int | None = None,
    ):
        self.num_experts = num_experts
        self.mtp_num_experts = num_experts if mtp_num_experts is None else mtp_num_experts
        self.expert_offset = expert_offset
        self.mtp_expert_offset = expert_offset if mtp_expert_offset is None else mtp_expert_offset
        self.target_model_prefix = target_model_prefix
        self.enable_mtp = enable_mtp
        # {(prefix, proj_name): {local_expert_id: tensor}}
        self._expert_buffer: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        # {prefix: {proj_name: stacked_tensor}} for gate/up merge waiting
        self._stacked_buffer: dict[str, dict[str, torch.Tensor]] = {}
        # {weight_name: {"weight": tensor, "scale": tensor}} for FP8/int8 checkpoint pairs.
        self._scaled_weight_buffer: dict[str, dict[str, torch.Tensor | str | bool]] = {}

    def _convert_key(self, name: str) -> str:
        return convert_deepseek_v4_checkpoint_key(
            name,
            target_model_prefix=self.target_model_prefix,
            enable_mtp=self.enable_mtp,
        )

    def _expert_range_for_prefix(self, prefix: str) -> tuple[int, int]:
        if _is_mtp_expert_prefix(prefix):
            return self.mtp_expert_offset, self.mtp_num_experts
        return self.expert_offset, self.num_experts

    def can_handle(self, name: str) -> bool:
        if _is_mtp_checkpoint_key(name) and not self.enable_mtp:
            return False
        converted_name = self._convert_key(name)
        return (
            converted_name != name
            or bool(_EXPERT_PATTERN.match(converted_name))
            or _weight_name_from_scale_name(converted_name) is not None
        )

    def can_handle_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if _is_mtp_checkpoint_key(name) and not self.enable_mtp:
            return False
        converted_name = self._convert_key(name)
        return self.can_handle(name) or (converted_name.endswith(".weight") and _is_quantized_weight(tensor))

    def fused_expert_target(self, name: str) -> tuple[str, int] | None:
        """Map a per-expert weight or its scale to ``(fused_param, expert_index)``."""
        if _is_mtp_checkpoint_key(name) and not self.enable_mtp:
            return None
        converted_name = self._convert_key(name)
        weight_name = _weight_name_from_scale_name(converted_name) or converted_name
        match = _EXPERT_PATTERN.match(weight_name)
        if not match:
            return None
        prefix, expert_id_str, proj_name = match.groups()
        proj_name = _PROJ_NAME_ALIASES.get(proj_name, proj_name)
        fused_proj = "down_proj" if proj_name == "down_proj" else "gate_up_proj"
        return f"{prefix}.experts.{fused_proj}", int(expert_id_str)

    def for_expert_range(
        self,
        start: int,
        num_local: int,
        *,
        mtp_start: int | None = None,
        mtp_num_local: int | None = None,
    ) -> DeepseekV4CheckpointTensorConverter:
        if mtp_start is None and mtp_num_local is None:
            max_experts = max(self.num_experts, self.mtp_num_experts) if self.enable_mtp else self.num_experts
            if start < 0 or num_local <= 0 or start + num_local > max_experts:
                raise ValueError(
                    f"Expert range [{start}, {start + num_local}) is outside this converter's {max_experts} experts."
                )
            return DeepseekV4CheckpointTensorConverter(
                num_experts=num_local,
                expert_offset=self.expert_offset + start,
                target_model_prefix=self.target_model_prefix,
                enable_mtp=self.enable_mtp,
                mtp_num_experts=num_local,
                mtp_expert_offset=self.mtp_expert_offset + start,
            )

        resolved_mtp_start = start if mtp_start is None else mtp_start
        resolved_mtp_local = num_local if mtp_num_local is None else mtp_num_local
        if start < 0 or num_local <= 0 or start + num_local > self.num_experts:
            raise ValueError(
                f"Expert range [{start}, {start + num_local}) is outside this converter's {self.num_experts} experts."
            )
        if self.enable_mtp and (
            resolved_mtp_start < 0
            or resolved_mtp_local <= 0
            or resolved_mtp_start + resolved_mtp_local > self.mtp_num_experts
        ):
            raise ValueError(
                f"MTP expert range [{resolved_mtp_start}, {resolved_mtp_start + resolved_mtp_local}) "
                f"is outside this converter's {self.mtp_num_experts} MTP experts."
            )
        return DeepseekV4CheckpointTensorConverter(
            num_experts=num_local,
            expert_offset=self.expert_offset + start,
            target_model_prefix=self.target_model_prefix,
            enable_mtp=self.enable_mtp,
            mtp_num_experts=resolved_mtp_local,
            mtp_expert_offset=self.mtp_expert_offset + resolved_mtp_start,
        )

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        if _is_mtp_checkpoint_key(name) and not self.enable_mtp:
            return ConvertedCheckpointTensor(name, tensor)
        packed_fp4 = tensor.dtype == torch.int8 and bool(_RAW_FP4_EXPERT_WEIGHT_PATTERN.fullmatch(name))
        name = self._convert_key(name)
        weight_name = _weight_name_from_scale_name(name)
        if weight_name is not None:
            if not _is_block_scale_tensor(tensor):
                return ConvertedCheckpointTensor(name, tensor)
            return self._convert_scaled_weight_part(weight_name, "scale", tensor, name)

        if name.endswith(".weight") and _is_quantized_weight(tensor):
            return self._convert_scaled_weight_part(name, "weight", tensor, None, packed_fp4=packed_fp4)

        return self._convert_weight(name, tensor)

    def _convert_scaled_weight_part(
        self,
        weight_name: str,
        part_name: str,
        tensor: torch.Tensor,
        scale_name: str | None,
        *,
        packed_fp4: bool = False,
    ) -> ConvertedCheckpointTensor | None:
        parts = self._scaled_weight_buffer.setdefault(weight_name, {})
        parts[part_name] = tensor
        if part_name == "weight":
            parts["packed_fp4"] = packed_fp4
        if scale_name is not None:
            parts["scale_name"] = scale_name
        if "weight" not in parts or "scale" not in parts:
            return None

        weight = parts["weight"]
        scale = parts["scale"]
        del self._scaled_weight_buffer[weight_name]
        assert isinstance(weight, torch.Tensor)
        assert isinstance(scale, torch.Tensor)
        return self._convert_weight(
            weight_name,
            _dequantize_scaled_weight(weight, scale, packed_fp4=bool(parts.get("packed_fp4", False))),
        )

    def _convert_weight(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = _EXPERT_PATTERN.match(name)
        if not match:
            return ConvertedCheckpointTensor(name, tensor)

        prefix, expert_id_str, proj_name = match.groups()
        proj_name = _PROJ_NAME_ALIASES.get(proj_name, proj_name)
        expert_offset, expected_experts = self._expert_range_for_prefix(prefix)
        expert_id = int(expert_id_str) - expert_offset
        if not 0 <= expert_id < expected_experts:
            raise ValueError(
                f"{name}: expert {int(expert_id_str)} is outside this converter's range "
                f"[{expert_offset}, {expert_offset + expected_experts})."
            )
        buf_key = (prefix, proj_name)

        if buf_key not in self._expert_buffer:
            self._expert_buffer[buf_key] = {}
        self._expert_buffer[buf_key][expert_id] = tensor

        if len(self._expert_buffer[buf_key]) < expected_experts:
            return None

        # Stack all experts: [E, I, H] or [E, H, I]
        stacked = torch.stack([self._expert_buffer[buf_key][i] for i in range(expected_experts)])
        del self._expert_buffer[buf_key]

        if proj_name == "down_proj":
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", stacked)

        # gate_proj or up_proj — buffer for merging with the other
        if prefix not in self._stacked_buffer:
            self._stacked_buffer[prefix] = {}
        self._stacked_buffer[prefix][proj_name] = stacked

        if "gate_proj" in self._stacked_buffer[prefix] and "up_proj" in self._stacked_buffer[prefix]:
            gate = self._stacked_buffer[prefix].pop("gate_proj")
            up = self._stacked_buffer[prefix].pop("up_proj")
            if not self._stacked_buffer[prefix]:
                del self._stacked_buffer[prefix]
            merged = torch.cat([gate, up], dim=1)  # [E, 2*I, H]
            return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", merged)

        return None

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        """Validate that all buffers were flushed.

        Raises RuntimeError if any buffers remain unflushed, since incomplete
        expert tensors cannot be merged into valid fused format and indicate
        a corrupted or incomplete checkpoint.
        """
        errors: list[str] = []
        if self._expert_buffer:
            unflushed = {k: len(v) for k, v in self._expert_buffer.items()}
            errors.append(
                f"unflushed per-expert buffer (incomplete experts, expected {self.num_experts}): {unflushed}"
            )
        if self._stacked_buffer:
            unflushed = {k: list(v.keys()) for k, v in self._stacked_buffer.items()}
            errors.append(f"unflushed stacked buffer (missing gate/up pair): {unflushed}")
        finalized: list[ConvertedCheckpointTensor] = []
        for weight_name, parts in self._scaled_weight_buffer.items():
            if "weight" in parts:
                errors.append(f"unflushed scaled weight buffer (missing scale for {weight_name})")
                continue
            scale = parts.get("scale")
            scale_name = parts.get("scale_name")
            if isinstance(scale, torch.Tensor) and isinstance(scale_name, str):
                finalized.append(ConvertedCheckpointTensor(scale_name, scale))
        if errors:
            raise RuntimeError("DeepseekV4 checkpoint converter: incomplete checkpoint detected. " + "; ".join(errors))
        self._scaled_weight_buffer.clear()
        return finalized

    def validate_export_support(
        self,
        model: torch.nn.Module | None = None,
        *,
        fqn_to_index_mapping: Mapping[str, int] | None = None,
        should_quantize: Callable[[str], bool] | None = None,
    ) -> None:
        """Fail before training or export if quantized ``.scale`` export is requested on a non-SM90 device."""
        validate_deepseek_v4_export_quant_support(
            fqn_to_index_mapping,
            model=model,
            target_model_prefix=self.target_model_prefix,
            should_quantize=should_quantize,
            enable_mtp=self.enable_mtp,
        )

    def export_weights(
        self,
        model: torch.nn.Module,
        *,
        should_quantize: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor]]:
        """Export model parameters in DeepSeek-V4 release format with FP8/FP4 quantization when applicable."""
        config = getattr(model, "config", None)
        fqn_to_index_mapping = getattr(model, "_veomni_fqn_to_index_mapping", None)
        if fqn_to_index_mapping is None:
            fqn_to_index_mapping = _build_default_deepseek_v4_fqn_to_index_mapping(
                model,
                target_model_prefix=self.target_model_prefix,
                enable_mtp=self.enable_mtp,
            )
            model._veomni_fqn_to_index_mapping = fqn_to_index_mapping

        _quant_cfg, fp8_block_size, expert_dtype, _has_quant = _extract_quant_config(config)
        scale_fmt = "ue8m0"
        scale_dtype = torch.float8_e8m0fnu if expert_dtype == "fp4" else torch.float32

        # Expand any v5 fused expert keys in fqn_to_index_mapping into per-expert release keys.
        expanded_fused: dict[str, int] = {}
        removed_fused: list[str] = []
        for key, shard_idx in fqn_to_index_mapping.items():
            if key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key.removesuffix(".gate_up_proj")
                if _is_mtp_expert_prefix(prefix) and not self.enable_mtp:
                    continue
                expected_experts = self.mtp_num_experts if _is_mtp_expert_prefix(prefix) else self.num_experts
                for e in range(expected_experts):
                    for w_name in ("w1", "w3"):
                        origin_k = _revert_deepseek_v4_checkpoint_key(
                            f"{prefix}.{e}.{w_name}.weight",
                            target_model_prefix=self.target_model_prefix,
                            enable_mtp=self.enable_mtp,
                        )
                        expanded_fused[origin_k] = shard_idx
                removed_fused.append(key)
            elif key.endswith(".mlp.experts.down_proj"):
                prefix = key.removesuffix(".down_proj")
                if _is_mtp_expert_prefix(prefix) and not self.enable_mtp:
                    continue
                expected_experts = self.mtp_num_experts if _is_mtp_expert_prefix(prefix) else self.num_experts
                for e in range(expected_experts):
                    origin_k = _revert_deepseek_v4_checkpoint_key(
                        f"{prefix}.{e}.w2.weight",
                        target_model_prefix=self.target_model_prefix,
                        enable_mtp=self.enable_mtp,
                    )
                    expanded_fused[origin_k] = shard_idx
                removed_fused.append(key)
        if expanded_fused:
            for key in removed_fused:
                fqn_to_index_mapping.pop(key, None)
            fqn_to_index_mapping.update(expanded_fused)

        # HF's renamings are not invertible (e.g. reversing ``.norm.`` -> ``.kv_norm.`` also rewrites the native
        # ``attn.kv_norm``), so map names back by inverting the load-time renaming over the checkpoint index.
        origin_names: dict[str, str] = {}
        for origin in fqn_to_index_mapping:
            if _is_mtp_checkpoint_key(origin) and not self.enable_mtp:
                continue
            converted = convert_deepseek_v4_checkpoint_key(
                origin, target_model_prefix=self.target_model_prefix, enable_mtp=self.enable_mtp
            )
            if "mlp.experts." in converted:
                fields = converted.split(".")
                if len(fields) >= 2 and fields[-2] in _PROJ_NAME_TO_W_NAME:
                    fields[-2] = _PROJ_NAME_TO_W_NAME[fields[-2]]
                    converted = ".".join(fields)
            origin_names[converted] = origin

        export_weight_names = set()
        for name, param in export_weights(model):
            if _is_mtp_checkpoint_key(name) and not self.enable_mtp:
                continue
            # 1. replace mlp.experts.0.(gate_proj|up_proj|down_proj).weight to mlp.experts.0.(w1|w2|w3).weight
            if "mlp.experts." in name:
                fields = name.split(".")
                fields[-2] = _PROJ_NAME_TO_W_NAME[fields[-2]]
                name = ".".join(fields)

            # 2. reverse the renaming
            origin = origin_names.get(name)
            if origin is None:
                if not fqn_to_index_mapping:
                    origin = _revert_deepseek_v4_checkpoint_key(
                        name, target_model_prefix=self.target_model_prefix, enable_mtp=self.enable_mtp
                    )
                else:
                    is_tied_head = name in ("lm_head.weight", "model.lm_head.weight") and bool(
                        getattr(config, "tie_word_embeddings", False)
                    )
                    if is_tied_head:
                        logger.info_rank0(f"Skipping tied weight not in fqn_to_index_mapping: {name}")
                    else:
                        logger.warning_rank0(f"Skipping exported weight not in fqn_to_index_mapping: {name}")
                    continue

            # 3. check if the weight needs to be quantized
            scale_name = None
            if origin.endswith(".weight"):
                scale_name = origin.removesuffix(".weight") + ".scale"

            if scale_name is None or scale_name not in fqn_to_index_mapping:
                export_weight_names.add(origin)
                if should_quantize is not None and not should_quantize(origin):
                    continue
                if origin.endswith(".weight") and param.dtype == torch.float32:
                    param = param.to(torch.bfloat16)
                yield origin, param
                continue

            export_weight_names.add(origin)
            export_weight_names.add(scale_name)
            if should_quantize is not None and not should_quantize(origin):
                continue

            # 4. quantize the weight
            fp4_quantize = "ffn.experts." in origin and expert_dtype == "fp4"
            from veomni.models.transformers.deepseek_v4.export_quant import fp4_act_quant, fp8_weight_quant

            param = param.to(torch.bfloat16)
            if fp4_quantize:
                weight, scale = fp4_act_quant(param, block_size=32)
                weight = weight.view(torch.int8)
            else:
                weight, scale = fp8_weight_quant(
                    param, block_size=fp8_block_size, scale_fmt=scale_fmt, scale_dtype=scale_dtype
                )

            yield origin, weight
            yield scale_name, scale

        missing_weight_names = fqn_to_index_mapping.keys() - export_weight_names
        if not self.enable_mtp:
            missing_weight_names = {name for name in missing_weight_names if not _is_mtp_checkpoint_key(name)}
        if missing_weight_names:
            logger.warning(f"Export weights missing: {missing_weight_names}")


def create_deepseek_v4_checkpoint_tensor_converter(model):
    """Factory function registered on model classes via _create_checkpoint_tensor_converter."""
    base_model = getattr(model, "model", model)
    mtp_modules = getattr(base_model, "mtp", None)
    enable_mtp = bool(mtp_modules is not None and len(mtp_modules) > 0)
    mtp_num_experts = getattr(model.config, "dspark_n_routed_experts", None) or model.config.n_routed_experts
    return DeepseekV4CheckpointTensorConverter(
        num_experts=model.config.n_routed_experts,
        target_model_prefix=_get_deepseek_v4_target_model_prefix(model),
        enable_mtp=enable_mtp,
        mtp_num_experts=mtp_num_experts,
    )


def _get_deepseek_v4_target_model_prefix(model) -> str:
    """Infer whether *model* uses the ``'model.'`` CausalLM prefix or bare base-model parameter names."""
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for name, _ in named_parameters():
            if name.startswith("model."):
                return "model."
            if name.startswith(_BASE_MODEL_KEY_PREFIXES):
                return ""

    if type(model).__name__.endswith("ForCausalLM"):
        return "model."
    return ""
