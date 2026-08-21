"""Checkpoint compatibility for BAGEL's combined QKV projections.

The runtime uses one merged ``qkv_proj_und`` or ``qkv_proj_gen`` parameter per
MoT branch to issue one linear projection. Existing checkpoints keep their
separate Q/K/V keys, so each load/save API needs an adapter at its own boundary:
PyTorch state-dict hooks, Transformers weight converters, or VeOmni's streaming
checkpoint converter.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import ConversionOps, WeightConverter

from ......models.checkpoint_tensor_loading import ConvertedCheckpointTensor


_QKV_COMPONENTS = ("q", "k", "v")
_QKV_BRANCHES = (
    ("qkv_proj_und", ("q_proj", "k_proj", "v_proj")),
    ("qkv_proj_gen", ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen")),
)
_QKV_CHECKPOINT_RE = re.compile(
    r"^(?P<prefix>.*\.self_attn\.)"
    r"(?P<projection>[qkv]_proj(?:_moe_gen)?)\."
    r"(?P<kind>weight|bias)$"
)


def combine_qkv_state_dict_pre_hook(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Accept the legacy split Q/K/V schema in direct PyTorch and DCP loads."""
    del module, local_metadata, strict, missing_keys, unexpected_keys
    for combined_name, checkpoint_names in _QKV_BRANCHES:
        for kind in ("weight", "bias"):
            combined_key = f"{prefix}{combined_name}.{kind}"
            checkpoint_keys = [f"{prefix}{name}.{kind}" for name in checkpoint_names]
            if combined_key in state_dict or not any(key in state_dict for key in checkpoint_keys):
                continue
            if not all(key in state_dict for key in checkpoint_keys):
                present = [key for key in checkpoint_keys if key in state_dict]
                error_msgs.append(
                    f"Incomplete BAGEL QKV checkpoint group for {combined_key}: found {present}, "
                    f"expected {checkpoint_keys}."
                )
                continue

            state_dict[combined_key] = torch.cat([state_dict.pop(key) for key in checkpoint_keys], dim=0)


def split_qkv_state_dict_post_hook(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
) -> None:
    """Expose legacy split Q/K/V keys while keeping combined runtime parameters."""
    del local_metadata
    split_sizes = module.qkv_split_sizes
    for combined_name, checkpoint_names in _QKV_BRANCHES:
        for kind in ("weight", "bias"):
            combined_key = f"{prefix}{combined_name}.{kind}"
            tensor = state_dict.get(combined_key)
            if tensor is None or tensor.is_meta or hasattr(tensor, "device_mesh"):
                # Transformers inspects a meta model's state dict to discover the
                # runtime parameter schema before applying its weight converters.
                # DCP likewise requires DTensor keys to remain resolvable to the
                # live combined parameters before the HF export layer converts them.
                continue

            state_dict.pop(combined_key)
            for checkpoint_name, chunk in zip(
                checkpoint_names,
                tensor.split(split_sizes, dim=0),
                strict=True,
            ):
                state_dict[f"{prefix}{checkpoint_name}.{kind}"] = chunk


class _ConcatenateQKV(ConversionOps):
    """Concatenate checkpoint Q/K/V tensors along the projection dimension."""

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if len(target_patterns) != 1:
            raise ValueError("Combined BAGEL QKV conversion requires one target pattern.")

        tensors: list[torch.Tensor] = []
        for source_pattern in source_patterns:
            values = input_dict.pop(source_pattern)
            tensors.extend(values if isinstance(values, list) else [values])
        return {target_patterns[0]: torch.cat(tensors, dim=0)}

    @property
    def reverse_op(self) -> ConversionOps:
        return _SplitQKV()


class _SplitQKV(ConversionOps):
    """Restore combined QKV tensors to BAGEL's external checkpoint schema."""

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del source_patterns
        if len(input_dict) != 1 or len(target_patterns) != 3:
            raise ValueError("BAGEL QKV export requires one combined source and three target patterns.")

        values = next(iter(input_dict.values()))
        tensor = values[0] if isinstance(values, list) else values
        config = kwargs["config"]
        head_dim = config.hidden_size // config.num_attention_heads
        split_sizes = (
            config.num_attention_heads * head_dim,
            config.num_key_value_heads * head_dim,
            config.num_key_value_heads * head_dim,
        )
        return dict(zip(target_patterns, tensor.split(split_sizes, dim=0), strict=True))

    @property
    def reverse_op(self) -> ConversionOps:
        return _ConcatenateQKV()


def _qkv_weight_converters() -> list[WeightConverter]:
    converters: list[WeightConverter] = []
    for branch, source_suffix in (("und", ""), ("gen", "_moe_gen")):
        for kind in ("weight", "bias"):
            converters.append(
                WeightConverter(
                    source_patterns=[
                        f"self_attn.{component}_proj{source_suffix}.{kind}" for component in _QKV_COMPONENTS
                    ],
                    target_patterns=f"self_attn.qkv_proj_{branch}.{kind}",
                    operations=[_ConcatenateQKV()],
                )
            )
    return converters


# Transformers v5 can bypass load_state_dict while loading and reverses this
# mapping during save, preserving the legacy split Q/K/V checkpoint schema.
for model_identifier in ("BagelQwen2MoT", "bagel_qwen2_mot"):
    register_checkpoint_conversion_mapping(
        model_identifier,
        _qkv_weight_converters(),
        overwrite=True,
    )


class BagelQwen2MoTCheckpointTensorConverter:
    """Stream legacy Q/K/V tensors into the combined runtime parameters.

    VeOmni's ``load_model_weights`` assigns tensors directly instead of calling
    ``load_state_dict``, so the PyTorch pre-hook cannot perform this merge.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return _QKV_CHECKPOINT_RE.fullmatch(name) is not None

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        match = _QKV_CHECKPOINT_RE.fullmatch(name)
        if match is None:
            return ConvertedCheckpointTensor(name=name, tensor=tensor)

        projection = match.group("projection")
        component = projection[0]
        branch = "gen" if projection.endswith("_moe_gen") else "und"
        group_key = (match.group("prefix"), branch, match.group("kind"))
        group = self._pending.setdefault(group_key, {})
        if component in group:
            raise ValueError(f"Duplicate BAGEL QKV checkpoint tensor: {name}.")
        group[component] = tensor
        if group.keys() != set(_QKV_COMPONENTS):
            return None

        self._pending.pop(group_key)
        combined = torch.cat([group[component] for component in _QKV_COMPONENTS], dim=0)
        prefix, branch, kind = group_key
        return ConvertedCheckpointTensor(
            name=f"{prefix}qkv_proj_{branch}.{kind}",
            tensor=combined,
        )

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        if self._pending:
            incomplete = ", ".join(
                f"{prefix}qkv_proj_{branch}.{kind}" for prefix, branch, kind in sorted(self._pending)
            )
            raise ValueError(f"Incomplete BAGEL QKV checkpoint groups: {incomplete}.")
        return []
