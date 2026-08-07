# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Adapter state-dict extraction, key remapping, and PEFT-format file I/O.

The native equivalent of ``peft.get_peft_model_state_dict`` /
``peft.set_peft_model_state_dict`` / ``peft.load_peft_weights``. All on-disk
keys use PEFT's convention: the adapter-name infix is stripped
(``...lora_A.default.weight`` on the live model becomes ``...lora_A.weight`` on
disk), and the ``base_model.model.`` prefix is preserved verbatim.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from ..utils import logging


if TYPE_CHECKING:
    import torch.nn as nn

    from .config import VeOmniLoraConfig


logger = logging.get_logger(__name__)

SAFETENSORS_NAME = "adapter_model.safetensors"
BIN_NAME = "adapter_model.bin"

# Segments that identify a LoRA tensor in a live-model FQN.
_LORA_SEGMENTS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
# MoE expert LoRA spec sub-module names (see moe_layers).
_MOE_SPEC_NAMES = ("gate_proj", "up_proj", "down_proj")


def _is_lora_key(key: str) -> bool:
    parts = key.split(".")
    return any(seg in parts for seg in _LORA_SEGMENTS)


def strip_adapter_name(key: str, adapter_name: str) -> str:
    """``...lora_A.<adapter>.weight`` -> ``...lora_A.weight`` (PEFT on-disk form)."""
    return key.replace(f".{adapter_name}.", ".", 1) if f".{adapter_name}." in key else key


def insert_adapter_name(key: str, adapter_name: str) -> str:
    """``...lora_A.weight`` -> ``...lora_A.<adapter>.weight`` (live-model form).

    Inserts the adapter name after each ``lora_A`` / ``lora_B`` segment,
    mirroring PEFT's ``set_peft_model_state_dict`` remap.
    """
    parts = key.split(".")
    out: list[str] = []
    for p in parts:
        out.append(p)
        if p in _LORA_SEGMENTS:
            out.append(adapter_name)
    return ".".join(out)


def get_lora_state_dict(
    model: nn.Module,
    adapter_name: str = "default",
    config: VeOmniLoraConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Collect the adapter-only state dict in PEFT on-disk key format.

    Replaces ``peft.get_peft_model_state_dict``: returns ``lora_A`` / ``lora_B``
    weights (plus, per ``bias`` policy, trainable biases and, if configured,
    ``modules_to_save`` copies), with the adapter-name infix stripped so the
    keys match a stock PEFT ``adapter_model`` file.
    """
    full = dict(model.named_parameters())
    bias_policy = config.bias if config is not None else "none"
    modules_to_save = list(config.modules_to_save) if (config and config.modules_to_save) else []

    out: dict[str, torch.Tensor] = {}
    for name, param in full.items():
        keep = _is_lora_key(name)
        if not keep and bias_policy != "none" and name.endswith("bias") and param.requires_grad:
            keep = True
        if not keep and modules_to_save and ".modules_to_save." in name:
            keep = True
        if keep:
            out[strip_adapter_name(name, adapter_name)] = param
    return out


def _find_adapter_file(adapter_path: str) -> tuple[str, bool]:
    """Return ``(file_path, is_safetensors)`` for the adapter in ``adapter_path``."""
    safetensors_path = os.path.join(adapter_path, SAFETENSORS_NAME)
    bin_path = os.path.join(adapter_path, BIN_NAME)
    if os.path.isfile(safetensors_path):
        return safetensors_path, True
    if os.path.isfile(bin_path):
        return bin_path, False
    raise FileNotFoundError(f"No {SAFETENSORS_NAME} or {BIN_NAME} found in {adapter_path!r}.")


def load_adapter_state_dict(adapter_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read the raw adapter tensors (PEFT on-disk keys). Replaces ``load_peft_weights``."""
    file_path, is_safetensors = _find_adapter_file(adapter_path)
    if is_safetensors:
        from safetensors.torch import load_file

        return load_file(file_path, device=device)
    return torch.load(file_path, map_location=device, weights_only=True)


def save_adapter_file(
    state: dict[str, torch.Tensor],
    save_directory: str,
    safe_serialization: bool = True,
) -> str:
    """Write the adapter state dict to disk in PEFT format. Returns the file path."""
    os.makedirs(save_directory, exist_ok=True)
    cpu_state = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}
    if safe_serialization:
        from safetensors.torch import save_file

        path = os.path.join(save_directory, SAFETENSORS_NAME)
        save_file(cpu_state, path, metadata={"format": "pt"})
    else:
        path = os.path.join(save_directory, BIN_NAME)
        torch.save(cpu_state, path)
    return path


def infer_moe_mode_from_adapter(adapter_path: str) -> str:
    """Infer ``independent`` / ``shared`` from a stock-PEFT adapter's tensor shapes.

    MoE expert LoRA tensors live under an ``.experts.<spec>.lora_{A,B}`` path.
    Independent (per-expert) tensors are 3-D (leading expert dim); shared
    tensors are 2-D. Defaults to ``independent`` if no MoE tensors are found
    (safest: it is the shape-preserving superset).
    """
    state = load_adapter_state_dict(adapter_path, device="cpu")
    for key, tensor in state.items():
        parts = key.split(".")
        is_moe = ".experts." in key and any(spec in parts for spec in _MOE_SPEC_NAMES)
        if is_moe and _is_lora_key(key):
            return "independent" if tensor.ndim == 3 else "shared"
    logger.warning_rank0(
        f"infer_moe_mode_from_adapter: no MoE expert LoRA tensors found in {adapter_path!r}; "
        "defaulting moe_mode='independent'."
    )
    return "independent"
