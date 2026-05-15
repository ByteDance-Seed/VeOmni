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

import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed.parallel_state import get_parallel_state
from ..models.module_utils import BroadcastMetadata, _dispatch_parameter
from ..utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.get_logger(__name__)


def build_lora_key_overrides(model: "nn.Module") -> "Dict[str, str]":
    """Build a mapping from bare base-model parameter names to PEFT-wrapped FQNs.

    When a base checkpoint is loaded into a PEFT-wrapped model, each target
    ``Linear`` is replaced by a ``LoraLinear`` that stores the original weight
    under a ``base_layer`` sub-module.  This function produces a remapping dict
    so callers can translate checkpoint keys transparently, e.g.::

        "layers.0.self_attn.q_proj.weight"
        -> "base_model.model.layers.0.self_attn.q_proj.base_layer.weight"

    Keys absent from the returned dict should receive a plain
    ``"base_model.model."`` prefix.

    Returns:
        A ``{checkpoint_key: model_fqn}`` dict for every LoRA layer's
        parameters and buffers.  Empty dict if the model has no LoRA layers.
    """
    from typing import Dict

    overrides: Dict[str, str] = {}
    for fqn, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        inner = fqn[len("base_model.model.") :] if fqn.startswith("base_model.model.") else fqn
        inner_dot = inner + ("." if inner else "")
        wrap_dot = fqn + ("." if fqn else "") + "base_layer."
        for pname, _ in module.base_layer.named_parameters():
            overrides[inner_dot + pname] = wrap_dot + pname
        for bname, _ in module.base_layer.named_buffers():
            overrides[inner_dot + bname] = wrap_dot + bname
    return overrides


def _read_adapter_name(adapter_path: str) -> str:
    """Read the adapter name from adapter_config.json, defaulting to 'default'."""
    import json

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("adapter_name", "default") or "default"
    return "default"


def _remap_adapter_key(key: str, adapter_name: str) -> str:
    """Remap a PEFT-saved key to model FQN format.

    PEFT saves ``lora_A.weight`` but the model FQN is ``lora_A.<adapter_name>.weight``.
    """
    parts = key.split(".")
    new_parts = []
    for p in parts:
        new_parts.append(p)
        if p in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            new_parts.append(adapter_name)
    return ".".join(new_parts)


# fsdp2 meta device load on every rank
@torch.no_grad()
def load_lora_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    adapter_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    parameter_names_to_load: Optional[set] = None,
) -> None:
    """Load PEFT adapter (LoRA) weights from disk into the model on every rank.

    Mirrors ``load_model_weights`` but targets adapter files.  Each rank reads
    ``adapter_model.safetensors`` (or ``.bin``) directly, remaps PEFT key names
    to model FQN format, and dispatches tensors into the (potentially sharded) model.
    Use when every rank has access to the checkpoint (e.g. shared filesystem).

    Args:
        parameter_names_to_load: If provided, each successfully loaded parameter
            name is discarded from this set so that ``post_process_after_weight_loading``
            does not re-initialise adapter weights that have already been loaded.
    """
    from peft import load_peft_weights

    adapter_name = _read_adapter_name(adapter_path)
    raw_sd = load_peft_weights(adapter_path, device=init_device)
    for name, tensor in raw_sd.items():
        name = _remap_adapter_key(name, adapter_name)
        _dispatch_parameter(model, name, tensor, dtensor_factory)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)


# fsdp2 init lora parameters during post_process_after_weight_loading
def _init_lora_parameter(module: "nn.Module", name: str):
    pieces = name.split(".")
    lora_layer = module
    for piece in pieces:
        if piece.startswith("lora_"):
            break
        lora_layer = getattr(lora_layer, piece)
    if "lora_A" in name and hasattr(lora_layer, "reset_lora_parameters"):
        for adapter in getattr(lora_layer, "lora_A", {}).keys():
            lora_layer.reset_lora_parameters(adapter, init_lora_weights=True)
    # lora_B is initialized during lora_A reset_lora_parameters


# fsdp2 meta device rank0 load and broadcast adapter weights
@torch.no_grad()
def rank0_load_and_broadcast_adapter_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    adapter_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    parameter_names_to_load: Optional[set] = None,
) -> None:
    """Rank-0 loads PEFT adapter weights from disk then broadcasts to all ranks.

    Args:
        parameter_names_to_load: If provided, each successfully loaded parameter
            name is discarded from this set so that ``post_process_after_weight_loading``
            does not re-initialise adapter weights that have already been loaded.
    """
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    adapter_sd = {}
    if global_rank == 0:
        from peft import load_peft_weights

        adapter_name = _read_adapter_name(adapter_path)
        raw_sd = load_peft_weights(adapter_path, device="cpu")
        remapped = {_remap_adapter_key(k, adapter_name): v for k, v in raw_sd.items()}
        if remapped:
            first_raw = next(iter(raw_sd))
            first_remapped = next(iter(remapped))
            logger.info_rank0(
                f"Loaded {len(remapped)} adapter weight(s) from {adapter_path}, "
                f"key remap example: {first_raw} -> {first_remapped}"
            )
        adapter_sd = remapped

    if not dist.is_available() or not dist.is_initialized():
        for name, tensor in adapter_sd.items():
            _dispatch_parameter(model, name, tensor, dtensor_factory)
        return

    global_rank = get_parallel_state().global_rank
    torch_device = torch.device(init_device)

    # Broadcast the number of adapter keys so all ranks know the loop count
    count_tensor = torch.tensor(
        [len(adapter_sd)],
        dtype=torch.int64,
        device=torch_device if torch_device.type != "cpu" else torch.device("cpu"),
    )
    dist.broadcast(count_tensor, src=0)
    num_keys = int(count_tensor.item())

    if num_keys == 0:
        return

    sorted_keys = sorted(adapter_sd.keys()) if global_rank == 0 else [None] * num_keys

    for i in range(num_keys):
        if global_rank == 0:
            name = sorted_keys[i]
            tensor = adapter_sd[name].to(torch_device, non_blocking=True)
            metadata = BroadcastMetadata(False, name, tensor.shape, tensor.dtype)
        else:
            metadata = BroadcastMetadata(False, None, None, None)

        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=0)
        metadata = metadata_list[0]

        name = metadata.name
        shape = metadata.shape
        dtype = metadata.dtype

        logger.info_rank0(f"loading {name=}")

        if global_rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device=torch_device)

        start_time = time.perf_counter()
        dist.broadcast(tensor, src=0)
        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, broadcast time (ms) spent: {1000 * (time.perf_counter() - start_time)}"
        )
        _dispatch_parameter(model, name, tensor, dtensor_factory)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)
        del tensor

    logger.info_rank0(f"rank0_broadcast_adapter_weights: loaded {num_keys} adapter param(s)")
