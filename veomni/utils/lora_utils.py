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

    from ..distributed.parallel_plan import ParallelPlan

logger = logging.get_logger(__name__)


def build_lora_key_overrides(model: "nn.Module") -> "Dict[str, str]":
    """Build a mapping from bare base-model parameter names to PEFT-wrapped FQNs.

    When a base checkpoint is loaded into a PEFT-wrapped model, each target
    ``Linear`` is replaced by a ``LoraLinear`` that stores the original weight
    under a ``base_layer`` sub-module.  This function produces a remapping dict
    so callers can translate checkpoint keys transparently, e.g.::

        "layers.0.self_attn.q_proj.weight"
        -> "base_model.model.layers.0.self_attn.q_proj.base_layer.weight"

    For VeOmni MoE-LoRA wrappers (:class:`~veomni.utils.moe_lora.LoraSharedExperts`
    / :class:`~veomni.utils.moe_lora.LoraIndependentExperts`), the lifted base
    parameter (e.g. fused ``gate_up_proj``) lives inside a per-spec
    ``_LoraSpec`` sub-module under ``base_layer.weight``. Because the original
    checkpoint key was a *bare* ``nn.Parameter`` on the experts module (no
    ``.weight`` suffix in the saved key), the holder marks itself with
    ``_is_bare_param_holder = True`` so this function can emit the bare->wrapped
    mapping with no suffix on the source side, e.g.::

        "layers.0.mlp.experts.gate_up_proj"
        -> "base_model.model.layers.0.mlp.experts.gate_up_proj.base_layer.weight"

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
        # MoE-LoRA bare-Param case: the checkpoint key for the wrapped base is
        # the wrapper-spec FQN itself (no `.weight` because the original module
        # exposed an `nn.Parameter` directly, not a Linear).
        if getattr(module.base_layer, "_is_bare_param_holder", False):
            overrides[inner] = wrap_dot + "weight"
            continue
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


_LORA_EXACT_PARTS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def _remap_adapter_key(key: str, adapter_name: str) -> str:
    """Remap a PEFT-saved key to model FQN format.

    PEFT saves ``lora_A.weight`` but the model FQN is ``lora_A.<adapter_name>.weight``.
    Both standard PEFT layers and VeOmni's MoE-LoRA wrappers
    (:class:`~veomni.utils.moe_lora.LoraSharedExperts` /
    :class:`~veomni.utils.moe_lora.LoraIndependentExperts`) follow this
    exact convention: per-spec sub-modules expose ``lora_A`` / ``lora_B``
    ``nn.ModuleDict`` containers keyed by adapter name, so a single
    insertion of ``adapter_name`` after the ``lora_A`` / ``lora_B`` segment
    suffices. Multiple LoRA segments per key (uncommon) all get the
    adapter name inserted, matching PEFT's symmetric behaviour.
    """
    parts = key.split(".")
    new_parts = []
    for p in parts:
        new_parts.append(p)
        if p in _LORA_EXACT_PARTS:
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
    parallel_plan: Optional["ParallelPlan"] = None,
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
        parallel_plan: Optional ExtraParallel plan; same role as in
            :func:`rank0_load_and_broadcast_adapter_weights` -- forwarded to
            ``_dispatch_parameter`` so EP-sharded LoRA tensors are sliced
            from the disk-side full ``[E, ...]`` shape down to the local
            ``[E_local, ...]`` shape before the DTensor copy.
    """
    from peft import load_peft_weights

    adapter_name = _read_adapter_name(adapter_path)
    raw_sd = load_peft_weights(adapter_path, device=init_device)
    for name, tensor in raw_sd.items():
        name = _remap_adapter_key(name, adapter_name)
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)


# fsdp2 init lora parameters during post_process_after_weight_loading
def _init_lora_parameter(module: "nn.Module", name: str):
    """Dispatch ``reset_lora_parameters`` for one LoRA tensor name.

    Walks the module tree along ``name``'s pieces and reacts to two cases:

    * **MoE-LoRA wrapper**: when an ancestor is a
      :class:`~veomni.utils.moe_lora.LoraSharedExperts` /
      :class:`~veomni.utils.moe_lora.LoraIndependentExperts`, dispatch
      ``reset_lora_parameters(init_lora_weights=True)`` on the wrapper.
      The wrapper re-initialises every adapter on every spec idempotently
      so we dispatch once and return — avoids the per-adapter loop that
      would re-init the same tensors many times when iterated key-by-key.
    * **Standard PEFT LoRA layer**: when the immediate ``lora_A`` /
      ``lora_B`` parent has ``reset_lora_parameters``, loop over its
      adapters and reset each. ``lora_B`` is reset implicitly by the
      ``lora_A`` reset, so we only need to dispatch on ``lora_A`` keys.
    """
    from .moe_lora import is_lora_moe_experts

    pieces = name.split(".")
    cursor = module
    for piece in pieces:
        if is_lora_moe_experts(cursor):
            cursor.reset_lora_parameters(init_lora_weights=True)
            return
        if piece.startswith("lora_"):
            break
        cursor = getattr(cursor, piece)

    if is_lora_moe_experts(cursor):
        cursor.reset_lora_parameters(init_lora_weights=True)
        return

    if "lora_A" in name and hasattr(cursor, "reset_lora_parameters"):
        for adapter in getattr(cursor, "lora_A", {}).keys():
            cursor.reset_lora_parameters(adapter, init_lora_weights=True)


# fsdp2 meta device rank0 load and broadcast adapter weights
@torch.no_grad()
def rank0_load_and_broadcast_adapter_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    adapter_path: str,
    init_device: Literal["cpu", "cuda", "npu"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    parameter_names_to_load: Optional[set] = None,
    parallel_plan: Optional["ParallelPlan"] = None,
) -> None:
    """Rank-0 loads PEFT adapter weights from disk then broadcasts to all ranks.

    Args:
        parameter_names_to_load: If provided, each successfully loaded parameter
            name is discarded from this set so that ``post_process_after_weight_loading``
            does not re-initialise adapter weights that have already been loaded.
        parallel_plan: Optional ExtraParallel plan. When set, ``_dispatch_parameter``
            calls ``parallel_plan.shard_tensor`` on each broadcast adapter tensor
            so EP-sharded LoRA params (those registered by
            :func:`~veomni.distributed.parallel_plan._extend_plan_for_moe_lora_independent`
            for :class:`~veomni.utils.moe_lora.LoraIndependentExperts`) get
            sliced from the disk-side full ``[E, ...]`` shape down to the
            local ``[E_local, ...]`` shape before the DTensor ``.copy_()``.
            Without this the copy fails with a sharding-propagation error
            (full-shape source vs sliced-shape destination).
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
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
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
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)
        del tensor

    logger.info_rank0(f"rank0_broadcast_adapter_weights: loaded {num_keys} adapter param(s)")
