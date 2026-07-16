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

"""FSDP2 meta-device adapter weight loading + init for the native LoRA stack.

Native replacements for the ``peft``-based helpers previously in
``veomni/utils/lora_utils.py``:

* :func:`build_lora_key_overrides` — bare base-key -> wrapped-FQN map so a base
  checkpoint loads into ``...base_layer.weight`` destinations.
* :func:`load_lora_weights` — every rank reads the PEFT-format adapter file.
* :func:`rank0_load_and_broadcast_lora_weights` — rank-0 reads then broadcasts.
* :func:`init_lora_parameter` — post-load kaiming/zero init for un-loaded LoRA
  params, with a missing-name-set guard that prevents clobbering loaded weights.

All adapter files are read natively (safetensors / torch), never via ``peft``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed.parallel_state import get_parallel_state
from ..models.module_utils import BroadcastMetadata, _dispatch_parameter
from ..utils import logging
from ..utils.device import get_device_type
from .state_dict import insert_adapter_name, load_adapter_state_dict


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..distributed.parallel_plan import ParallelPlan

logger = logging.get_logger(__name__)

_DEFAULT_ADAPTER = "default"


def _is_moe_lora_wrapper(module: nn.Module) -> bool:
    """True for either MoE-LoRA wrapper flavour (lazy import; False pre-Phase-2)."""
    try:
        from .moe_layers import is_lora_moe_experts
    except Exception:
        return False
    return is_lora_moe_experts(module)


def build_lora_key_overrides(model: nn.Module) -> dict[str, str]:
    """Map bare base-model param names to their wrapped ``base_layer`` FQNs.

    When a base checkpoint is loaded into a LoRA-wrapped model, each target
    ``nn.Linear`` is replaced by a :class:`~veomni.lora.layers.LoraLinear`
    storing the original weight under ``base_layer``. This produces the
    remapping so callers can translate checkpoint keys, e.g.::

        "layers.0.self_attn.q_proj.weight"
        -> "base_model.model.layers.0.self_attn.q_proj.base_layer.weight"

    MoE-LoRA wrappers lift a bare fused ``nn.Parameter`` under
    ``<spec>.base_layer.weight`` and mark the holder ``_is_bare_param_holder``
    so the source key carries no ``.weight`` suffix.
    """
    overrides: dict[str, str] = {}
    for fqn, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        inner = fqn[len("base_model.model.") :] if fqn.startswith("base_model.model.") else fqn
        inner_dot = inner + ("." if inner else "")
        wrap_dot = fqn + ("." if fqn else "") + "base_layer."
        if getattr(module.base_layer, "_is_bare_param_holder", False):
            overrides[inner] = wrap_dot + "weight"
            continue
        for pname, _ in module.base_layer.named_parameters():
            overrides[inner_dot + pname] = wrap_dot + pname
        for bname, _ in module.base_layer.named_buffers():
            overrides[inner_dot + bname] = wrap_dot + bname
    return overrides


def make_peft_key_mapper(model: nn.Module, is_peft_model: bool) -> Callable[[str], str]:
    """Return a fn mapping a *bare* base-model FQN to its live-model destination.

    Centralises the checkpoint-key remap every weight loader
    (:func:`~veomni.models.module_utils.load_model_weights` /
    ``load_model_weights_ep_sharded`` / ``rank0_load_and_broadcast_weights``)
    needs when a base checkpoint is loaded into a LoRA-wrapped model:

    * ``is_peft_model=False`` -> identity (the checkpoint key is already the
      live-model key).
    * ``is_peft_model=True``  -> :func:`build_lora_key_overrides` remap (wrapped
      targets go to ``...base_layer.weight``) with a plain ``base_model.model.``
      prefix fallback for un-wrapped params.

    Apply it *after* ``maybe_convert_checkpoint_tensor`` so converter-produced
    merged keys (e.g. the Qwen3-MoE per-expert -> fused ``...experts.gate_up_proj``)
    also flow through the ``base_layer.weight`` rename when their experts module is
    wrapped by ``LoraSharedExperts`` / ``LoraIndependentExperts``.
    """
    if not is_peft_model:
        return lambda bare_name: bare_name
    overrides = build_lora_key_overrides(model)
    return lambda bare_name: overrides.get(bare_name, "base_model.model." + bare_name)


@torch.no_grad()
def load_lora_weights(
    model: nn.Module | PreTrainedModel,
    adapter_path: str,
    init_device: str | None = None,
    dtensor_factory: Callable[[torch.Tensor, Any, Any], torch.Tensor] | None = None,
    parameter_names_to_load: set | None = None,
    parallel_plan: ParallelPlan | None = None,
    adapter_name: str = _DEFAULT_ADAPTER,
) -> None:
    """Load adapter weights on every rank (shared-filesystem path).

    Reads the PEFT-format adapter file natively, remaps on-disk keys
    (``...lora_A.weight``) to live-model keys (``...lora_A.<adapter>.weight``),
    and dispatches tensors into the (possibly sharded) model.
    """
    init_device = init_device or get_device_type()
    raw_sd = load_adapter_state_dict(adapter_path, device=init_device)
    for name, tensor in raw_sd.items():
        name = insert_adapter_name(name, adapter_name)
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)


@torch.no_grad()
def rank0_load_and_broadcast_lora_weights(
    model: nn.Module | PreTrainedModel,
    adapter_path: str,
    init_device: str | None = None,
    dtensor_factory: Callable[[torch.Tensor, Any, Any], torch.Tensor] | None = None,
    parameter_names_to_load: set | None = None,
    parallel_plan: ParallelPlan | None = None,
    adapter_name: str = _DEFAULT_ADAPTER,
) -> None:
    """Rank-0 reads the adapter file, then broadcasts each tensor to all ranks."""
    init_device = init_device or get_device_type()
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    adapter_sd: dict[str, torch.Tensor] = {}
    if global_rank == 0:
        raw_sd = load_adapter_state_dict(adapter_path, device="cpu")
        adapter_sd = {insert_adapter_name(k, adapter_name): v for k, v in raw_sd.items()}
        if adapter_sd:
            first_raw = next(iter(raw_sd))
            first_remapped = next(iter(adapter_sd))
            logger.info_rank0(
                f"Loaded {len(adapter_sd)} adapter weight(s) from {adapter_path}, "
                f"key remap example: {first_raw} -> {first_remapped}"
            )

    if not dist.is_available() or not dist.is_initialized():
        for name, tensor in adapter_sd.items():
            _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        return

    global_rank = get_parallel_state().global_rank
    torch_device = torch.device(init_device)

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

        if global_rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device=torch_device)

        start_time = time.perf_counter()
        dist.broadcast(tensor, src=0)
        logger.info_rank0(
            f"{name=}, {shape=}, {dtype=}, broadcast time (ms): {1000 * (time.perf_counter() - start_time)}"
        )
        _dispatch_parameter(model, name, tensor, dtensor_factory, parallel_plan=parallel_plan)
        if parameter_names_to_load is not None:
            parameter_names_to_load.discard(name)
        del tensor

    logger.info_rank0(f"rank0_load_and_broadcast_lora_weights: loaded {num_keys} adapter param(s)")


def init_lora_parameter(
    model: nn.Module,
    name: str,
    parameter_names_left: set[str] | None = None,
) -> None:
    """Initialise one un-loaded LoRA tensor (kaiming ``A`` / zero ``B``).

    Walks ``model`` along ``name`` to the owning LoRA module and calls its
    ``reset_lora_parameters``. For MoE wrappers the decision is made at
    *wrapper* granularity using ``parameter_names_left`` (see
    :func:`_reset_moe_wrapper`): reset only when every LoRA tensor on that
    wrapper is missing; never touch a fully- or partially-loaded wrapper.
    """
    module: nn.Module = model
    prefix_parts: list[str] = []
    for piece in name.split("."):
        if _is_moe_lora_wrapper(module):
            _reset_moe_wrapper(module, ".".join(prefix_parts), parameter_names_left)
            return
        if piece.startswith("lora_"):
            break
        if not hasattr(module, piece):
            return
        module = getattr(module, piece)
        prefix_parts.append(piece)

    if _is_moe_lora_wrapper(module):
        _reset_moe_wrapper(module, ".".join(prefix_parts), parameter_names_left)
        return

    if "lora_A" in name and hasattr(module, "reset_lora_parameters"):
        for adapter in getattr(module, "lora_A", {}).keys():
            module.reset_lora_parameters(adapter, init_lora_weights=True)


def _moe_wrapper_lora_names(module: nn.Module, wrapper_prefix: str) -> list[str]:
    """FQNs of every LoRA ``A``/``B`` tensor owned by a MoE wrapper."""
    names: list[str] = []
    for local_name, _ in module.named_parameters():
        parts = local_name.split(".")
        if "lora_A" not in parts and "lora_B" not in parts:
            continue
        names.append(f"{wrapper_prefix}.{local_name}" if wrapper_prefix else local_name)
    return names


def _reset_moe_wrapper(
    module: nn.Module,
    wrapper_prefix: str,
    parameter_names_left: set[str] | None,
) -> None:
    """Fresh-init a MoE wrapper's LoRA params (kaiming ``A`` / zero ``B``).

    Decision is at wrapper granularity against ``parameter_names_left`` (the
    set of FQNs the weight loader did not fill). After ``model.to_empty(...)``
    every param is non-meta garbage, so a meta-device guard cannot tell "fresh"
    from "already loaded"; the missing-name set is the source of truth:

    * **all** wrapper LoRA names missing → reset once (idempotent via
      ``_lora_fresh_reset_done``);
    * **none** missing → no-op (loaded weights must not be clobbered);
    * **subset** missing → raise (partial adapter resume is fail-closed).
    """
    if getattr(module, "_lora_fresh_reset_done", False):
        return

    lora_names = _moe_wrapper_lora_names(module, wrapper_prefix)
    if not lora_names:
        return

    left = parameter_names_left if parameter_names_left is not None else set()
    missing = [n for n in lora_names if n in left]
    if not missing:
        return
    if len(missing) != len(lora_names):
        loaded = sorted(set(lora_names) - set(missing))
        raise RuntimeError(
            f"MoE LoRA wrapper {wrapper_prefix!r} is only partially loaded: "
            f"{len(missing)}/{len(lora_names)} LoRA tensor(s) missing "
            f"(e.g. {missing[0]!r}). Refusing to reset the wrapper because that "
            f"would clobber already-loaded weights "
            f"(e.g. {loaded[0]!r}). Provide a complete adapter or omit "
            f"adapter_path for a fresh init."
        )

    module.reset_lora_parameters(init_lora_weights=True)
    module._lora_fresh_reset_done = True
