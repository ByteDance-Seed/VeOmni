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


import enum
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..utils.device import empty_cache, get_device_id, get_device_type


class OffloadPolicy(enum.Enum):
    OFFLOAD = 0
    KEEP_ON_GPU = 1
    IGNORE = 2


class custom_save_on_cpu(saved_tensors_hooks):
    def __init__(self, gpu_limit_in_gb: float = 0, pin_memory: bool = False, min_offload_size: int = 1024) -> None:
        self.cur_gpu_ram_in_mb = 0.0

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[OffloadPolicy, torch.device, torch.Tensor]:
            tensor_num_bytes = tensor.element_size() * tensor.nelement()
            # heuristic to skip nn.Linear.weight
            if type(tensor.grad_fn).__name__ == "TBackward0" or tensor_num_bytes <= min_offload_size:
                return (OffloadPolicy.IGNORE, tensor.device, tensor)

            if self.cur_gpu_ram_in_mb < gpu_limit_in_gb * 1024:
                self.cur_gpu_ram_in_mb += tensor_num_bytes / 1024 / 1024
                return (OffloadPolicy.KEEP_ON_GPU, tensor.device, tensor)

            if not pin_memory:
                return (OffloadPolicy.OFFLOAD, tensor.device, tensor.cpu())

            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (OffloadPolicy.OFFLOAD, tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[OffloadPolicy, torch.device, torch.Tensor]) -> torch.Tensor:
            offload_policy, device, tensor = packed

            if offload_policy == OffloadPolicy.IGNORE:
                return tensor
            elif offload_policy == OffloadPolicy.KEEP_ON_GPU:
                tensor_num_bytes = tensor.element_size() * tensor.nelement()
                self.cur_gpu_ram_in_mb -= tensor_num_bytes / 1024 / 1024
                return tensor
            else:
                return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


def build_activation_offloading_context(
    enable_activation: bool = False,
    enable_gradient_checkpointing: bool = False,
    activation_gpu_limit: float = 0.0,
) -> Tuple[Union["saved_tensors_hooks", "nullcontext"], Union["saved_tensors_hooks", "nullcontext"]]:
    model_fwd_context, model_bwd_context = nullcontext(), nullcontext()
    if enable_activation:
        # pin_memory=False since CachingHostAllocator caches pinned memory aggressively.
        # torch._C._host_emptyCache() can be used after version 2.5.
        if enable_gradient_checkpointing:
            # inter-layer activations are always offloaded when enabling gradient checkpointing to avoid potential thrashing
            model_fwd_context = custom_save_on_cpu(gpu_limit_in_gb=0.0, pin_memory=False)
            model_bwd_context = custom_save_on_cpu(gpu_limit_in_gb=activation_gpu_limit, pin_memory=False)
        else:
            model_fwd_context = custom_save_on_cpu(gpu_limit_in_gb=activation_gpu_limit, pin_memory=False)

    return model_fwd_context, model_bwd_context


def _reset_training_state(model: "torch.nn.Module") -> None:
    """Force every FSDP2 param-group on ``model`` back to the IDLE training state.

    FSDP2's per-param-group ``_training_state`` is normally advanced by the
    forward / pre-backward hooks. In RL training loops where the same actor
    module is repeatedly placed on / off GPU between rollouts and optimizer
    steps, the training state can be stranded in ``FORWARD`` / ``PRE_BACKWARD``
    if an outer code path errors out or the engine swaps to vLLM mid-call.
    The next ``reshard()`` then trips an internal assert.

    This helper is intentionally defensive: the FSDP2 module / training-state
    APIs are private, so we tolerate ``ImportError`` / ``AttributeError`` to
    avoid breaking offloading on PyTorch versions where the layout shifts.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    except ImportError:
        return

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        param_group = getattr(state, "_fsdp_param_group", None) if state is not None else None
        if param_group is None:
            continue
        try:
            param_group._training_state = TrainingState.IDLE
        except AttributeError:
            continue


@torch.no_grad()
def offload_model_to_cpu(model: "torch.nn.Module", empty_device_cache: bool = True) -> None:
    """Move a model wrapped by FSDP2 ``fully_shard`` to CPU.

    Resets any stranded FSDP2 training state, calls ``reshard()`` to drop
    unsharded parameter all-gathers, and moves remaining parameters to CPU.

    Args:
        model: Root module returned by :func:`parallelize_model_fsdp2`.
        empty_device_cache: If ``True``, calls
            :func:`veomni.utils.device.empty_cache` after the move so the
            released device memory becomes available to peers (e.g. a
            co-located vLLM rollout).
    """
    _reset_training_state(model)
    reshard = getattr(model, "reshard", None)
    if callable(reshard):
        reshard()
    model.cpu()
    if empty_device_cache and get_device_type() != "cpu":
        empty_cache()


@torch.no_grad()
def load_model_to_gpu(model: "torch.nn.Module", device: Optional[Union[str, "torch.device", int]] = None) -> None:
    """Move a model wrapped by FSDP2 ``fully_shard`` back to a device.

    Args:
        model: Root module returned by :func:`parallelize_model_fsdp2`.
        device: Target device. Defaults to the current CUDA device.
    """
    if device is None:
        device = get_device_id() if get_device_type() != "cpu" else "cpu"
    model.to(device)


def _iter_inner_optimizers(optimizer: "torch.optim.Optimizer") -> "Iterable[torch.optim.Optimizer]":
    if optimizer is None:
        return ()
    if getattr(optimizer, "_is_multi_optimizer", False):
        return optimizer.optimizers_dict.values()
    return (optimizer,)


@torch.no_grad()
def offload_optimizer(optimizer: "torch.optim.Optimizer") -> None:
    """Move all optimizer state tensors to CPU in place.

    Compatible with VeOmni's ``MultiOptimizer`` wrapper as well as a plain
    :class:`torch.optim.Optimizer`.
    """
    for opt in _iter_inner_optimizers(optimizer):
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_optimizer(
    optimizer: "torch.optim.Optimizer",
    device: Union[str, "torch.device", int],
) -> None:
    """Move all optimizer state tensors back to ``device`` in place."""
    for opt in _iter_inner_optimizers(optimizer):
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)
