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

"""AdamW whose moment states live on the host and are streamed to the device in batches.

Memory shape of a training step
-------------------------------
A vanilla ``torch.optim.AdamW`` keeps ``exp_avg`` and ``exp_avg_sq`` resident on
the accelerator for the whole run: two extra copies of every parameter. This
optimizer keeps those tensors on pinned host memory instead and only
materializes the states of one *batch* of parameters at a time, sized from the
currently free device memory. Peak device memory therefore drops to roughly
``params + grads + one batch of states`` at the cost of host<->device copies
once per step.

Scope (v1)
----------
* Ascend NPU only, FSDP2 only. Dense models and MoE with ExtraParallel (e.g.
  expert parallelism) are supported: under ExtraParallel the factory builds one
  ``SwapAdamW`` per parameter group inside a ``MultiOptimizer``. The factory
  (``veomni.optim.optimizer.build_optimizer``) enforces the device/mode scope.
* Checkpoint save/load reads and writes the host buffers directly: the state
  holders keep their DTensor sharding metadata (mesh, placements, global shape)
  but their local storage is pointed at the live host buffer, so DCP never
  materializes optimizer states on the accelerator. See ``state_dict`` /
  ``load_state_dict``.
* The update kernel is ``torch._fused_adamw_`` on the accelerator (Ascend NPU
  registers it), applied per param group within each swapped-in batch. On CPU
  (unit tests) it falls back to plain elementwise AdamW math, which is
  numerically equivalent.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

from ..utils import logging
from ..utils.device import (
    create_stream,
    get_current_stream,
    get_device_id,
    get_device_type,
    get_torch_device,
    switch_to_specified_stream,
)
from ..utils.dtensor_utils import rewrap_dtensor_local


logger = logging.get_logger(__name__)


def _to_local(tensor: Tensor) -> Tensor:
    """Return the local shard of a DTensor, or the tensor itself otherwise."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _storage_nbytes(tensor: Tensor) -> int:
    return tensor.untyped_storage().nbytes()


class SwapAdamW(Optimizer):
    """AdamW with host-resident moment states streamed to device in batches.

    The optimizer state layout matches ``torch.optim.AdamW``: each parameter owns
    ``exp_avg`` and ``exp_avg_sq`` (plus ``max_exp_avg_sq`` when ``amsgrad``).
    Those tensors exist for the whole run, but their device storage is released
    between steps by resizing the underlying (untyped) storage to zero; the
    values are preserved in pinned host buffers.

    Args:
        params: Parameters or parameter groups, as for ``torch.optim.Optimizer``.
        mem_fraction_static: Fraction of the *currently free* device memory used
            to size one batch of streamed-in states. Must fit at least the
            largest single parameter's states.
        pin_memory: Pin the host-side state buffers. Improves transfer bandwidth
            at the cost of page-locked host memory.
        chunk_bytes: State bytes updated (and swapped out) per pipeline chunk.
            The chunk boundary lets a chunk's swap-out overlap the next chunk's
            swap-in; a byte budget keeps a chunk from being one huge parameter or
            hundreds of tiny ones.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        mem_fraction_static: float = 0.8,
        pin_memory: bool = True,
        chunk_bytes: int = 128 << 20,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < mem_fraction_static <= 1.0:
            raise ValueError(f"mem_fraction_static must be in (0, 1], got {mem_fraction_static}")
        if chunk_bytes < 1:
            raise ValueError(f"chunk_bytes must be >= 1, got {chunk_bytes}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.mem_fraction_static = float(mem_fraction_static)
        self.pin_memory = bool(pin_memory)
        # Upper bound on the state bytes updated (and swapped out) per pipeline
        # chunk. Smaller chunks let the swap-out of one overlap the swap-in of the
        # next; a byte budget rather than a param count keeps a chunk from
        # becoming one huge parameter on large models, or hundreds of tiny ones
        # (launch overhead) on models with many small parameters.
        self.chunk_bytes = int(chunk_bytes)

        # Each optimizer owns two copy streams: H2D states on ``_in_stream`` and D2H
        # on ``_out_stream``. The two directions then overlap instead of serializing
        # on a single stream. Leaves of an EP MultiOptimizer step one after another
        # (serial host loop, every leaf drains its streams at the end of its own
        # ``step``), so there is no cross-leaf copy to share.
        self._device_type = get_device_type()
        self._use_streams = self._device_type != "cpu"
        if self._use_streams:
            self._in_stream: Optional[Any] = create_stream()
            self._out_stream: Optional[Any] = create_stream()
        else:
            self._in_stream = None
            self._out_stream = None

        self._state_keys: Tuple[str, ...] = ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ())
        # Use the native fused AdamW kernel on real accelerators (Ascend NPU
        # registers ``torch._fused_adamw_``); keep the elementwise path on CPU so
        # the optimizer stays unit-testable without an accelerator.
        self._use_fused = self._use_streams and hasattr(torch, "_fused_adamw_")
        self._host_states: Dict[Tensor, Dict[str, Tensor]] = {}
        self._device_states: Dict[Tensor, Dict[str, Tensor]] = {}
        self._swap_in_events: Dict[Tensor, Any] = {}
        # Storages whose device->host copy is still in flight on ``_out_stream``.
        # They must not be released until that stream has drained, or the caching
        # allocator could hand the block out while the copy is still reading it.
        self._pending_free: List[Tensor] = []

        self._init_states()

    # ------------------------------------------------------------------ build

    def _init_states(self) -> None:
        """Move every parameter's moment state to host and free its device storage."""
        for group in self.param_groups:
            amsgrad = group["amsgrad"]
            for p in group["params"]:
                if not p.requires_grad or p in self._host_states:
                    continue

                host_state: Dict[str, Tensor] = {}
                device_state: Dict[str, Tensor] = {}
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key == "max_exp_avg_sq" and not amsgrad:
                        continue
                    dtensor = torch.zeros_like(p)
                    self.state[p][key] = dtensor
                    local_shard = _to_local(dtensor)
                    host_tensor = torch.zeros(
                        tuple(local_shard.shape),
                        dtype=local_shard.dtype,
                        device="cpu",
                        pin_memory=self.pin_memory,
                    )
                    host_state[key] = host_tensor
                    device_state[key] = local_shard
                    local_shard.untyped_storage().resize_(0)

                if "step" not in self.state[p]:
                    self.state[p]["step"] = torch.tensor(0.0)
                self._host_states[p] = host_state
                self._device_states[p] = device_state

    # -------------------------------------------------------------- checkpoint

    def _bind_host_states(self) -> None:
        """Rebind every state holder to its host buffer for DCP to read.

        The DTensor shells in ``self.state`` keep their mesh/placements but their
        local is the live host buffer, so DCP reads state off the accelerator. The
        live buffer is bound, not copied - the base (device) optimizer hands DCP
        live state too, so a snapshot would only add a full extra host copy.
        """
        for p, host_state in self._host_states.items():
            for key, host_tensor in host_state.items():
                holder = self.state[p][key]
                if isinstance(holder, DTensor):
                    self.state[p][key] = rewrap_dtensor_local(
                        host_tensor,
                        mesh=holder.device_mesh,
                        placements=holder.placements,
                        shape=torch.Size(holder.shape),
                        stride=holder.stride(),
                    )
                else:
                    self.state[p][key] = host_tensor

    def state_dict(self):
        """Optimizer state with all values read from host memory (see ``_bind_host_states``)."""
        self._bind_host_states()
        return super().state_dict()

    def load_state_dict(self, state_dict) -> None:
        """Load values into the host buffers without materializing them on device.

        The base implementation casts every value to ``param.device`` (moving the
        host buffers onto the accelerator), so the mapping is done here instead.
        State is aligned to the current parameters by param-group position, as the
        base class does; param-group hyperparameters are merged from the checkpoint.
        """
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        id_map = {}
        for group, saved_group in zip(groups, saved_groups):
            if len(group["params"]) != len(saved_group["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match the optimizer's group"
                )
            id_map.update(zip(saved_group["params"], group["params"]))
            group.update({k: v for k, v in saved_group.items() if k != "params"})
        for key, value in state_dict["state"].items():
            param = id_map.get(key)
            if param is not None:
                self._load_param_state(param, value)

    def _load_param_state(self, p: Tensor, value: dict) -> None:
        host_state = self._host_states.get(p)
        if host_state is None:
            return
        for key, loaded in value.items():
            if key == "step":
                # Own the value rather than aliasing the checkpoint's tensor:
                # ``_bump_step`` mutates it in place, so an alias would corrupt it.
                if isinstance(loaded, DTensor):
                    loaded = _to_local(loaded)
                scalar = loaded.item() if isinstance(loaded, torch.Tensor) else loaded
                step = self.state[p]["step"]
                if isinstance(step, torch.Tensor):
                    step.fill_(scalar)
                else:
                    self.state[p]["step"] = scalar
            elif key in host_state and loaded is not None:
                source = _to_local(loaded) if isinstance(loaded, DTensor) else loaded
                host_state[key].copy_(source.to(host_state[key].device))

    def swap_all_to_device(self) -> None:
        """Swap in every parameter's states to the device (used by the tests; DCP save reads host directly)."""
        self._release_pending_storages()
        for p, host_state in self._host_states.items():
            for key, host_tensor in host_state.items():
                local_shard = self._device_states[p][key]
                if _storage_nbytes(local_shard) == 0:
                    local_shard.untyped_storage().resize_(host_tensor.numel() * host_tensor.element_size())
                local_shard.copy_(host_tensor)

    def swap_all_to_host(self) -> None:
        """Copy every parameter's states to host and free device storage."""
        self._release_pending_storages()
        for p, host_state in self._host_states.items():
            for key, host_tensor in host_state.items():
                local_shard = self._device_states[p][key]
                if _storage_nbytes(local_shard) == 0:
                    continue
                host_tensor.copy_(local_shard)
                local_shard.untyped_storage().resize_(0)

    # ------------------------------------------------------------- step pieces

    def _param_state_bytes(self, p: Tensor) -> int:
        local_shard = _to_local(p)
        return sum(local_shard.numel() * local_shard.element_size() for _ in self._state_keys)

    def _device_free_bytes(self) -> int:
        """Bytes this process can still allocate on the device.

        ``total - allocated`` includes cached-but-unused allocator blocks that
        ``mem_get_info`` under-reports (freed states return to the pool, not the
        device), which would collapse the swap batch after the first big swap-in.
        """
        device = get_torch_device()
        try:
            total = int(device.get_device_properties(get_device_id()).total_memory)
            return total - int(device.memory_allocated())
        except Exception:
            try:
                free, _total = device.mem_get_info()
                return int(free)
            except (AttributeError, RuntimeError):
                pass
        logger.warning_once("SwapAdamW could not determine free device memory; swapping states in one batch.")
        return 1 << 62  # pragma: no cover

    def _batch_budget_bytes(self) -> int:
        if not self._use_streams:
            return 1 << 62
        return max(int(self._device_free_bytes() * self.mem_fraction_static), 0)

    def _release_pending_storages(self) -> None:
        """Release device storages whose host copies have finished."""
        if not self._pending_free:
            return
        if self._use_streams:
            self._out_stream.synchronize()
        for local_shard in self._pending_free:
            if _storage_nbytes(local_shard) != 0:
                local_shard.untyped_storage().resize_(0)
        self._pending_free.clear()

    def _swap_in_param(self, p: Tensor) -> None:
        """Copy ``p``'s states from host to device (caller owns the stream context)."""
        host_state = self._host_states[p]
        for key, host_tensor in host_state.items():
            local_shard = self._device_states[p][key]
            if _storage_nbytes(local_shard) == 0:
                local_shard.untyped_storage().resize_(host_tensor.numel() * host_tensor.element_size())
            local_shard.copy_(host_tensor, non_blocking=self._use_streams)
        if self._use_streams:
            self._swap_in_events[p] = get_current_stream().record_event()

    def _swap_out_param(self, p: Tensor) -> None:
        """Queue a device->host copy and defer the storage free (caller owns the stream)."""
        for key, host_tensor in self._host_states[p].items():
            local_shard = self._device_states[p][key]
            if _storage_nbytes(local_shard) == 0:
                continue
            host_tensor.copy_(local_shard, non_blocking=self._use_streams)
            self._pending_free.append(local_shard)

    def _swap_out_param_on_out_stream(self, p: Tensor) -> None:
        """Swap ``p`` back to host on the out stream, after the compute stream is done."""
        if not self._use_streams:
            self._swap_out_param(p)
            return
        compute_stream = get_current_stream()
        with switch_to_specified_stream(self._out_stream):
            get_current_stream().wait_stream(compute_stream)
            self._swap_out_param(p)

    def _wait_swapped_in(self, p: Tensor) -> None:
        event = self._swap_in_events.pop(p, None)
        if event is not None:
            get_current_stream().wait_event(event)

    def _swap_in_batch(self, params: Sequence[Tensor], index: int) -> int:
        """Swap ``params[index:]`` in to the device up to the memory budget.

        Returns the exclusive end index of the swapped-in range. The caller
        updates exactly those parameters and swaps them back out before asking
        for the next batch, so peak device state stays bounded by the budget.
        """
        self._release_pending_storages()
        stream_context = switch_to_specified_stream(self._in_stream) if self._use_streams else nullcontext()
        bytes_moved = 0
        budget = self._batch_budget_bytes()
        with stream_context:
            while index < len(params):
                param_bytes = self._param_state_bytes(params[index])
                if bytes_moved != 0 and bytes_moved + param_bytes > budget:
                    break
                self._swap_in_param(params[index])
                bytes_moved += param_bytes
                index += 1
        if bytes_moved == 0:
            raise RuntimeError(
                "SwapAdamW could not fit any optimizer state on device. "
                "Increase optimizer.swap_mem_fraction_static or reduce the parameter size."
            )
        return index

    def _bump_step(self, p: Tensor) -> int:
        step = self.state[p]["step"]
        if isinstance(step, torch.Tensor):
            step = step.add_(1)
            return int(step.item())
        step = int(step) + 1
        self.state[p]["step"] = step
        return step

    def _update_param(self, p: Tensor, group: Dict[str, Any]) -> None:
        local_p = _to_local(p)
        grad = _to_local(p.grad)
        if group.get("maximize", False):
            grad = -grad
        device_state = self._device_states[p]

        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        step = self._bump_step(p)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        if weight_decay != 0:
            local_p.mul_(1 - lr * weight_decay)

        exp_avg = device_state["exp_avg"]
        exp_avg_sq = device_state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group["amsgrad"]:
            max_exp_avg_sq = device_state["max_exp_avg_sq"]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

        local_p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

    def _update_group_fused(self, params: Sequence[Tensor], group: Dict[str, Any]) -> None:
        """Update a chunk of parameters that share one param group in a single fused call.

        ``torch._fused_adamw_`` takes one scalar hyperparameter set per call, so a
        batch spanning several groups is split into one call per group. Batching
        the call (instead of one call per parameter) amortizes the kernel launch.
        """
        if not params:
            return
        amsgrad = bool(group["amsgrad"])
        beta1, beta2 = group["betas"]
        device = _to_local(params[0]).device
        steps = []
        for p in params:
            step = self._bump_step(p)
            steps.append(torch.tensor(step, dtype=torch.int64, device=device))
        torch._fused_adamw_(
            [_to_local(p) for p in params],
            [_to_local(p.grad) for p in params],
            [self._device_states[p]["exp_avg"] for p in params],
            [self._device_states[p]["exp_avg_sq"] for p in params],
            [self._device_states[p]["max_exp_avg_sq"] for p in params] if amsgrad else [],
            steps,
            lr=group["lr"],
            beta1=beta1,
            beta2=beta2,
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            amsgrad=amsgrad,
            maximize=group.get("maximize", False),
        )

    def _update_chunk(self, params: Sequence[Tensor], group: Dict[str, Any]) -> None:
        """Update one chunk of parameters that share ``group``.

        Uses the native fused kernel when available, else the elementwise path
        (CPU tests). Callers keep a chunk within a single param group because a
        fused call takes one set of hyperparameters.
        """
        if self._use_fused:
            self._update_group_fused(params, group)
        else:
            for p in params:
                self._update_param(p, group)

    # ------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active: List[Tensor] = []
        group_by_param: Dict[Tensor, Dict[str, Any]] = {}
        for group in self.param_groups:
            for param in group["params"]:
                group_by_param[param] = group
                if not param.requires_grad or param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SwapAdamW does not support sparse gradients")
                active.append(param)

        index = 0
        num_params = len(active)
        while index < num_params:
            batch_end = self._swap_in_batch(active, index)
            group_start = index
            while group_start < batch_end:
                # A fused update takes one param group's hyperparameters, so a
                # chunk never crosses a group boundary. Within a group it is split
                # into ``chunk_bytes``-sized stages so the swap-out of one chunk
                # overlaps the swap-in of the next (and the compute in between).
                group = group_by_param[active[group_start]]
                group_end = group_start + 1
                while group_end < batch_end and group_by_param[active[group_end]] is group:
                    group_end += 1
                chunk_start = group_start
                while chunk_start < group_end:
                    chunk_end = chunk_start
                    acc_bytes = 0
                    while chunk_end < group_end:
                        param_bytes = self._param_state_bytes(active[chunk_end])
                        if chunk_end > chunk_start and acc_bytes + param_bytes > self.chunk_bytes:
                            break
                        acc_bytes += param_bytes
                        chunk_end += 1
                    chunk = active[chunk_start:chunk_end]
                    for param in chunk:
                        self._wait_swapped_in(param)
                    self._update_chunk(chunk, group)
                    for param in chunk:
                        self._swap_out_param_on_out_stream(param)
                    chunk_start = chunk_end
                group_start = group_end
            index = batch_end

        # Release the final batch's device storages before the next forward runs,
        # so the states are off the accelerator for the whole inter-step window.
        self._release_pending_storages()

        return loss


__all__ = ["SwapAdamW"]
