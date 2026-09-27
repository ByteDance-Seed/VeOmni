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
* Checkpoint save/load is **not** supported: the state tensors are intentionally
  left with zero-sized storage between steps, so writing them out would produce a
  corrupt checkpoint. ``veomni.checkpoint.dcp_checkpointer.OptimizerState``
  refuses to persist this optimizer rather than silently writing zeros.
* The update kernel is plain elementwise AdamW math, not
  ``torch._fused_adamw_`` (which the Ascend stack does not expose). This keeps
  the optimizer testable on CPU and independent of any private fused kernel.
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
    """

    _is_swap_optimizer = True

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
        stream: Optional[Any] = None,
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.mem_fraction_static = float(mem_fraction_static)
        self.pin_memory = bool(pin_memory)

        self._device_type = get_device_type()
        # ``stream`` lets several swap optimizers (e.g. the EP / non-EP leaves of
        # a MultiOptimizer) share one stream; otherwise each creates its own.
        self._use_streams = stream is not None or self._device_type != "cpu"
        if stream is not None:
            self._swap_stream: Optional[Any] = stream
        else:
            self._swap_stream = create_stream() if self._use_streams else None

        self._state_keys: Tuple[str, ...] = ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ())
        self._cpu_states: Dict[Tensor, Dict[str, Tensor]] = {}
        self._device_states: Dict[Tensor, Dict[str, Tensor]] = {}
        self._materialize_events: Dict[Tensor, Any] = {}
        # Storages whose device->host copy is still in flight on ``_swap_stream``.
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
                if not p.requires_grad or p in self._cpu_states:
                    continue

                cpu_state: Dict[str, Tensor] = {}
                device_state: Dict[str, Tensor] = {}
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key == "max_exp_avg_sq" and not amsgrad:
                        continue
                    dtensor = torch.zeros_like(p)
                    self.state[p][key] = dtensor
                    local = _to_local(dtensor)
                    cpu = torch.zeros(
                        tuple(local.shape),
                        dtype=local.dtype,
                        device="cpu",
                        pin_memory=self.pin_memory,
                    )
                    cpu_state[key] = cpu
                    device_state[key] = local
                    local.untyped_storage().resize_(0)

                self.state[p].setdefault("step", 0)
                self._cpu_states[p] = cpu_state
                self._device_states[p] = device_state

    # -------------------------------------------------------------- checkpoint

    def swap_all_to_device(self) -> None:
        """Materialize every parameter's states on device (used around save/export)."""
        self._flush_pending_frees()
        for p, cpu_state in self._cpu_states.items():
            for key, cpu in cpu_state.items():
                local = self._device_states[p][key]
                if _storage_nbytes(local) == 0:
                    local.untyped_storage().resize_(cpu.numel() * cpu.element_size())
                local.copy_(cpu)

    def swap_all_to_host(self) -> None:
        """Copy every parameter's states to host and free device storage."""
        self._flush_pending_frees()
        for p, cpu_state in self._cpu_states.items():
            for key, cpu in cpu_state.items():
                local = self._device_states[p][key]
                if _storage_nbytes(local) == 0:
                    continue
                cpu.copy_(local)
                local.untyped_storage().resize_(0)

    # ------------------------------------------------------------- step pieces

    def _state_bytes(self, p: Tensor) -> int:
        local = _to_local(p)
        return sum(local.numel() * local.element_size() for _ in self._state_keys)

    def _free_device_bytes(self) -> int:
        device = get_torch_device()
        try:
            free, _total = device.mem_get_info()
            return int(free)
        except (AttributeError, RuntimeError):
            pass
        try:
            total = int(device.get_device_properties(get_device_id()).total_memory)
            return total - int(device.memory_allocated())
        except Exception:  # pragma: no cover - last-resort fallback
            logger.warning_once("SwapAdamW could not determine free device memory; swapping states in one batch.")
            return 1 << 62

    def _swap_budget_bytes(self) -> int:
        if not self._use_streams:
            return 1 << 62
        return max(int(self._free_device_bytes() * self.mem_fraction_static), 0)

    def _flush_pending_frees(self) -> None:
        """Release device storages whose host copies have finished."""
        if not self._pending_free:
            return
        if self._use_streams:
            self._swap_stream.synchronize()
        for local in self._pending_free:
            if _storage_nbytes(local) != 0:
                local.untyped_storage().resize_(0)
        self._pending_free.clear()

    def _materialize_param(self, p: Tensor) -> None:
        cpu_state = self._cpu_states[p]
        for key, cpu in cpu_state.items():
            local = self._device_states[p][key]
            if _storage_nbytes(local) == 0:
                local.untyped_storage().resize_(cpu.numel() * cpu.element_size())
            local.copy_(cpu, non_blocking=self._use_streams)
        if self._use_streams:
            self._materialize_events[p] = get_current_stream().record_event()

    def _offload_param(self, p: Tensor) -> None:
        """Queue a device->host copy on the swap stream and defer the storage free."""
        for key, cpu in self._cpu_states[p].items():
            local = self._device_states[p][key]
            if _storage_nbytes(local) == 0:
                continue
            cpu.copy_(local, non_blocking=self._use_streams)
            self._pending_free.append(local)

    def _wait_materialized(self, p: Tensor) -> None:
        event = self._materialize_events.pop(p, None)
        if event is not None:
            get_current_stream().wait_event(event)

    def _swap_batch_to_device(self, params: Sequence[Tensor], index: int) -> int:
        """Materialize ``params[index:]`` up to the memory budget.

        Returns the exclusive end index of the materialized range. The caller
        updates exactly those parameters and offloads them again before asking
        for the next batch, so peak device state stays bounded by the budget.
        """
        self._flush_pending_frees()
        stream_context = switch_to_specified_stream(self._swap_stream) if self._use_streams else nullcontext()
        bytes_moved = 0
        budget = self._swap_budget_bytes()
        with stream_context:
            while index < len(params):
                param_bytes = self._state_bytes(params[index])
                if bytes_moved != 0 and bytes_moved + param_bytes > budget:
                    break
                self._materialize_param(params[index])
                bytes_moved += param_bytes
                index += 1
        if bytes_moved == 0:
            raise RuntimeError(
                "SwapAdamW could not fit any optimizer state on device. "
                "Increase optimizer.swap_mem_fraction_static or reduce the parameter size."
            )
        return index

    def _update_param(self, p: Tensor, group: Dict[str, Any]) -> None:
        state = self.state[p]
        local_p = _to_local(p)
        grad = _to_local(p.grad)
        device_state = self._device_states[p]

        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        step = int(state["step"]) + 1
        state["step"] = step
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
            batch_end = self._swap_batch_to_device(active, index)
            for cursor in range(index, batch_end):
                param = active[cursor]
                self._wait_materialized(param)
                self._update_param(param, group_by_param[param])
                if self._use_streams:
                    compute_stream = get_current_stream()
                    with switch_to_specified_stream(self._swap_stream):
                        get_current_stream().wait_stream(compute_stream)
                        self._offload_param(param)
                else:
                    self._offload_param(param)
            index = batch_end

        # Release the final batch's device storages before the next forward runs,
        # so the states are off the accelerator for the whole inter-step window.
        self._flush_pending_frees()

        return loss


__all__ = ["SwapAdamW"]
