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
optimizer keeps those tensors on host memory (pinned by default) instead and only
materializes the states of one *batch* of parameters at a time, sized from the
currently free device memory. Peak device memory therefore drops to roughly
``params + grads + one batch of states`` at the cost of host<->device copies
once per step.

Scope (v1)
----------
* Dense models and MoE with ExtraParallel (e.g. expert parallelism) are
  supported: under ExtraParallel the factory builds one ``SwapAdamW`` per
  parameter group inside a ``MultiOptimizer``. Validated on Ascend NPU + FSDP2.
* Checkpoint save/load reads and writes the host buffers directly (see
  ``state_dict`` / ``load_state_dict``). Proven MultiOptimizer recovery
  requires the same optimizer ownership and FSDP/EP mesh; it does not support
  elastic resharding.
* The update kernel is ``torch._fused_adamw_`` on CUDA and Ascend NPU, issued
  once per ``(device, dtype)`` bucket inside a param group, within each
  swapped-in batch (torch's own fused path groups the same way). Other backends
  (CPU unit tests, MLU, ...) take the plain elementwise AdamW path, which is
  numerically equivalent.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

from ..utils import logging
from ..utils.device import (
    FUSED_ADAMW_DEVICE_TYPES,
    STREAM_DEVICE_TYPES,
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


# Effectively unlimited budget: used when free device memory cannot be
# determined, or on CPU where there is no accelerator memory to budget, so all
# states are swapped in a single batch.
_INFINITE_BUDGET = 1 << 62


class SwapAdamW(Optimizer):
    """AdamW with host-resident moment states streamed to device in batches.

    The optimizer state layout matches ``torch.optim.AdamW``: each parameter owns
    ``exp_avg`` and ``exp_avg_sq`` (plus ``max_exp_avg_sq`` when ``amsgrad``).
    Those tensors exist for the whole run, but their device storage is released
    between steps by resizing the underlying (untyped) storage to zero; the
    values are preserved in host buffers (pinned when ``pin_memory`` is enabled).

    Args:
        params: Parameters or parameter groups, as for ``torch.optim.Optimizer``.
        device_memory_fraction: Fraction of the *currently free* device memory used
            to size each streamed-in batch, recomputed every step. Must fit at
            least the largest single parameter's states.
        pin_memory: Pin the host-side state buffers. Improves transfer bandwidth
            at the cost of page-locked host memory; a failed pinned allocation
            falls back to pageable memory with a rank-0 warning.
        chunk_bytes: State bytes updated (and swapped out) per pipeline chunk.
            A batch is swapped in up front; the chunk boundary then lets one
            chunk's swap-out overlap the next chunk's compute. A byte budget
            keeps a chunk from being one huge parameter or hundreds of tiny ones.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        device_memory_fraction: float = 0.8,
        pin_memory: bool = True,
        chunk_bytes: int = 128 << 20,
    ) -> None:
        # YAML resolves an exponent literal without a decimal point (``8e-1``) to
        # a string; coerce before the range check.
        device_memory_fraction = float(device_memory_fraction)
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
        if not 0.0 < device_memory_fraction <= 1.0:
            raise ValueError(f"device_memory_fraction must be in (0, 1], got {device_memory_fraction}")
        if chunk_bytes < 1:
            raise ValueError(f"chunk_bytes must be >= 1, got {chunk_bytes}")

        # The param-group key set must match stock ``torch.optim.AdamW`` exactly:
        # DCP checkpoints one entry per group key, and a strict load requires the
        # checkpoint to carry every key of the target optimizer, so a swap
        # checkpoint must load into stock AdamW (and vice versa).
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            decoupled_weight_decay=True,
        )

        self.device_memory_fraction = device_memory_fraction
        # Resolve the backend before the pinning gate: pinning is only
        # meaningful when an accelerator backend exists.
        self._device_type = get_device_type()
        # Pinned host buffers need an accelerator backend (``async_offload`` gates
        # pinning the same way); a host with no accelerator cannot serve it.
        self.pin_memory = bool(pin_memory) and self._device_type != "cpu"
        self.chunk_bytes = int(chunk_bytes)

        # Two copy streams so H2D and D2H overlap instead of serializing on one stream.
        # Only the backends whose stream/event API has been exercised take this path.
        self._use_streams = self._device_type in STREAM_DEVICE_TYPES
        if self._use_streams:
            self._in_stream: Any | None = create_stream()
            self._out_stream: Any | None = create_stream()
        else:
            self._in_stream = None
            self._out_stream = None

        self._state_keys: tuple[str, ...] = ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ())
        # ``hasattr`` is always True on modern torch (the op schema is registered
        # on every build), so gate on the backend explicitly: torch ships CUDA and
        # CPU kernels and Ascend NPU registers its own; other backends (e.g. MLU)
        # take the elementwise fallback.
        self._use_fused = self._device_type in FUSED_ADAMW_DEVICE_TYPES and hasattr(torch, "_fused_adamw_")
        self._host_states: dict[Tensor, dict[str, Tensor]] = {}
        self._device_states: dict[Tensor, dict[str, Tensor]] = {}
        self._swap_in_events: dict[Tensor, Any] = {}
        # Storages with an in-flight D2H on ``_out_stream``; freed only after the stream drains.
        self._pending_free: list[Tensor] = []

        # ``Optimizer.__init__`` routes every group through ``add_param_group``, which
        # builds the state, so the attributes above must exist before it runs.
        super().__init__(params, defaults)
        self._init_states()

    def add_param_group(self, param_group: dict) -> None:  # type: ignore[override]
        # ``_state_keys`` (and therefore ``_param_bytes``) is fixed at
        # construction; a group with a different ``amsgrad`` would need a third
        # moment buffer and would make the swap budget under-count.
        if isinstance(param_group, dict):
            # An explicit ``None`` survives the base class's ``setdefault`` and is
            # falsy, so normalize instead of skipping it.
            requested = bool(param_group.get("amsgrad", self.defaults["amsgrad"]))
            if requested != bool(self.defaults["amsgrad"]):
                raise ValueError(
                    "SwapAdamW cannot add a param group with amsgrad="
                    f"{requested} (built with amsgrad={self.defaults['amsgrad']}): "
                    "the moment-state layout is fixed at construction."
                )
        super().add_param_group(param_group)
        self._init_states()

    # ------------------------------------------------------------------ build

    def _alloc_host_buffer(self, local_shard: Tensor) -> Tensor:
        """Zero-initialized host buffer for one state shard.

        Page-locked host memory is a finite machine resource: on exhaustion the
        optimizer degrades to pageable buffers once — the same fallback
        ``async_offload`` uses — instead of failing the whole run at build time.
        Backends that cannot pin at all report that differently (``RuntimeError``
        for a failed allocation, ``NotImplementedError``/``TypeError`` for an
        unsupported call), so all three fall back here.
        """
        shape = tuple(local_shard.shape)
        try:
            return torch.zeros(shape, dtype=local_shard.dtype, device="cpu", pin_memory=self.pin_memory)
        except (RuntimeError, NotImplementedError, TypeError):
            if not self.pin_memory:
                raise
            logger.warning_rank0(
                "Pinned host allocation for SwapAdamW states failed (page-locked memory "
                "exhausted?); continuing with pageable host buffers."
            )
            self.pin_memory = False
            return torch.zeros(shape, dtype=local_shard.dtype, device="cpu", pin_memory=False)

    def _init_states(self) -> None:
        """Migrate every trainable parameter's moment state to host buffers."""
        # Precompute per-step invariants first: ``_ensure_param_state`` reads the
        # param-to-group map, and the per-param state sizes never change during
        # training (they are derived from the parameter, not from its state).
        self._group_by_param: dict[Tensor, dict[str, Any]] = {}
        self._param_bytes: dict[Tensor, int] = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._group_by_param[p] = group
                local_shard = _to_local(p)
                self._param_bytes[p] = sum(local_shard.numel() * local_shard.element_size() for _ in self._state_keys)

        # Frozen parameters get their state on the first step that sees a
        # gradient, as stock AdamW does (see ``_ensure_param_state``).
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._ensure_param_state(p, bool(group["amsgrad"]))

    def _ensure_param_state(self, p: Tensor, amsgrad: bool) -> None:
        """Allocate ``p``'s host/device moment state if it has none yet.

        Stock AdamW creates state when a parameter first receives a gradient, so a
        parameter that is frozen at construction (and later unfrozen) must get its
        buffers on that first step, not at construction. Allocation happens once
        per parameter; when it fires from ``step()`` it briefly materializes one
        local shard on the device (``empty_like``) and frees that storage again
        immediately.
        """
        if p in self._host_states:
            return

        host_state: dict[str, Tensor] = {}
        device_state: dict[str, Tensor] = {}
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key == "max_exp_avg_sq" and not amsgrad:
                continue
            # Only shape/dtype/placements are taken from this DTensor; the
            # zero-filled storage is freed immediately below, so skip the fill.
            dtensor = torch.empty_like(p)
            self.state[p][key] = dtensor
            local_shard = _to_local(dtensor)
            host_state[key] = self._alloc_host_buffer(local_shard)
            device_state[key] = local_shard
            local_shard.untyped_storage().resize_(0)

        if "step" not in self.state[p]:
            self.state[p]["step"] = 0
        self._host_states[p] = host_state
        self._device_states[p] = device_state

    # -------------------------------------------------------------- checkpoint

    def _bind_host_states(self) -> None:
        """Rebind every state holder to its live host buffer for DCP to read.

        The DTensor shells keep their mesh/placements; their local becomes the
        host buffer, so DCP never materializes state on the accelerator.
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
        """Optimizer state with all values read from host memory (see ``_bind_host_states``).

        ``step`` is stored internally as a Python int but exported as a CPU
        tensor so DCP checkpoints stay compatible with stock AdamW (which also
        stores step as a tensor).

        The host buffers are handed to DCP live (no copy here), but DCP's staging
        stager deep-copies them before writing, so a save transiently needs about
        twice the host optimizer-state memory (on top of the host buffers).
        """
        if getattr(self, "_veomni_dcp_restore_failed", False):
            raise RuntimeError("SwapAdamW cannot save after a failed checkpoint restore; rebuild the optimizer")
        staged = getattr(self, "_veomni_dcp_staged_state", None)
        if staged is None:
            self._bind_host_states()
            state_dict = super().state_dict()
        else:
            # DCP loads into independent CPU buffers until its provenance and
            # inventory checks have completed. Never expose the live host
            # buffers as the load template. User hooks still observe the live
            # optimizer, as they do outside DCP; only the serialized values
            # come from the staged mapping.
            live = getattr(self, "_veomni_dcp_live_state", self.state)
            current = self.state
            self.state = live
            try:
                for hook in self._optimizer_state_dict_pre_hooks.values():
                    hook(self)
            finally:
                self.state = current
            pre_hooks = self._optimizer_state_dict_pre_hooks
            post_hooks = self._optimizer_state_dict_post_hooks
            self._optimizer_state_dict_pre_hooks = type(pre_hooks)()
            self._optimizer_state_dict_post_hooks = type(post_hooks)()
            self.state = staged
            try:
                state_dict = super().state_dict()
            finally:
                self.state = current
                self._optimizer_state_dict_pre_hooks = pre_hooks
                self._optimizer_state_dict_post_hooks = post_hooks
            # A normal SwapAdamW state-dict post hook sees Python integer
            # steps. Keep that hook contract even though DCP's scratch uses
            # tensors as its eventual destination.
            state_dict["state"] = {
                key: ({**value, "step": int(value["step"].item())} if "step" in value else value)
                for key, value in state_dict["state"].items()
            }
            self.state = live
            try:
                for hook in post_hooks.values():
                    result = hook(self, state_dict)
                    if result is not None:
                        state_dict = result
            finally:
                self.state = current

        # ``Optimizer.state_dict`` hands back the live inner state dicts; copy
        # them before swapping the int step for the tensor stock AdamW expects.
        def serialize_step(step):
            if torch.is_tensor(step):
                if step.ndim != 0:
                    raise RuntimeError("SwapAdamW checkpoint step must be a scalar")
                step = step.item()
            if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
                raise RuntimeError("SwapAdamW checkpoint step must be a finite integer")
            if not 0 <= step <= 2**24 or int(step) != step:
                raise RuntimeError("SwapAdamW checkpoint step cannot be represented exactly as float32")
            return torch.tensor(float(step), dtype=torch.float32, device="cpu")

        state_dict["state"] = {
            key: ({**value, "step": serialize_step(value["step"])} if "step" in value else value)
            for key, value in state_dict["state"].items()
        }
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """Load values into the host buffers without materializing them on device.

        The base implementation casts every value to ``param.device`` (moving the
        host buffers onto the accelerator), so the mapping is done here instead.
        State is aligned to the current parameters by param-group position, as the
        base class does; param-group hyperparameters are merged from the checkpoint,
        except ``amsgrad``, which is structural and must match the construction-time
        value.

        As in ``torch.optim.Optimizer``, the checkpoint *replaces* the state: a
        parameter it does not carry goes back to zero moments and step 0 instead
        of keeping this optimizer instance's history. That applies to a direct
        ``load_state_dict``/``set_optimizer_state_dict`` call. A DCP load whose
        template came from an already-stepped optimizer keeps the template's own
        values for keys the planner does not overwrite (partial loads), so resume
        into a fresh optimizer to start omitted parameters from zero.
        """
        state_dict = state_dict.copy()
        live_state = getattr(self, "_veomni_dcp_live_state", None)
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            if live_state is not None and not getattr(hook, "_veomni_internal_dcp_filter", False):
                staged_state = self.state
                self.state = live_state
                try:
                    result = hook(self, state_dict)
                finally:
                    self.state = staged_state
            else:
                result = hook(self, state_dict)
            if result is not None:
                state_dict = result

        if live_state is None:
            self._apply_state_dict(state_dict)
            for hook in self._optimizer_load_state_dict_post_hooks.values():
                hook(self)
        else:
            staged_state = self.state
            self.state = live_state
            try:
                self._apply_state_dict(state_dict)
                for hook in self._optimizer_load_state_dict_post_hooks.values():
                    hook(self)
            finally:
                self.state = staged_state

    def _apply_state_dict(self, state_dict) -> None:
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        id_map = {}
        group_updates = []
        for group, saved_group in zip(groups, saved_groups):
            if len(group["params"]) != len(saved_group["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match the optimizer's group"
                )
            saved_amsgrad = bool(saved_group.get("amsgrad", group["amsgrad"]))
            if saved_amsgrad != bool(group["amsgrad"]):
                raise ValueError(
                    "Cannot load a checkpoint whose param group has amsgrad="
                    f"{saved_amsgrad} into a SwapAdamW built with amsgrad={group['amsgrad']}: "
                    "the moment-state layout is fixed at construction."
                )
            id_map.update(zip(saved_group["params"], group["params"]))
            # ``amsgrad`` is structural (it decides whether a third moment buffer
            # exists); keep the constructor's value rather than the checkpoint's.
            group_updates.append({k: v for k, v in saved_group.items() if k not in ("params", "amsgrad")})

        loaded = set()
        pending = []
        staged = getattr(self, "_veomni_dcp_staged_state", None)
        for key, value in state_dict["state"].items():
            param = id_map.get(key)
            if param is None:
                if staged is not None:
                    raise RuntimeError("SwapAdamW checkpoint contains an unowned parameter state")
                continue
            # A parameter absent from the checkpoint still gets an entry from the
            # flattened loader, carrying only empty dicts (``{"exp_avg": {}, ...}``),
            # so judge presence by values that actually hold state -- not by the key
            # existing or the entry being truthy.
            if any(isinstance(v, (torch.Tensor, int, float)) for v in value.values()):
                loaded.add(param)
            pending.append((param, value))

        if staged is not None:
            if loaded != set(staged):
                raise RuntimeError("SwapAdamW loaded state does not match the proven checkpoint inventory")
            for param, value in pending:
                if param not in loaded:
                    continue
                if set(value) != {"step", *self._state_keys}:
                    raise RuntimeError("SwapAdamW checkpoint parameter state is incomplete")
                step = value["step"]
                if not torch.is_tensor(step) or step.ndim != 0 or step.dtype != torch.float32:
                    raise RuntimeError("SwapAdamW checkpoint step must be a float32 scalar")
                step_value = step.item()
                if not math.isfinite(step_value) or not 0 <= step_value <= 2**24 or not step_value.is_integer():
                    raise RuntimeError("SwapAdamW checkpoint step is not an exact non-negative integer")
                for field in self._state_keys:
                    moment = value[field]
                    if not torch.is_tensor(moment) or moment.shape != param.shape or moment.dtype != param.dtype:
                        raise RuntimeError(f"SwapAdamW checkpoint state {field} has an incompatible shape or dtype")

        for group, update in zip(groups, group_updates):
            group.update(update)
        for param, value in pending:
            self._load_param_state(param, value)
        # ``torch.optim.Optimizer.load_state_dict`` installs a fresh state mapping,
        # so an unseen parameter carries no history there; reset ours to match.
        for param, host_state in self._host_states.items():
            if param in loaded:
                continue
            for host_tensor in host_state.values():
                host_tensor.zero_()
            self.state[param]["step"] = 0
        if staged is not None:
            for param in set(self._host_states) - loaded:
                self._host_states.pop(param)
                self._device_states.pop(param)
                self.state.pop(param, None)

    def _load_param_state(self, p: Tensor, value: dict) -> None:
        # Reached from a direct ``load_state_dict`` call. The DCP resume path goes
        # through ``set_optimizer_state_dict`` instead, which both drops state for
        # ``requires_grad=False`` parameters and builds its template from the live
        # state dict, so moments for a frozen parameter are not restored that way.
        host_state = self._host_states.get(p)
        if host_state is None:
            # An entry whose values are all empty dicts marks a parameter absent
            # from the checkpoint (the flattened loader reconstructs it that way).
            if not any(isinstance(v, torch.Tensor) for v in value.values()):
                return
            # The checkpoint carries state for a parameter this optimizer was built
            # with frozen. Stock ``load_state_dict`` restores it regardless of
            # ``requires_grad``, so allocate the host buffers first.
            self._ensure_param_state(p, bool(self._group_by_param[p]["amsgrad"]))
            host_state = self._host_states[p]
        for key, loaded in value.items():
            if isinstance(loaded, DTensor):
                loaded = _to_local(loaded)
            if key == "step":
                if isinstance(loaded, torch.Tensor):
                    self.state[p]["step"] = int(loaded.item())
                elif isinstance(loaded, (int, float)):
                    self.state[p]["step"] = int(loaded)
                # An empty dict marks a param absent from the checkpoint (the
                # flattened loader reconstructs missing entries that way); keep
                # the construction-time step 0 in that case.
            elif key in host_state and isinstance(loaded, torch.Tensor):
                # Same for the moments: leave the zero-initialized host buffer.
                if loaded.shape != host_state[key].shape:
                    raise ValueError(
                        f"optimizer state {key!r} has shape {tuple(loaded.shape)} in the checkpoint but "
                        f"{tuple(host_state[key].shape)} here; DCP should reshard before this point."
                    )
                host_state[key].copy_(loaded.to(host_state[key].device))

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

    def _device_free_bytes(self) -> int:
        """Bytes this process can still allocate on the device.

        Driver-level free (``mem_get_info``) plus the allocator's cached-but-unused
        blocks (``reserved - allocated``).
        """
        device = get_torch_device()
        try:
            free, _total = device.mem_get_info()
            cached = int(device.memory_reserved()) - int(device.memory_allocated())
            return int(free) + max(cached, 0)
        except (AttributeError, RuntimeError, NotImplementedError):
            try:
                total = int(device.get_device_properties(get_device_id()).total_memory)
                return total - int(device.memory_allocated())
            except (AttributeError, RuntimeError, NotImplementedError):
                pass
        logger.warning_once("SwapAdamW could not determine free device memory; swapping states in one batch.")
        return _INFINITE_BUDGET  # pragma: no cover

    def _batch_budget_bytes(self) -> int:
        """Bytes of moment state swapped in per batch.

        Only the moment states are budgeted: gradients and the elementwise path's
        per-parameter temporaries (the negated grad for ``maximize``, ``denom``)
        still need headroom, so keep ``device_memory_fraction`` below 1.0.
        """
        if self._device_type == "cpu":
            return _INFINITE_BUDGET
        return max(int(self._device_free_bytes() * self.device_memory_fraction), 0)

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
                param_bytes = self._param_bytes[params[index]]
                if bytes_moved != 0 and bytes_moved + param_bytes > budget:
                    break
                # bytes_moved == 0 marks the batch's first parameter: it must
                # fit within the budget or the batch cannot start at all. Empty
                # shards (uneven EP/FSDP sharding) move zero bytes and leave
                # this check armed for the next real parameter.
                if bytes_moved == 0 and param_bytes > budget:
                    raise RuntimeError(
                        "SwapAdamW could not fit any optimizer state on device. "
                        "Increase model.optimizer.state_offload_device_memory_fraction or reduce the parameter size."
                    )
                self._swap_in_param(params[index])
                bytes_moved += param_bytes
                index += 1
        return index

    def _bump_step(self, p: Tensor) -> int:
        step = int(self.state[p]["step"]) + 1
        self.state[p]["step"] = step
        return step

    def _update_param(self, p: Tensor, group: dict[str, Any]) -> None:
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

    def _update_group_fused(self, params: Sequence[Tensor], group: dict[str, Any]) -> None:
        """Update a chunk with one fused call per local device and dtype.

        torch's own fused path groups params/grads/moments/step by
        ``(device, dtype)`` and issues one kernel call per group
        (``torch/optim/adam.py::_fused_adam``); a chunk here can span dtypes,
        so split it the same way.
        """
        if not params:
            return
        buckets: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        for p in params:
            local_p, local_g = _to_local(p), _to_local(p.grad)
            if local_g.device != local_p.device or local_g.dtype != local_p.dtype:
                raise RuntimeError(
                    "SwapAdamW fused updates require each gradient to match its "
                    f"parameter's device and dtype; got a {local_g.dtype} grad on "
                    f"{local_g.device} for a {local_p.dtype} param on {local_p.device}."
                )
            buckets.setdefault((local_p.device, local_p.dtype), []).append(p)
        for bucket in buckets.values():
            self._fused_adamw_bucket(bucket, group)

    def _fused_adamw_bucket(self, params: Sequence[Tensor], group: dict[str, Any]) -> None:
        """Run one ``torch._fused_adamw_`` call over a same-device, same-dtype bucket."""
        amsgrad = bool(group["amsgrad"])
        beta1, beta2 = group["betas"]
        device = _to_local(params[0]).device
        steps = []
        for p in params:
            step = self._bump_step(p)
            # float32 matches torch's own fused AdamW step contract
            # (``_get_scalar_dtype(is_fused=True)``); some kernels reject int64.
            steps.append(torch.tensor(step, dtype=torch.float32, device=device))
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

    def _update_chunk(self, params: Sequence[Tensor], group: dict[str, Any]) -> None:
        """Update one chunk of parameters that share ``group``."""
        if self._use_fused:
            self._update_group_fused(params, group)
        else:
            for p in params:
                self._update_param(p, group)

    # ------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure=None):
        if getattr(self, "_veomni_dcp_restore_failed", False):
            raise RuntimeError("SwapAdamW cannot train after a failed checkpoint restore; rebuild the optimizer")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active: list[Tensor] = []
        for param in self._group_by_param:
            if not param.requires_grad or param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("SwapAdamW does not support sparse gradients")
            # A parameter frozen at construction (or added that way) has no state
            # yet; create it here, as stock AdamW does on the first gradient.
            self._ensure_param_state(param, bool(self._group_by_param[param]["amsgrad"]))
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
                group = self._group_by_param[active[group_start]]
                group_end = group_start + 1
                while group_end < batch_end and self._group_by_param[active[group_end]] is group:
                    group_end += 1
                chunk_start = group_start
                while chunk_start < group_end:
                    chunk_end = chunk_start
                    acc_bytes = 0
                    while chunk_end < group_end:
                        param_bytes = self._param_bytes[active[chunk_end]]
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
