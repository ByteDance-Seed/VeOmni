# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates.
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
#
# DT-FSDP2: Disaggregated-Tensor Fully Sharded Data Parallel 2.
#
# A monkey-patch on PyTorch's native FSDP2 (``fully_shard``) that splits a
# transformer layer's `fully_shard` into per-sub-module (attn, mlp, …) calls.
# This avoids all-gathering the entire layer's weights at once, reducing peak
# memory fragmentation for large MoE models.
#
# Adapted from MindSpeed-MM's reference implementation:
#   mindspeed_mm/fsdp/ops/fully_shard/fully_shard.py
#
# Usage (called automatically by :func:`parallelize_model_fsdp2` when
# ``train.accelerator.fsdp_config.enable_dt_fsdp2=True``)::
#
#     from veomni.distributed.dt_fsdp2 import apply_dt_fsdp2_patch
#     apply_dt_fsdp2_patch()
#     # Now ``fully_shard(..., hook_module=...)`` accepts the extra kwarg.

import functools
import weakref
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed._composable_state import _insert_module_state
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.utils import _get_root_modules
from torch.utils._pytree import tree_map

# ---------------------------------------------------------------------------
# Internal PyTorch FSDP2 imports (private API; may change across versions)
# ---------------------------------------------------------------------------
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    HSDPMeshInfo,
    TrainingState,
    _cast_fp_tensor,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, alloc_storage
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext, FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import (
    FSDPState,
    _register_group_forward_hooks,
    disable_if_config_true,
    logger,
)
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule, _unimplemented_deepcopy

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Mapping from original module class to the dynamically created FSDP-wrapped class
cls_to_fsdp_cls: dict[type, type] = {}

# Tracks the number of communication contexts assigned to each hook module.
# Key: hook_module, Value: count of contexts used (used to generate next index)
HOOK_MODULE_COMM_CTX_COUNT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Extended NamedTuples (add hook_module field)
# ---------------------------------------------------------------------------

class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event  # all-gather copy-out
    hook_module: nn.Module


class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event  # reduce-scatter event
    hook_module: nn.Module


class AllReduceState(NamedTuple):
    all_reduce_input: torch.Tensor
    event: torch.Event  # all-reduce event


# ---------------------------------------------------------------------------
# Core API: fully_shard (patched)
# ---------------------------------------------------------------------------

@contract(state_cls=FSDPState)
def fully_shard(
    module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
    ignored_params: Optional[set[nn.Parameter]] = None,
    hook_module: Optional[nn.Module] = None,
):
    """Applies FSDP2 to *module*, optionally registering hooks on *hook_module*.

    When *hook_module* is provided, the forward pre/post hooks are registered
    on *hook_module* instead of on *module* itself.  This allows multiple FSDP
    units (e.g. ``self_attn`` and ``mlp``) to share a single hook-site
    (typically the parent transformer block), aligning with activation
    checkpointing boundaries.

    Each unique *hook_module* receives an auto-incremented ``comm_ctx_index``
    so that sub-modules within the same block get independent communication-
    context slots, preventing buffer overwrites.

    Args:
        module: The module to shard.
        mesh: DeviceMesh.  ``None`` creates a default 1D mesh.
        reshard_after_forward: Whether to reshard after forward.
        shard_placement_fn: Custom shard-dim placement callback.
        mp_policy: Mixed-precision policy.
        offload_policy: CPU offload policy.
        ignored_params: Parameters to exclude from sharding.
        hook_module: Module on which to register the FSDP hooks.
            ``None`` (default) falls back to the standard behaviour
            (hooks on *module* itself).
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined]

    # Determine hook_module
    if hook_module is not None:
        _hook_module = hook_module
    else:
        _hook_module = modules[0] if len(modules) > 0 else modules

    # Auto-increment comm_ctx_index
    if _hook_module not in HOOK_MODULE_COMM_CTX_COUNT:
        HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = 0
    comm_ctx_index = HOOK_MODULE_COMM_CTX_COUNT[_hook_module]
    HOOK_MODULE_COMM_CTX_COUNT[_hook_module] = comm_ctx_index + 1

    # Initialize state with custom parameters
    state.init(
        modules,
        device,
        mp_policy,
        hook_module=hook_module,
        comm_ctx_index=comm_ctx_index,
    )

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for mod in modules:
        cls = mod.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        mod.__class__ = new_cls
    return arg_module


# ---------------------------------------------------------------------------
# Patched FSDPState methods
# ---------------------------------------------------------------------------

def hook_module_init(
    self,
    modules: Tuple[nn.Module, ...],
    device: torch.device,
    mp_policy: MixedPrecisionPolicy,
    hook_module: Optional[nn.Module] = None,
    comm_ctx_index: int = 0,
) -> None:
    """Custom ``FSDPState.init`` that supports *hook_module* and *comm_ctx_index*.

    When *hook_module* is given the forward hooks are registered on that
    module instead of on the first managed module.
    """
    for mod in modules:
        _insert_module_state(mod, self)
    self._modules = modules
    self._device = device
    self._device_handle = _get_device_handle(device.type)
    self._mp_policy = mp_policy
    self.comm_ctx_index = comm_ctx_index

    # Register hooks
    if hook_module is not None:
        self._pre_forward_hook_handle = hook_module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = hook_module.register_forward_hook(
            self._post_forward, prepend=False
        )
        self.hook_module = weakref.ref(hook_module)
    else:
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle
        self.hook_module = weakref.ref(modules[0])


def _copy_fsdp_comm_ctx(new_comm_ctx: FSDPCommContext, comm_ctx: FSDPCommContext) -> FSDPCommContext:
    """Copy critical stream and state references from *comm_ctx* to *new_comm_ctx*.

    Streams are **shared** (same objects), while state slots
    (``all_gather_state``, ``reduce_scatter_state``, ``post_forward_order``)
    are independent per context.
    """
    new_comm_ctx.device_handle = comm_ctx.device_handle

    # Share streams
    new_comm_ctx.all_gather_copy_in_stream = comm_ctx.all_gather_copy_in_stream
    new_comm_ctx.all_gather_stream = comm_ctx.all_gather_stream
    new_comm_ctx.reduce_scatter_stream = comm_ctx.reduce_scatter_stream
    new_comm_ctx.all_reduce_stream = comm_ctx.all_reduce_stream

    # Copy initial state (these will diverge after first use)
    new_comm_ctx.all_gather_state = comm_ctx.all_gather_state
    new_comm_ctx.reduce_scatter_state = comm_ctx.reduce_scatter_state
    new_comm_ctx.post_forward_order = comm_ctx.post_forward_order

    return new_comm_ctx


def hook_module_init_shared_state(self) -> None:
    """Custom ``FSDPState._init_shared_state`` that creates a ``global_comm_ctx`` list.

    Every unique ``comm_ctx_index`` used by any state in the tree gets its
    own ``FSDPCommContext`` entry.  States that share the same index reuse
    the same context (and therefore the same state slots), while states with
    different indices get independent slots (but share physical CUDA streams).
    """
    self._comm_ctx.lazy_init(self._device)
    if not hasattr(self, "global_comm_ctx"):
        self.global_comm_ctx = [self._comm_ctx]

    # Collect all unique comm_ctx_indices
    global_comm_ctx_list = [0]
    for state in self._state_ctx.all_states:
        if state.comm_ctx_index not in global_comm_ctx_list:
            global_comm_ctx_list.append(state.comm_ctx_index)
            new_comm_ctx = FSDPCommContext()
            new_comm_ctx = _copy_fsdp_comm_ctx(new_comm_ctx, self._comm_ctx)
            self.global_comm_ctx.append(new_comm_ctx)

    # Assign the correct comm_ctx to each state based on its index
    for state in self._state_ctx.all_states:
        state._state_ctx = self._state_ctx
        _comm_ctx = self.global_comm_ctx[global_comm_ctx_list.index(state.comm_ctx_index)]
        setattr(state, "global_comm_ctx", self.global_comm_ctx)
        state._comm_ctx = _comm_ctx
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.comm_ctx = _comm_ctx
            setattr(fsdp_param_group, "hook_module", state.hook_module)
            setattr(fsdp_param_group, "global_comm_ctx", self.global_comm_ctx)


def _root_post_backward_final_callback(self) -> None:
    """Custom root post-backward callback that synchronises ALL global_comm_ctx entries."""
    if not compiled_autograd_enabled():
        logger.debug("FSDP::root_post_backward")
    with torch.profiler.record_function("FSDP::root_post_backward_callback"):
        for state in self._state_ctx.all_states:
            fsdp_param_group = state._fsdp_param_group
            if (
                fsdp_param_group
                and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
            ):
                fsdp_param_group.post_backward()
            state._training_state = TrainingState.IDLE
            if fsdp_param_group:
                fsdp_param_group._training_state = TrainingState.IDLE
            if self._state_ctx.is_last_backward:
                state._finalize_backward()
        if self._state_ctx.is_last_backward:
            self._comm_ctx.post_forward_order.clear()
            if self._comm_ctx.reduce_scatter_state is not None:
                self._device_handle.current_stream().wait_event(
                    self._comm_ctx.reduce_scatter_state.event
                )
                self._comm_ctx.reduce_scatter_state = None

            # Wait for ALL global comm contexts
            if hasattr(self, "global_comm_ctx"):
                for _comm_ctx in self.global_comm_ctx:
                    _comm_ctx.post_forward_order.clear()
                    if _comm_ctx.reduce_scatter_state is not None:
                        self._device_handle.current_stream().wait_event(
                            _comm_ctx.reduce_scatter_state.event
                        )
                    _comm_ctx.reduce_scatter_state = None

        self._state_ctx.post_backward_final_callback_queued = False


@disable_if_config_true
def _pre_forward(
    self, module: nn.Module, args: Tuple[Any, ...], kwargs: dict[str, Any]
) -> Tuple[Tuple[Any, ...], dict[str, Any]]:
    """Custom pre-forward that waits for all global_comm_ctx AG events before prefetching.

    Optimization 2: Refined event dependencies — wait for ALL previous
    all-gather events (across all global_comm_ctx slots) before triggering
    a new prefetch, preventing excessive prefetch and bandwidth contention.
    """
    if self._training_state == TrainingState.PRE_BACKWARD:
        return args, kwargs
    self._training_state = TrainingState.FORWARD
    args, kwargs = self._root_pre_forward(module, args, kwargs)
    if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
        with torch.profiler.record_function("FSDP::cast_forward_inputs"):
            cast_fn = functools.partial(
                _cast_fp_tensor, self._mp_policy.param_dtype
            )
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
    if self._fsdp_param_group:
        args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
    for fsdp_state in self._states_to_forward_prefetch:
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            prefetch_all_gather_copy_in_stream = target_param_group.comm_ctx.all_gather_copy_in_stream
            # Wait for ALL previous global AG events before this prefetch
            for comm_ctx in self.global_comm_ctx:
                if comm_ctx.all_gather_state and comm_ctx.all_gather_state.event:
                    prefetch_all_gather_copy_in_stream.wait_event(
                        comm_ctx.all_gather_state.event
                    )
            FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
    return args, kwargs


@disable_if_config_true
def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
    """Custom post-forward that frees all-gather states for ALL global_comm_ctx entries."""
    if self._training_state == TrainingState.PRE_BACKWARD:
        return output
    if self._fsdp_param_group:
        output = self._fsdp_param_group.post_forward(module, input, output)
    output = self._register_pre_backward_hook(output)
    self._training_state = TrainingState.IDLE
    if self._state_ctx.iter_forward_root is self:
        for comm_ctx in self.global_comm_ctx:
            if comm_ctx.all_gather_state:
                self._comm_ctx.all_gather_copy_in_stream.wait_event(
                    comm_ctx.all_gather_state.event
                )
                self._comm_ctx.all_gather_stream.wait_event(comm_ctx.all_gather_state.event)
            comm_ctx.all_gather_state = None  # free the all-gather result
        self._state_ctx.iter_forward_root = None
    if self._mp_policy.output_dtype is not None:
        with torch.profiler.record_function("FSDP::cast_forward_outputs"):
            output = tree_map(
                functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                output,
            )
    return output


# ---------------------------------------------------------------------------
# Patched FSDPParamGroup methods — PyTorch 2.7 variant
# ---------------------------------------------------------------------------

def _pt27_wait_for_unshard(self):
    """Wait for preceding AG operations (PyTorch 2.7 API).

    Skips waiting when the previous all-gather belongs to the SAME
    hook_module (same-layer sub-modules don't block each other).
    """
    if not self._all_gather_result:
        return
    async_op = self._all_gather_result.all_gather_work is not None
    if self._training_state == TrainingState.FORWARD:  # implicit prefetch
        for comm_ctx in self.global_comm_ctx:
            if prev_all_gather_state := comm_ctx.all_gather_state:
                if prev_all_gather_state.hook_module != self.hook_module:
                    self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                    comm_ctx.all_gather_state = None  # free

    with torch.profiler.record_function(self._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out(
            self._all_gather_result,
            self.fsdp_params,
            self._all_gather_process_group,
        )
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()
    all_gather_copy_out_event = self.device_handle.Event()
    all_gather_copy_out_event.record()
    if not async_op and self._training_state == TrainingState.FORWARD:
        self.comm_ctx.all_gather_state = AllGatherState(
            self._all_gather_result, all_gather_copy_out_event, self.hook_module
        )
    else:
        self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    self._all_gather_result = None  # free unless saved in `all_gather_state`


def _pt27_post_backward(self, *unused: Any):
    """Custom post-backward for PyTorch 2.7.

    Waits for reduce-scatter events from OTHER hook modules before reducing.
    """
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()

    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()

    if len(fsdp_params_with_grad) == 0:
        return

    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_reduce")):
        # Wait for local context reduce-scatter
        if (
            self.comm_ctx.reduce_scatter_state is not None
            and self.comm_ctx.reduce_scatter_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
            self.comm_ctx.reduce_scatter_state = None

        # Wait for GLOBAL context reduce-scatters from DIFFERENT hook modules
        for comm_ctx in self.global_comm_ctx:
            if (
                comm_ctx.reduce_scatter_state
                and comm_ctx.reduce_scatter_state.event is not None
                and comm_ctx.reduce_scatter_state.hook_module != self.hook_module
            ):
                self.device_handle.current_stream().wait_event(
                    comm_ctx.reduce_scatter_state.event
                )
                comm_ctx.reduce_scatter_state = None

        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            if self._all_reduce_hook is None:
                raise AssertionError(
                    "all reduce hook stream is specified but hook itself is missing."
                )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.reduce_scatter_reduce_op,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
        )

        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event, self.hook_module
        )
        if all_reduce_input is not None:
            if all_reduce_event is None:
                raise AssertionError("all_reduce_event cannot be None.")
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )


# ---------------------------------------------------------------------------
# Patched FSDPParamGroup methods — PyTorch ≥ 2.9 variant
# ---------------------------------------------------------------------------

def _pt29_wait_for_unshard(self):
    """Wait for preceding AG operations (PyTorch ≥ 2.9 API).

    Adds a world_size==1 fast-path and uses the updated ``foreach_reduce``
    signature (``_reduce_scatter_comm``, ``gradient_divide_factor``,
    ``force_sum_reduction_for_comms``).
    """
    if not self._all_gather_result:
        return
    async_op = self._all_gather_result.all_gather_work is not None
    if self._training_state == TrainingState.FORWARD:  # implicit prefetch
        for comm_ctx in self.global_comm_ctx:
            if prev_all_gather_state := comm_ctx.all_gather_state:
                if prev_all_gather_state.hook_module != self.hook_module:
                    self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                    comm_ctx.all_gather_state = None  # free
    world_size = self._all_gather_process_group.size()
    if world_size == 1:
        for fsdp_param in self.fsdp_params:
            all_gather_input = fsdp_param.all_gather_inputs[0]
            fsdp_param.init_all_gather_outputs(
                [all_gather_input.numel()],
                [all_gather_input.dtype],
                world_size,
                self.device,
                force_recreate=False,
            )
            tensor = fsdp_param.all_gather_outputs[0]
            alloc_storage(tensor)
            with torch.autograd._unsafe_preserve_version_counter(tensor):
                tensor.copy_(all_gather_input)
    else:
        with torch.profiler.record_function(self._with_fqn("FSDP::all_gather_copy_out")):
            foreach_all_gather_copy_out(
                self._all_gather_result,
                self.fsdp_params,
                self._all_gather_process_group,
            )

    for fsdp_param in self.fsdp_params:
        fsdp_param.init_unsharded_param()
    self._to_unsharded()
    all_gather_copy_out_event = self.device_handle.Event()
    all_gather_copy_out_event.record()

    if not async_op and self._training_state == TrainingState.FORWARD and world_size > 1:
        self.comm_ctx.all_gather_state = AllGatherState(
            self._all_gather_result, all_gather_copy_out_event, self.hook_module
        )
    else:
        self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    self._all_gather_result = None  # free unless saved in `all_gather_state`


def _pt29_post_backward(self, *unused: Any):
    """Custom post-backward for PyTorch ≥ 2.9.

    Uses updated ``foreach_reduce`` signature with ``_reduce_scatter_comm``,
    ``gradient_divide_factor``, and ``force_sum_reduction_for_comms``.
    """
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()

    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward:
            self.reshard()

    if len(fsdp_params_with_grad) == 0:
        return

    with torch.profiler.record_function(self._with_fqn("FSDP::post_backward_reduce")):
        # Wait for local context reduce-scatter
        if (
            self.comm_ctx.reduce_scatter_state is not None
            and self.comm_ctx.reduce_scatter_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
            self.comm_ctx.reduce_scatter_state = None

        # Wait for GLOBAL context reduce-scatters from DIFFERENT hook modules
        for comm_ctx in self.global_comm_ctx:
            if (
                comm_ctx.reduce_scatter_state
                and comm_ctx.reduce_scatter_state.hook_module != self.hook_module
            ):
                self.device_handle.current_stream().wait_event(
                    comm_ctx.reduce_scatter_state.event
                )
                comm_ctx.reduce_scatter_state = None

        all_reduce_pg = self._all_reduce_process_group if self._is_hsdp else None
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            if self._all_reduce_hook is None:
                raise AssertionError(
                    "all reduce hook stream is specified but hook itself is missing."
                )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            self._reduce_scatter_process_group,
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.gradient_divide_factor,
            self._all_reduce_process_group if self._is_hsdp else None,
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
        )

        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event, self.hook_module
        )
        if all_reduce_input is not None:
            if all_reduce_event is None:
                raise AssertionError("all_reduce_event cannot be None.")
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
            )


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_dt_fsdp2_patch() -> None:
    """Apply all DT-FSDP2 monkey-patches.  Must be called BEFORE any ``fully_shard``.

    Patches:
    - ``FSDPState.init``, ``_init_shared_state``, ``_pre_forward``,
      ``_post_forward``, ``_root_post_backward_final_callback``
    - ``FSDPParamGroup.wait_for_unshard``, ``post_backward``
    - ``torch.distributed.fsdp.fully_shard``

    The PyTorch version variant is auto-detected at call time so the same
    code works on both NPU (torch 2.7.1) and GPU (torch 2.11.0).
    """
    # Patch FSDPState methods
    FSDPState.init = hook_module_init
    FSDPState._init_shared_state = hook_module_init_shared_state
    FSDPState._root_post_backward_final_callback = _root_post_backward_final_callback
    FSDPState._pre_forward = _pre_forward
    FSDPState._post_forward = _post_forward

    # Patch FSDPParamGroup methods — version-adaptive
    if hasattr(FSDPParamGroup, "_reduce_scatter_comm"):
        # PyTorch ≥ 2.9 (includes 2.11.0 on GPU)
        FSDPParamGroup.wait_for_unshard = _pt29_wait_for_unshard
        FSDPParamGroup.post_backward = _pt29_post_backward
    elif hasattr(FSDPParamGroup, "reduce_scatter_reduce_op"):
        # PyTorch 2.7 (NPU)
        FSDPParamGroup.wait_for_unshard = _pt27_wait_for_unshard
        FSDPParamGroup.post_backward = _pt27_post_backward
    else:
        raise RuntimeError(
            f"DT-FSDP2: Unsupported PyTorch {torch.__version__}. "
            f"FSDPParamGroup API does not match any known variant."
        )

    # Override the public fully_shard API in torch.distributed.fsdp (which
    # is the same object as torch.distributed._composable.fsdp.fully_shard
    # — PyTorch re-exports it).
    import torch.distributed.fsdp as fsdp

    fsdp.fully_shard = fully_shard

    # Also patch the _composable.fsdp module so that existing imports in
    # VeOmni (which use `from torch.distributed._composable.fsdp import
    # fully_shard`) pick up the patched version when re-imported.
    import torch.distributed._composable.fsdp as _composable_fsdp

    _composable_fsdp.fully_shard = fully_shard


# ---------------------------------------------------------------------------
# Sub-module discovery for automatic DT-FSDP2 sharding
# ---------------------------------------------------------------------------

#: Known sub-module attribute names on transformer decoder layers, in the
#: order they are executed during forward (attn-first, then mlp/ffn, then
#: norms/other).  Missing attributes are silently skipped so this one list
#: works across all model architectures.
DTFSDP2_SUB_MODULE_NAMES: Tuple[str, ...] = (
    "self_attn",                # Standard attention (Llama, Qwen, DeepSeek, GPT-OSS …)
    "linear_attn",              # Qwen3_5 GatedDeltaNet (per-layer choice)
    "mlp",                      # Dense MLP **or** SparseMoE block — all models use ``mlp``
    "attn_hc",                  # DeepSeekV4 HyperConnection (attention path)
    "ffn_hc",                   # DeepSeekV4 HyperConnection (FFN path)
    "input_layernorm",          # Pre-attention LayerNorm / RMSNorm
    "post_attention_layernorm",  # Pre-FFN LayerNorm / RMSNorm
)


def discover_dtfsdp2_submodules(layer_mod: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return ``(attr_name, sub_module)`` pairs that DT-FSDP2 should shard individually.

    Only children that actually exist on *layer_mod* and that are not already
    wrapped as ``FSDPModule`` (e.g. expert-parallel expert modules) are returned.
    The order follows :data:`DTFSDP2_SUB_MODULE_NAMES` which matches the typical
    forward execution order (attn → mlp → norms).
    """
    result: List[Tuple[str, nn.Module]] = []
    for attr_name in DTFSDP2_SUB_MODULE_NAMES:
        sub = getattr(layer_mod, attr_name, None)
        if sub is not None and isinstance(sub, nn.Module) and not isinstance(sub, FSDPModule):
            result.append((attr_name, sub))
    return result
