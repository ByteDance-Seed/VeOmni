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


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Literal, Optional, Tuple

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ..utils import logging
from ..utils.device import get_device_type


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

# The active value is execution-context-local. The source of truth is the state
# bound to the model being built or executed; there is no process-global default.
_CURRENT_PARALLEL_STATE: ContextVar[Optional["ParallelState"]] = ContextVar(
    "veomni_current_parallel_state", default=None
)

_MODEL_PARALLEL_STATE_ATTR = "_veomni_parallel_state"

# Process-group creation is collective and expensive. States with identical
# topologies share their meshes within one default-process-group lifetime.
_PARALLEL_STATE_CACHE: Dict[tuple, "ParallelState"] = {}
_PARALLEL_STATE_CACHE_PROCESS_GROUP: Any = None
_PARALLEL_STATE_CACHE_SESSION = object()


def _parallel_state_cache_session():
    """Return an identity token for the current default-process-group lifetime."""
    global _PARALLEL_STATE_CACHE_PROCESS_GROUP, _PARALLEL_STATE_CACHE_SESSION
    default_group = dist.group.WORLD if dist.is_initialized() else None
    if default_group is not _PARALLEL_STATE_CACHE_PROCESS_GROUP:
        _PARALLEL_STATE_CACHE_PROCESS_GROUP = default_group
        _PARALLEL_STATE_CACHE_SESSION = object()
    return _PARALLEL_STATE_CACHE_SESSION


def requires_mesh(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(self: "ParallelState", *args, **kwargs):
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized.")

        return fn(self, *args, **kwargs)

    return _inner


@dataclass(frozen=True)
class ParallelState:
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ulysses_size: int = 1
    dp_mode: Literal["ddp", "fsdp2"] = "fsdp2"
    device_type: str = get_device_type()
    include_sp_in_fsdp: bool = True
    device_mesh: Optional["DeviceMesh"] = None
    extra_parallel_names: Tuple[str] = ("ep",)
    extra_parallel_sizes: Dict[str, int] = field(default_factory=lambda: {"ep": 1})
    extra_parallel_fsdp_device_mesh: Dict[str, Optional["DeviceMesh"]] = field(default_factory=lambda: {"ep": None})
    async_enabled: Optional[bool] = False

    def __post_init__(self):
        if not self.include_sp_in_fsdp:
            raise NotImplementedError("Decoupled sequence parallel has not been implemented.")

        if self.cp_size > 1:
            raise NotImplementedError("Ring attention is not supported yet.")

        if self.pp_size * self.dp_size * self.cp_size * self.ulysses_size * self.tp_size != self.world_size:
            raise ValueError("The product of parallel sizes should be equal to the world size.")

        if self.dp_replicate_size * self.dp_shard_size != self.dp_size:
            raise ValueError(
                f"The product of dp_replicate_size: {self.dp_replicate_size} and dp_shard_size: {self.dp_shard_size} should be equal to dp_size: {self.dp_size}."
            )

        if self.sp_enabled and self.device_mesh is None:
            raise ValueError(
                "A sequence-parallel ParallelState must be built with a device mesh through build_parallel_state()."
            )

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        if self.is_initialized:
            return dist.get_rank()
        return -1

    @property
    def world_size(self) -> int:
        if self.is_initialized:
            return dist.get_world_size()
        return 1

    # ------------------------------ DP ------------------------------ #
    @property
    def dp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp")
        return None

    @property
    def dp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp")

        return self.fsdp_rank

    @property
    @requires_mesh
    def dp_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp"]

        raise self.fsdp_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    # ------------------------------ DP replicate ------------------------------ #
    @property
    def dp_replicate_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_replicate")

    @property
    def dp_replicate_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_replicate")

    @property
    @requires_mesh
    def dp_replicate_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_replicate"]

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate_size > 1

    # ------------------------------ DP shard ------------------------------ #
    @property
    def dp_shard_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard")

    @property
    def dp_shard_sp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard_sp")

    @property
    def dp_shard_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_shard")

    @property
    @requires_mesh
    def dp_shard_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_shard"]

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard_size >= 1

    # ----------------------------- FSDP ----------------------------- #
    @property
    def fsdp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_sp")

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_sp")

        return self.global_rank

    @property
    def dp_shard_sp_enabled(self) -> bool:
        return self.dp_shard_enabled and self.sp_enabled

    @property
    @requires_mesh
    def fsdp_mesh(self) -> "DeviceMesh":
        if self.dp_replicate_enabled:
            # HSDP
            if self.dp_shard_sp_enabled:
                return self.device_mesh["dp_replicate", "dp_shard_sp"]
            elif self.dp_shard_enabled:
                return self.device_mesh["dp_replicate", "dp_shard"]
            else:
                # DDP
                return self.device_mesh["dp_replicate"]
        # FSDP
        elif self.dp_shard_sp_enabled:
            return self.device_mesh["dp_shard_sp"]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_shard"]
        else:
            return self.device_mesh["dp"]

    @property
    def fsdp_enabled(self) -> bool:
        return self.fsdp_size > 1

    @property
    def fsdp_size(self) -> int:
        return self.world_size // (self.pp_size * self.tp_size)

    # ------------------------------ TP ------------------------------ #
    @property
    @requires_mesh
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp")

    @property
    @requires_mesh
    def tp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["tp"]

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    # ------------------------------ PP ------------------------------ #
    @property
    @requires_mesh
    def pp_rank(self) -> int:
        return self.device_mesh.get_local_rank("pp")

    @property
    @requires_mesh
    def pp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["pp"]

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    @requires_mesh
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    @requires_mesh
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)

    # ------------------------------ EP ------------------------------ #
    @property
    @requires_mesh
    def ep_mesh(self) -> "DeviceMesh":
        return self.extra_parallel_mesh("ep")

    @property
    @requires_mesh
    def ep_fsdp_mesh(self) -> "DeviceMesh":
        return self.extra_parallel_fsdp_mesh("ep")

    @cached_property
    def ep_group(self) -> "ProcessGroup":
        return self.extra_parallel_group("ep")

    @property
    def ep_enabled(self) -> bool:
        return self.extra_parallel_enabled("ep")

    @property
    def ep_size(self) -> int:
        return self.extra_parallel_sizes["ep"]

    @property
    def ep_rank(self) -> int:
        return self.extra_parallel_rank("ep")

    @property
    def ep_fsdp_size(self) -> int:
        return self.extra_parallel_fsdp_size("ep")

    @property
    def ep_gradient_divide_factor(self) -> int:
        return self.extra_parallel_gradient_divide_factor("ep")

    # ------------------------------ Parallel list ------------------------------ #
    @requires_mesh
    def extra_parallel_mesh(self, para_name) -> "DeviceMesh":
        return self.extra_parallel_fsdp_device_mesh[para_name][para_name]

    @requires_mesh
    def extra_parallel_fsdp_mesh(self, para_name) -> "DeviceMesh":
        return self.extra_parallel_fsdp_device_mesh[para_name][para_name, f"{para_name}_fsdp"]

    @requires_mesh
    def extra_parallel_group(self, para_name) -> "ProcessGroup":
        if self.extra_parallel_enabled(para_name):
            return self.extra_parallel_mesh(para_name).get_group()
        else:
            return None

    def extra_parallel_enabled(self, para_name) -> bool:
        return self.extra_parallel_sizes[para_name] > 1

    def extra_parallel_rank(self, para_name) -> int:
        return self.extra_parallel_fsdp_device_mesh[para_name].get_local_rank(para_name)

    def extra_parallel_fsdp_size(self, para_name) -> int:
        assert self.extra_parallel_enabled(para_name), (
            f"{para_name}_fsdp_size is only available when {para_name} is enabled ({para_name}_size > 1)"
        )
        return self.fsdp_size // self.extra_parallel_sizes[para_name]

    def extra_parallel_gradient_divide_factor(self, para_name) -> int:
        # We assume the world size is the total dp size by now
        # TP and PP would make this assumption not true
        assert self.tp_size == 1
        assert self.pp_size == 1
        # For ep+fsdp2, the grad divide factor should alwasy be world size (no matter HSDP or not)
        # SP does not affect this since SP groups still replicate params
        # and their grads are all-reduced which would match grads for the same data without SP.
        return self.world_size

    @property
    def any_extra_parallel_enabled(self) -> bool:
        return any(self.extra_parallel_enabled(para_name) for para_name in self.extra_parallel_names)

    # ------------------------------ SP ------------------------------ #
    @property
    def sp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.sp_enabled:
            return self.device_mesh.get_group("sp")
        return None

    @property
    def sp_rank(self) -> int:
        if self.device_mesh is not None and self.sp_enabled:
            return self.device_mesh.get_local_rank("sp")
        return -1

    @property
    def sp_enabled(self) -> bool:
        return self.cp_size > 1 or self.ulysses_size > 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_size * self.cp_size

    @property
    def ulysses_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.ulysses_enabled:
            return self.device_mesh.get_group("ulysses")
        return None

    @property
    def ulysses_rank(self) -> int:
        if self.device_mesh is not None and self.ulysses_enabled:
            return self.device_mesh.get_local_rank("ulysses")
        return -1

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None and self.cp_enabled:
            return self.device_mesh.get_group("cp")
        return None

    @property
    def cp_rank(self) -> int:
        if self.device_mesh is not None and self.cp_enabled:
            return self.device_mesh.get_local_rank("cp")
        return -1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1


def build_parallel_state(
    dp_size: int = 1,
    dp_replicate_size: int = 1,
    dp_shard_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ulysses_size: int = 1,
    dp_mode: Literal["ddp", "fsdp2"] = "fsdp2",
    device_type: str = None,
    include_sp_in_fsdp: bool = True,
    extra_parallel_sizes: Tuple[int] = (1,),
    extra_parallel_placement_innermost: Tuple[bool] = (False,),
    extra_parallel_names: Tuple[str] = ("ep",),
    async_enabled: Optional[bool] = False,
) -> "ParallelState":
    """
    Build or reuse a parallel state without changing the default state.

    All ranks must request topologies in the same order because creating device
    meshes is collective. Cache entries are scoped to the current default
    process group so distributed teardown cannot leave reusable stale groups.
    """
    if device_type is None:
        device_type = get_device_type()

    # Set dp_shard_size to dp_size if dp_shard_size and dp_replicate_size are not set when dp enabled
    if dp_size > 1 and dp_shard_size == 1 and dp_replicate_size == 1:
        dp_shard_size = dp_size

    extra_parallel_sizes = tuple(extra_parallel_sizes)
    extra_parallel_placement_innermost = tuple(extra_parallel_placement_innermost)
    extra_parallel_names = tuple(extra_parallel_names)

    # Note that Expert Parallel is included into Extra Parallel
    assert len(extra_parallel_sizes) == len(extra_parallel_placement_innermost) == len(extra_parallel_names), (
        "each extra parallel should correspond to a size, a placement and a name"
    )

    cache_key = (
        _parallel_state_cache_session(),
        dp_size,
        dp_replicate_size,
        dp_shard_size,
        tp_size,
        pp_size,
        cp_size,
        ulysses_size,
        dp_mode,
        device_type,
        include_sp_in_fsdp,
        extra_parallel_sizes,
        extra_parallel_placement_innermost,
        extra_parallel_names,
        async_enabled,
    )
    if cached_state := _PARALLEL_STATE_CACHE.get(cache_key):
        logger.info_rank0("Reusing parallel state for an identical topology.")
        return cached_state

    logger.info_rank0(
        f"Initializing parallel state: dp_size {dp_size}, dp_replicate_size {dp_replicate_size}, "
        + f"dp_shard_size {dp_shard_size},tp_size {tp_size}, pp_size {pp_size}, cp_size {cp_size}, ulysses_size {ulysses_size}, "
        + ", ".join(
            [
                f"{para_name}_size {para_size}"
                for para_name, para_size in zip(extra_parallel_names, extra_parallel_sizes)
            ]
        )
    )

    device_mesh = None

    extra_parallel_fsdp_device_mesh = {f"{para_name}": None for para_name in extra_parallel_names}

    mesh_shape = []
    mesh_dim_names = []
    for d, name in zip(
        [pp_size, dp_replicate_size, dp_shard_size, ulysses_size, cp_size, tp_size],
        ["pp", "dp_replicate", "dp_shard", "ulysses", "cp", "tp"],
    ):
        if d > 1 or name in ["dp_shard"]:
            mesh_shape.append(d)
            mesh_dim_names.append(name)

    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )

    # Mesh for data loading (no communication on this mesh)
    dp_mesh_dim_names = []
    # Mesh for param sharding
    dp_shard_sp_mesh_dim_names = []
    # Mesh for loss all-reduce
    dp_sp_mesh_dim_names = []
    # Mesh for sequence parallel
    sp_mesh_dim_names = []

    if dp_replicate_size > 1:
        dp_mesh_dim_names.append("dp_replicate")
        dp_sp_mesh_dim_names.append("dp_replicate")
    if dp_shard_size >= 1:
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_sp_mesh_dim_names.append("dp_shard")
        dp_sp_mesh_dim_names.append("dp_shard")
    if ulysses_size > 1:
        dp_shard_sp_mesh_dim_names.append("ulysses")
        sp_mesh_dim_names.append("ulysses")
        dp_sp_mesh_dim_names.append("ulysses")
    if cp_size > 1:
        dp_shard_sp_mesh_dim_names.append("cp")
        sp_mesh_dim_names.append("cp")
        dp_sp_mesh_dim_names.append("cp")

    if dp_mesh_dim_names != []:
        device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

    if dp_shard_sp_mesh_dim_names != []:
        device_mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_sp")

    if dp_sp_mesh_dim_names != []:
        device_mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

    if sp_mesh_dim_names != []:
        device_mesh[tuple(sp_mesh_dim_names)]._flatten(mesh_dim_name="sp")

    for para_size, para_outside, para_name in zip(
        extra_parallel_sizes, extra_parallel_placement_innermost, extra_parallel_names
    ):
        if para_size > 1:
            # TODO: drop para_outside?
            assert not para_outside, f"{para_name} is not supported when para_outside is True."

            # NOTE: Support HSDP for extra parallel. For example, world_size=1024
            # - dense param device_mesh: (dp_replicate, dp_shard_sp)=(4, 256)
            # - ep_size=8, expert parallel device_mesh: (ep_replicate, ep_fsdp, ep)=(4, 32, 8)
            # Note that ep_size should be a factor of dp_shard_sp_size.
            param_mesh_shape, para_mesh_dim_names = [], []
            if dp_replicate_size > 1:
                param_mesh_shape.append(dp_replicate_size)
                para_mesh_dim_names.append(f"{para_name}_replicate")
            dp_shard_sp_size = device_mesh["dp_shard_sp"].size()
            assert dp_shard_sp_size % para_size == 0, (
                f"{para_name}_size({para_size}) must be a factor of dp_shard_sp_size({dp_shard_sp_size})"
            )
            para_fsdp_size = dp_shard_sp_size // para_size
            param_mesh_shape.append(para_fsdp_size)
            param_mesh_shape.append(para_size)
            para_mesh_dim_names.append(f"{para_name}_fsdp")
            para_mesh_dim_names.append(para_name)

            extra_parallel_fsdp_device_mesh[f"{para_name}"] = init_device_mesh(
                device_type=device_type,
                mesh_shape=param_mesh_shape,
                mesh_dim_names=para_mesh_dim_names,
            )

    logger.info_rank0(f"Device mesh: {device_mesh}")
    for para_name in extra_parallel_names:
        logger.info_rank0(f"{para_name} FSDP device mesh: {extra_parallel_fsdp_device_mesh[para_name]}")

    parallel_state = ParallelState(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode=dp_mode,
        device_type=device_type,
        include_sp_in_fsdp=include_sp_in_fsdp,
        device_mesh=device_mesh,
        extra_parallel_names=extra_parallel_names,
        extra_parallel_sizes=dict(zip(extra_parallel_names, extra_parallel_sizes)),
        extra_parallel_fsdp_device_mesh=extra_parallel_fsdp_device_mesh,
        async_enabled=async_enabled,
    )

    _PARALLEL_STATE_CACHE[cache_key] = parallel_state
    return parallel_state


def clear_parallel_state_cache() -> None:
    """Clear cached meshes after the default distributed process group ends."""
    global _PARALLEL_STATE_CACHE_PROCESS_GROUP, _PARALLEL_STATE_CACHE_SESSION
    _PARALLEL_STATE_CACHE_PROCESS_GROUP = None
    _PARALLEL_STATE_CACHE_SESSION = object()
    _CURRENT_PARALLEL_STATE.set(None)
    _PARALLEL_STATE_CACHE.clear()


@contextmanager
def use_parallel_state(parallel_state: "ParallelState") -> Iterator["ParallelState"]:
    """Activate a state for explicit model construction or execution."""
    token = _CURRENT_PARALLEL_STATE.set(parallel_state)
    try:
        yield parallel_state
    finally:
        _CURRENT_PARALLEL_STATE.reset(token)


def bind_model_parallel_state(model: Any, parallel_state: "ParallelState") -> Any:
    """Bind state to a model and make every root forward activate that state."""
    setattr(model, _MODEL_PARALLEL_STATE_ATTR, parallel_state)
    original_forward = getattr(model, "forward", None)
    if callable(original_forward) and not getattr(model, "_veomni_parallel_forward_wrapped", False):

        @wraps(original_forward)
        def model_owned_forward(*args, **kwargs):
            with use_parallel_state(get_model_parallel_state(model)):
                return original_forward(*args, **kwargs)

        model.forward = model_owned_forward
        model._veomni_parallel_forward_wrapped = True
    return model


def get_model_parallel_state(model: Any) -> "ParallelState":
    """Return the state owned by ``model`` or raise when it is unbound."""
    parallel_state = getattr(model, _MODEL_PARALLEL_STATE_ATTR, None)
    if parallel_state is None:
        raise ValueError(
            "The model has no bound ParallelState. Pass parallel_state to build_parallelize_model() "
            "or call bind_model_parallel_state() first."
        )
    return parallel_state


def resolve_model_parallel_state(model: Any, parallel_state: Optional["ParallelState"] = None) -> "ParallelState":
    """Resolve an explicit state, then a model-owned or active build state."""
    if parallel_state is not None:
        return parallel_state
    if (model_state := getattr(model, _MODEL_PARALLEL_STATE_ATTR, None)) is not None:
        return model_state
    return get_parallel_state()


@contextmanager
def use_model_parallel_state(model: Any) -> Iterator["ParallelState"]:
    """Run modeling code under the state owned by ``model``."""
    with use_parallel_state(get_model_parallel_state(model)) as parallel_state:
        yield parallel_state


@torch.compiler.assume_constant_result
def get_parallel_state() -> "ParallelState":
    """Return the state active in this explicit build/model execution context."""
    if (parallel_state := _CURRENT_PARALLEL_STATE.get()) is not None:
        return parallel_state
    raise RuntimeError(
        "No ParallelState is active. Build one with build_parallel_state(), bind it to the model, "
        "and use use_parallel_state() for pre-model build code."
    )


@torch.compiler.assume_constant_result
def get_current_parallel_state() -> Optional["ParallelState"]:
    """Return the state active in this execution context, without fallback."""
    return _CURRENT_PARALLEL_STATE.get()
