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


import gc
import hashlib
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..distributed.parallel_state import get_parallel_state
from ..optim.optimizer import restore_optimizer_param_group_defaults
from ..utils import logging
from ..utils.device import empty_cache, synchronize
from ..utils.dist_utils import any_rank_failed, raise_if_any_rank_failed
from .checkpointer import CheckpointerBase
from .layout import (
    DCP_MARKER_FILENAME,
    OPTIMIZER_DIRNAME,
    WEIGHTS_DIRNAME,
    dcp_markers,
    model_dir,
    optimizer_dir,
    step_dir,
    weights_dir,
)
from .layout import (
    LR_SCHEDULER_FILENAME as _LR_SCHEDULER_FILENAME,
)


logger = logging.get_logger(__name__)

_LR_SCHEDULER_KEY = "lr_scheduler"


class _ModelStrictLoadPlanner(DefaultLoadPlanner):
    """Allow partial optimizer state while requiring a complete full-model DCP."""

    def __init__(self, strict_model: bool):
        super().__init__(allow_partial_load=True)
        self.strict_model = strict_model

    def create_local_plan(self):
        plan = super().create_local_plan()
        if not self.strict_model:
            return plan

        assert self.metadata is not None
        missing_model_keys = sorted(
            key
            for key, path in self.mappings.items()
            if path and path[0] == "model" and key not in self.metadata.state_dict_metadata
        )
        if missing_model_keys:
            preview = ", ".join(missing_model_keys[:10])
            suffix = " ..." if len(missing_model_keys) > 10 else ""
            raise RuntimeError(
                f"DCP is missing {len(missing_model_keys)} model key(s) required for a full-model resume: "
                f"{preview}{suffix}"
            )

        return plan


def _validate_extra_parallel_meshes(parallel_state) -> None:
    """Fail-fast precondition for ExtraParallel state dict preprocessing.

    At least one ExtraParallel mesh must be non-None, and at least one
    of those meshes must carry the ExtraParallel + FSDP composition:
    2D ``(ep_fsdp, ep)`` for plain FSDP or 3D
    ``(ep_replicate, ep_fsdp, ep)`` for HSDP.
    """
    extra_parallel_mesh = {
        para: parallel_state.extra_parallel_fsdp_device_mesh[para][para]
        if parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in parallel_state.extra_parallel_names
    }
    assert any(m is not None for m in extra_parallel_mesh.values()), (
        "At least one extra_parallel mesh should be not None"
    )
    assert any(
        parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
        and parallel_state.extra_parallel_fsdp_device_mesh[para].ndim in (2, 3)
        for para in parallel_state.extra_parallel_names
    ), (
        "At least one extra_parallel fsdp_device_mesh must carry the ExtraParallel+FSDP "
        "composition: 2D (ep_fsdp, ep) for plain FSDP or 3D (ep_replicate, ep_fsdp, ep) for HSDP"
    )


def _apply_extra_parallel_dim(
    state_dict: Dict[str, Any],
    extra_parallel_fqn2spec_info: Dict[str, Any],
    parallel_state,
    action: str,
    *,
    key_match: str,
) -> Dict[str, Any]:
    """Drop or restore the ExtraParallel dimension on each tensor in a state dict.

    Shared by ``ModelState`` and ``OptimizerState``.  The only meaningful
    difference between the two callers is how state-dict keys map to
    ExtraParallel FQNs:

    * ``"exact"`` (model): the state-dict key IS the FQN,
      e.g. ``"model.layers.0.mlp.experts.gate_proj"``.
    * ``"substring"`` (optimizer): the state-dict key contains the FQN
      with extra prefix/suffix, e.g.
      ``"state.model.layers.0.mlp.experts.gate_proj.exp_avg"``.

    Non-tensor values and 0-D tensors are skipped unconditionally — they
    appear only in optimizer state dicts (param-group hyperparams, scalar
    ``step`` tensors); model state dicts never contain them, so the guard
    is a safe no-op there.
    """
    assert action in ("drop", "restore"), f"action must be 'drop' or 'restore', got {action!r}"
    assert key_match in ("exact", "substring"), f"key_match must be 'exact' or 'substring', got {key_match!r}"
    assert extra_parallel_fqn2spec_info is not None, "fqn2spec_info must not be None"

    _validate_extra_parallel_meshes(parallel_state)

    extra_parallel_keys = list(extra_parallel_fqn2spec_info.keys()) if key_match == "substring" else None

    for name in sorted(state_dict.keys()):
        if key_match == "exact":
            if name not in extra_parallel_fqn2spec_info:
                continue
            spec_info = extra_parallel_fqn2spec_info[name]
        else:  # "substring"
            matches = [k for k in extra_parallel_keys if k in name]
            if not matches:
                continue
            assert len(matches) == 1, f"Ambiguous ExtraParallel spec match for state key '{name}': {matches}"
            spec_info = extra_parallel_fqn2spec_info[matches[0]]

        if not isinstance(spec_info.placement, Shard):
            continue

        # Persistent ExtraParallel parameters already use their complete
        # runtime placement (for example PLE ``[Shard(1), Shard(0)]`` on
        # ``(ple_fsdp, ple)``). They are not managed by FSDP2, so there is no
        # temporary ExtraParallel dimension to drop or restore around DCP.
        if getattr(spec_info, "persistent_fsdp_shard_dim", None) is not None:
            continue

        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 0:
            continue

        assert spec_info.para_fsdp_mesh is not None, f"ExtraParallel spec {name} must have an ExtraParallel FSDP mesh"

        # Drop the innermost ExtraParallel (e.g. ``ep``) dim and keep the FSDP
        # sub-mesh: 1D ``(ep_fsdp,)`` for plain FSDP, or 2D
        # ``(ep_replicate, ep_fsdp)`` for HSDP. Mirrors the mesh slicing used
        # when sharding the module in ``torch_parallelize``.
        fsdp_submesh = spec_info.para_fsdp_mesh[spec_info.para_fsdp_mesh.mesh_dim_names[:-1]]
        # The ExtraParallel (e.g. ``ep``) dim shards the tensor along
        # ``spec_info.placement.dim`` (dim 0 for experts), while FSDP shards it
        # along ``spec_info.fsdp_shard_dim`` (1 by default, 0 for Muon zero-comm).
        # These are two DIFFERENT tensor dims, so the placements must use each
        # one explicitly instead of a fixed ``[Shard(0), Shard(1)]``.
        ep_shard_dim = spec_info.placement.dim
        fsdp_shard_dim = spec_info.fsdp_shard_dim
        if action == "drop":
            tensor = drop_extra_parallel_dim(tensor, fsdp_submesh, fsdp_shard_dim)
        else:
            tensor = restore_extra_parallel_dim(
                tensor, spec_info.para_fsdp_mesh, fsdp_submesh, ep_shard_dim, fsdp_shard_dim
            )
        state_dict[name] = tensor

    return state_dict


class ModelState(Stateful):
    """A wrapper around a model to make it stateful.

    Args:
        model: model to wrap.
        trainable_only: when ``True`` the state_dict only contains parameters with
            ``requires_grad=True`` (uses ``StateDictOptions(ignore_frozen_params=True)``).
            This is the LoRA / PEFT path: frozen base weights are skipped on save and
            ``set_model_state_dict`` runs in ``strict=False`` mode on load so the
            (already populated from ``model_path``) base params are left untouched.
    """

    def __init__(self, model, trainable_only: bool = False, parallel_state=None):
        self.model = model
        self.trainable_only = trainable_only

        # Determine whether this is ExtraParallel+FSDP2 case
        # If so, we need to restore Para(e.g. EP)-dim before saving to DCP
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )

    @torch.no_grad()
    def state_dict(self):
        options = StateDictOptions(ignore_frozen_params=True) if self.trainable_only else None
        model_state_dict = get_model_state_dict(model=self.model, options=options)
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore ExtraParallel dim for ExtraParallel (e.g. Experts/Embeds) module"
            )
            # As fsdp+extra parallel and pure extra parallel have different placements, e.g. [Shard(0), Shard(1)] and [Shard(0)],
            # restoring state dict should be extra parallel aware.
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "restore")

        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop ExtraParallel-dim when loading from DCP checkpoints
        so that ExtraParallel-FSDP would not be confused
        """
        model_state_dict = state_dict
        if self.should_extra_parallel_aware:
            model_state_dict = self.get_state_dict_with_extra_parallel_dim_preprocess(model_state_dict, "drop")

        options = StateDictOptions(strict=False) if self.trainable_only else None
        set_model_state_dict(model=self.model, model_state_dict=model_state_dict, options=options)

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="exact",
        )


class OptimizerState(Stateful):
    """A wrapper around an optimizer to make it stateful.

    On save, only optimizer state that actually exists is persisted — params
    that never received a gradient (e.g. unused MoE experts, frozen LoRA
    base weights) are simply absent from the checkpoint.

    On load, ``allow_partial_load=True`` is passed to the DCP load planner
    so missing optimizer entries are skipped.  For a fresh optimizer (the
    normal resume path), ``set_optimizer_state_dict`` internally calls
    ``_init_optim_state`` which pre-fills zero/default state for every
    param; DCP then overwrites the entries that exist in the checkpoint.
    Params absent from the checkpoint keep their default-initialised state,
    equivalent to what AdamW would create on the next ``step()`` call.

    Note: ``allow_partial_load`` is set globally on the DCP planner (it
    cannot be scoped to optimizer-only). ``_ModelStrictLoadPlanner`` therefore
    validates model-key completeness from checkpoint metadata before loading a
    non-LoRA full-model DCP.

    Args:
        model: the model whose parameters the optimizer owns.
        optimizer: the optimizer to save or load.
        parallel_state: optional parallel state; otherwise ``get_parallel_state()`` is used.
        load: when ``True`` (the normal resume path), ``state_dict()`` returns a
            dense state dict so DCP knows where to place restored optimizer
            entries.  When ``False`` (the save path), the returned dict is
            sparse and skips sub-optimizers that have never been stepped,
            avoiding synthetic zero state for unused parameters (e.g. unused
            MoE experts) in the checkpoint.
    """

    def __init__(self, model, optimizer, parallel_state=None, *, load: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )
        self._load = load

    def state_dict(self):
        if self.should_extra_parallel_aware:
            logger.info_rank0(
                "Getting optimizer state_dict from OptimizerState wrapper, would restore ExtraParallel dim for Experts module"
            )
            assert self.optimizer._is_multi_optimizer, (
                "ExtraParallel is enabled but optimizer is not a MultiOptimizer instance"
            )
            vanilla_optim_sd = self.optimizer.state_dict()
            optim_sd_with_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                vanilla_optim_sd, "restore"
            )
            return optim_sd_with_extra_parallel_dim

        if getattr(self.optimizer, "_is_multi_optimizer", False):
            if self._load:
                return self.optimizer.state_dict()
            return self.optimizer._sparse_state_dict()

        return get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)

    def load_state_dict(self, state_dict):
        optim_state_from_dcp_load = state_dict
        if self.should_extra_parallel_aware:
            # we need to drop ExtraParallel dim before loading them into optimizers
            optim_state_without_extra_parallel_dim = self.get_state_dict_with_extra_parallel_dim_preprocess(
                optim_state_from_dcp_load, "drop"
            )
            # Delegate to MultiOptimizer (it will split/filter correctly)
            self.optimizer.load_state_dict(optim_state_without_extra_parallel_dim)
            # MultiOptimizer sub-optimizers can also lose param-group hyperparams
            # (betas/...) for empty groups after load; restore recurses into them.
            restore_optimizer_param_group_defaults(self.optimizer)
            return

        if getattr(self.optimizer, "_is_multi_optimizer", False):
            self.optimizer.load_state_dict(optim_state_from_dcp_load)
            restore_optimizer_param_group_defaults(self.optimizer)
            return

        # Single torch optimizer.
        # ``strict=False`` matches the DCP planner's allow_partial_load intent:
        # params that never received a gradient (and thus have no saved Adam
        # state) keep the default-initialized state that
        # ``set_optimizer_state_dict`` / ``_init_optim_state`` already created.
        # Torch 2.11+ raises under the default strict=True when any
        # requires_grad param is missing from the checkpoint (DeepSeek-V4
        # indexer ``position_bias`` is one such case on short toy runs).
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
            options=StateDictOptions(strict=False),
        )
        restore_optimizer_param_group_defaults(self.optimizer)

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        return _apply_extra_parallel_dim(
            state_dict,
            self.extra_parallel_fqn2spec_info,
            self.parallel_state,
            action,
            key_match="substring",
        )


def drop_extra_parallel_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh, fsdp_shard_dim: int = 1):
    """
    Drop ExtraParallel dims after loading from DCP so that ExtraParallel-FSDP would not be confused.

    ``device_mesh`` is the FSDP sub-mesh (ExtraParallel dim already excluded):
    1D ``(ep_fsdp,)`` for plain FSDP, or 2D ``(ep_replicate, ep_fsdp)`` for HSDP.
    ``fsdp_shard_dim`` is the tensor dim FSDP shards along (1 by default, 0 for
    the Muon zero-comm layout). The number of placements on the loaded DTensor
    reflects the full saved mesh:

    * 1 placement: pure ExtraParallel, no FSDP -> return the plain local tensor.
    * 2 placements: ``(ep_fsdp, ep)`` -> keep FSDP ``Shard(fsdp_shard_dim)`` on
      the 1D sub-mesh.
    * 3 placements: ``(ep_replicate, ep_fsdp, ep)`` (HSDP) -> keep
      ``[Replicate(), Shard(fsdp_shard_dim)]`` on the 2D sub-mesh.
    """

    num_placements = len(loaded_tensor.placements)
    if num_placements == 1:
        tensor_to_put = loaded_tensor.to_local()
    elif num_placements == 2:
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(fsdp_shard_dim)]
        )
    elif num_placements == 3:
        tensor_to_put = DTensor.from_local(
            loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Replicate(), Shard(fsdp_shard_dim)]
        )
    else:
        raise RuntimeError(
            "Expect ExtraParallel parameters from checkpoints to be DTensor with 1-dim (no FSDP), "
            f"2-dim (ExtraParallel+FSDP) or 3-dim (ExtraParallel+HSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def restore_extra_parallel_dim(
    orgin_tensor: torch.Tensor,
    fsdp_mesh: DeviceMesh,
    extra_parallel_fsdp_mesh: DeviceMesh,
    ep_shard_dim: int = 0,
    fsdp_shard_dim: int = 1,
):
    """
    Restore ExtraParallel dim so that DCP can be aware about ExtraParallel ranks

    The ExtraParallel (e.g. ``ep``) dim and the FSDP dim shard the tensor along
    DIFFERENT axes: ``ep_shard_dim`` (the EP plan's ``Shard.dim``, dim 0 for
    experts) and ``fsdp_shard_dim`` (1 by default, 0 for the Muon zero-comm
    layout). They must be mapped to the matching mesh dims explicitly.

    args:
        orgin_tensor (torch.Tensor): The orgin tensor (FSDP-local shard).
        fsdp_mesh (DeviceMesh): The full ExtraParallel mesh, i.e. 2D
            ``(ep_fsdp, ep)`` for plain FSDP or 3D
            ``(ep_replicate, ep_fsdp, ep)`` for HSDP.
        extra_parallel_fsdp_mesh (DeviceMesh): The FSDP sub-mesh (ExtraParallel
            dim excluded), used for the pure-ExtraParallel (no FSDP) path.
        ep_shard_dim (int): Tensor dim the ExtraParallel dim shards along.
        fsdp_shard_dim (int): Tensor dim FSDP shards along.

    Note:
        When ``ep_shard_dim == fsdp_shard_dim`` (e.g. the Muon zero-comm layout
        shards experts on dim 0 just like EP), both mesh dims split the SAME
        tensor dim. The mesh order ``(ep_fsdp, ep)`` then composes ``ep_fsdp``
        OUTER of ``ep``, whereas the physical layout is EP-outer / FSDP-inner,
        so the reconstructed GLOBAL tensor is block-transposed along that dim.
        Save->load into the SAME parallel config still round-trips correctly
        (drop is the exact inverse); only resharding / external reads of such a
        checkpoint see the permuted layout. Fixing this needs the EP mesh dims
        reordered so ``ep`` precedes ``ep_fsdp`` (a parallel_state change).
    """
    assert fsdp_mesh.ndim in (2, 3), f"global_mesh.ndim must be 2 or 3, got {fsdp_mesh.ndim}"

    if isinstance(orgin_tensor, DTensor):
        # ExtraParallel+FSDP2. mesh order is (ep_fsdp, ep) or, for HSDP,
        # (ep_replicate, ep_fsdp, ep): ep_fsdp -> Shard(fsdp_shard_dim),
        # ep -> Shard(ep_shard_dim), ep_replicate -> Replicate().
        if fsdp_mesh.ndim == 3:
            placements = [Replicate(), Shard(fsdp_shard_dim), Shard(ep_shard_dim)]
        else:
            placements = [Shard(fsdp_shard_dim), Shard(ep_shard_dim)]
        dtensor = DTensor.from_local(orgin_tensor._local_tensor, device_mesh=fsdp_mesh, placements=placements)
    elif torch.is_tensor(orgin_tensor):
        # If there is no FSDP but only ExtraParallel
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=extra_parallel_fsdp_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


def _local_rank() -> int:
    """This process's rank within its node, as set by the elastic launcher.

    Elects one rank per node for the node-local staging work; defaults to 0 so
    single-process runs still take the leader path.
    """
    value = os.environ.get("LOCAL_RANK", "0")
    return int(value) if value.isdigit() else 0


def _stage_key(path: str) -> str:
    """Directory name that isolates one run's staged files from another's.

    Substituting separators would collide -- it maps ``/tmp/a_b/c`` and
    ``/tmp/a/b_c`` onto one name -- so the name is a digest of the absolute path,
    prefixed with the basename to stay identifiable on disk.
    """
    absolute = os.path.abspath(path)
    digest = hashlib.sha256(absolute.encode("utf-8")).hexdigest()[:16]
    return f"{os.path.basename(absolute) or 'ckpt'}-{digest}"


class _Promotion:
    """Failure state shared by the phases of one promotion.

    ``error`` is what this rank saw, ``failed`` what the whole group saw: the work
    is split across ranks -- one leader per node copies that node's files -- so a
    failure starts out visible to a single rank.
    """

    def __init__(self) -> None:
        self.error: Optional[BaseException] = None
        self.failed = False


def _promotion_phase(state: _Promotion, work, *, participates: bool, always: bool = False) -> None:
    """Run one phase on the ranks that take part, then let every rank agree on the result.

    The closing reduction is the phase's only collective and every rank reaches it
    on every path, including the failing one. Collectives are untagged, so a rank
    that returned early would leave the others pairing up with the wrong one from
    then on and the save would hang instead of failing; one collective per phase
    keeps the counts equal by construction rather than by inspection.

    ``always`` marks a phase that must run even after a failure -- cleanup.

    ``BaseException`` because ``work`` is arbitrary and the guarantee above is
    structural: anything that escapes this catch skips the reduction, and the
    ranks that did reach it wait for a peer that has already left.
    """
    if participates and (always or not state.failed):
        try:
            work()
        except BaseException as e:  # noqa: BLE001 - raised once every phase is done
            if state.error is None:
                state.error = e
    state.failed = any_rank_failed(state.error is not None) or state.failed


_STAGE_ROOT = "veomni_ckpt_stage"


def _prepare_stage_dir(stage_dir: str, path: str) -> str:
    """Create the empty staging directory for the run writing to ``path``.

    One directory per run, shared by every checkpoint it writes and emptied
    first. That is also how a save killed part-way is cleaned up: its copy -- the
    size of the model plus its optimizer state -- is left exactly where the next
    save clears it, instead of stranded under a key naming its own step.

    Keyed on ``path`` because ``stage_dir`` is often something generic like /tmp
    and this directory gets swept: two runs sharing a node must not land in the
    same place, or one would delete the other's staged data.

    Only the node leader touches the filesystem; peers would race the sweep and
    have no need to, since the reduction below is a collective. That reduction
    also keeps a per-node failure -- a full or read-only scratch disk -- from
    being seen by one rank alone, which would leave the rest waiting in
    ``dcp.save`` on a collective that never arrives.
    """
    stage_path = os.path.join(stage_dir, _STAGE_ROOT, _stage_key(path))
    error: Optional[Exception] = None
    if _local_rank() == 0:
        try:
            shutil.rmtree(stage_path, ignore_errors=True)
            os.makedirs(stage_path, exist_ok=True)
        except Exception as e:  # noqa: BLE001 - raised once every rank has agreed
            error = e
    if any_rank_failed(error is not None):
        raise error or RuntimeError(f"another rank could not prepare a staging directory under {stage_dir}")
    return stage_path


def _promote_staged_checkpoint(stage_path: str, final_path: str, step_root: Optional[str] = None) -> None:
    """Copy a staged checkpoint to its destination, then drop the staged copy.

    The staging directory is node-local, so one rank per node copies all of it
    rather than each rank working out which files it wrote; that keeps this
    independent of DCP's file naming.

    Four phases, each ending in a single collective (see ``_promotion_phase``),
    with errors re-raised on every rank once every phase has run.

    ``.metadata`` is what DCP reads as "this DCP directory is complete", and the
    staged tree holds one per directory -- ``ckpt/`` and ``optimizer/``. The
    destination is emptied first, before anything is overwritten, and the new
    markers go last and only if every rank's data landed -- so a reader sees
    either the previous complete checkpoint or none, never a completion marker
    over data that is only partly there, and never this save's files mixed with
    a previous one's.

    This is the *only* place a staged save invalidates its destination. Doing it
    when the save starts, as the unstaged path does, would throw away the
    previous checkpoint before the new one exists anywhere -- and keeping that
    checkpoint readable until the last possible moment is what staging is for.

    ``step_root`` is the step directory, given when the destination may hold a
    pre-split checkpoint whose marker sits there rather than inside
    ``final_path``. Nested files are copied in the data phase, before any
    ``.metadata`` is. ``lr_scheduler.pt`` is one of those files.
    """
    metadata_name = DCP_MARKER_FILENAME
    is_node_leader = _local_rank() == 0
    is_coordinator = (not dist.is_initialized()) or dist.get_rank() == 0
    state = _Promotion()

    def staged_markers() -> list[str]:
        """Relative path of every ``.metadata`` in the staged tree.

        DCP writes the marker from its coordinator rank, so only that rank has
        them staged -- which is also the only rank that deletes and copies them.
        """
        rels: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(stage_path):
            for filename in filenames:
                if filename == metadata_name:
                    rels.append(os.path.relpath(os.path.join(dirpath, filename), stage_path))
        return sorted(rels)

    def clear_destination() -> None:
        """Empty the destination, so the staged tree replaces it rather than merges.

        Deleting only the files about to be overwritten would leave whatever the
        previous save wrote and this one does not: a weights-only save over a
        step that has an optimizer keeps that optimizer's shards *and* its
        ``.metadata``, and the step then reads as a complete checkpoint pairing
        this run's weights with an earlier run's optimizer.

        Nothing is lost that the destination could still have used. A marker has
        to go before its shards are overwritten either way, and without one DCP
        will not read the directory at all.

        One module's directory. A sibling module of the same job lives beside it,
        is not being written, and keeps everything it has.
        """
        # The step may predate this layout, in which case DCP left its marker at
        # the step root rather than in here. Delete this import (and
        # veomni/checkpoint/legacy_v0_1_12.py) to drop that layout.
        from .legacy_v0_1_12 import drop_marker

        shutil.rmtree(final_path, ignore_errors=True)
        os.makedirs(final_path, exist_ok=True)
        if step_root is not None:
            drop_marker(step_root)

    def copy_this_nodes_files() -> None:
        """Copy every staged file on this node, except the completion markers.

        Walks the tree so nested directories are copied, not skipped.
        """
        os.makedirs(final_path, exist_ok=True)

        names: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(stage_path):
            for filename in filenames:
                if filename == metadata_name:
                    continue
                names.append(os.path.relpath(os.path.join(dirpath, filename), stage_path))
        names.sort()

        def _copy(rel: str) -> None:
            """Copy one staged file to the destination, preserving its relative path."""
            src = os.path.join(stage_path, rel)
            dst = os.path.join(final_path, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)

        if names:
            with ThreadPoolExecutor(max_workers=min(16, len(names))) as pool:
                list(pool.map(_copy, names))

    def copy_markers() -> None:
        """Copy the completion markers last, or leave none behind if that fails."""
        copied: list[str] = []
        try:
            for rel in staged_markers():
                dst = os.path.join(final_path, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(stage_path, rel), dst)
                copied.append(dst)
        except BaseException:
            # copyfile creates the destination before writing it, so a failure
            # can leave a truncated marker -- worse than none, since DCP would
            # read it as a complete directory. Earlier markers go too: half a
            # model is not resumable, and leaving one valid directory behind
            # would misreport which part survived.
            for dst in copied + [os.path.join(final_path, rel) for rel in staged_markers()]:
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                except OSError:
                    logger.error(f"could not remove a partially written {dst}", exc_info=True)
            raise

    def drop_staged_copy() -> None:
        """Free the scratch disk.

        Runs after a failure too: the copy is as large as the model plus its
        optimizer state, and nothing is lost by dropping it -- without a marker
        the destination reads as incomplete, which it is.
        """
        shutil.rmtree(stage_path, ignore_errors=True)

    _promotion_phase(state, clear_destination, participates=is_coordinator)
    _promotion_phase(state, copy_this_nodes_files, participates=is_node_leader)
    _promotion_phase(state, copy_markers, participates=is_coordinator)
    _promotion_phase(state, drop_staged_copy, participates=is_node_leader, always=True)

    if state.error is not None:
        raise state.error
    if state.failed:
        raise RuntimeError("checkpoint promotion failed on another rank; no completion marker was written")


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    # One in-flight async save per slot. A step writes the weights and the
    # optimizer into two directories, so there are two saves to track rather than
    # one.
    #
    # Each slot gets its own Gloo group. ``dcp.async_save`` hands the group it is
    # given to a background thread that runs the *whole* save, collectives
    # included -- it asserts a CPU backend for exactly that reason. Two saves
    # sharing one group would have two threads issuing collectives on it at once,
    # and ranks that interleave them differently deadlock. Separate groups let
    # the two writes overlap; sharing one would force the second to wait out the
    # first, which is what draining before each save used to do.
    _save_futures: Dict[str, Any] = {}
    _async_process_groups: Dict[str, Any] = {}

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: int = None,
        module: str = "",
        trainable_only: bool = False,
        save_to_lowest_rank: bool = False,
        parallel_state=None,
        stage_dir: Optional[str] = None,
    ) -> None:
        """
        save training state to distributed checkpoint

        Writes three things under ``model/`` (see ``veomni.checkpoint.layout``):
        ``ckpt/`` for the weights, ``optimizer/`` for the optimizer state, and a
        replicated ``lr_scheduler.pt``. Weights and optimizer are separate DCP
        directories so the weights can be shipped or converted on their own; a
        single directory interleaves both into the same ``.distcp`` files.

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            global_steps: step this checkpoint belongs to. Given, the checkpoint goes
                into a per-step subdirectory of ``path`` and ``path`` identifies the run,
                which is what ``stage_dir`` keys its staging directory on. Callers that
                fold the step into ``path`` themselves get a staging directory per step.
            module: name of the model this checkpoint belongs to, for a job that trains
                several (SeedOmni V2). Nests every artifact under ``model/<module>/``.
                Empty for a single-model job, which is the only difference between the
                two cases.
            trainable_only: when True, only persist parameters with ``requires_grad=True``
                (LoRA / PEFT path). Frozen base weights are skipped on save and must be
                re-materialised from ``model.model_path`` at resume time. The optimizer
                state is already trainable-only by construction (the optimizer is built
                from ``filter(lambda p: p.requires_grad, ...)``), so this flag only
                affects the model state dump.
            save_to_lowest_rank: forwarded to the DCP ``DefaultSavePlanner``. When True, each
                replicated shard is written by the lowest global rank that holds it, instead of
                being load-balanced across all replica holders. On a non-shared filesystem this
                concentrates the (already deduplicated) copy onto the lowest-ranked replica group
                instead of scattering it across replicas; in the standard HSDP layout (shard within
                a node, replicate across nodes) that group is one node, which then holds a complete
                checkpoint. Note this only consolidates *replicated* data: unique shards from
                expert/tensor/pipeline parallelism are never deduplicated and remain distributed.
                See ``CheckpointConfig.dcp_save_to_lowest_rank``.
            stage_dir: write the checkpoint here and copy it to ``path`` afterwards,
                instead of writing straight to ``path``. Intended for a destination far
                slower than local disk. The whole ``model/`` subtree — both DCP
                directories and the scheduler sidecar — lands under the staging
                directory and is copied together, with every ``.metadata`` copied
                last. Staging is keyed per module as well as per run, so two modules
                of one job do not sweep each other's staged files. The caller
                owns the choice: this does not probe
                for a usable directory or check free space, and an unusable ``stage_dir``
                fails the save rather than silently writing elsewhere. See
                ``CheckpointConfig.stage_dir``.
        return:
            None
        """
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        # Rejected up front, before anything reaches disk: an async write is still
        # running when save() returns and drops the staged copy, so it would write
        # straight to the slow destination staging was meant to avoid.
        if stage_dir and save_async:
            raise ValueError("stage_dir cannot be combined with save_async")

        # ``is not None`` rather than truthiness: step 0 is a step like any other, and
        # folding it onto ``path`` would write it over the run's own directory.
        checkpoint_dir = step_dir(path, global_steps) if global_steps is not None else path
        model_root = model_dir(checkpoint_dir, module)

        # Keyed on the module's own run-level path, not just ``path``: a
        # multi-module job calls this once per module, and a single key would have
        # each module clear the previous one's staged files.
        stage_key_path = os.path.join(path, module) if module else path
        stage_path = _prepare_stage_dir(stage_dir, stage_key_path) if stage_dir else None
        write_root = stage_path or model_root

        if stage_path is None:
            # Nothing stands between this save and the previous checkpoint's
            # shards, so its markers have to go before the first byte lands --
            # otherwise a save that dies part-way leaves one describing shards
            # that are half this step and half the last. A staged save invalidates
            # its destination in ``_promote_staged_checkpoint`` instead, at the
            # point where it is about to overwrite it for real.
            #
            # The manifest is not ours either way; ``GlobalStateCallback`` clears
            # the one it writes.
            cls._remove_dcp_markers(checkpoint_dir, module)

        cls._create_checkpoint_dir(write_root)

        # Sidecar first, then the DCP directories. Ordering no longer carries the
        # completeness guarantee it used to — ``checkpoint_manifest.json`` does,
        # and it is written only after every module's save has returned — but
        # writing the small replicated file first still means a save that dies
        # part-way leaves less behind.
        cls._save_lr_scheduler(checkpoint_dir=write_root, state=state)

        try:
            cls.execute_save(
                save_state={
                    "model": ModelState(state["model"], trainable_only=trainable_only, parallel_state=parallel_state)
                },
                storage_writer=cls._create_storage_writer(os.path.join(write_root, WEIGHTS_DIRNAME)),
                save_async=save_async,
                save_to_lowest_rank=save_to_lowest_rank,
                slot=WEIGHTS_DIRNAME,
            )

            if "optimizer" in state and state["optimizer"] is not None:
                cls.execute_save(
                    save_state={
                        "optimizer": OptimizerState(
                            model=state["model"],
                            optimizer=state["optimizer"],
                            parallel_state=parallel_state,
                            load=False,
                        )
                    },
                    storage_writer=cls._create_storage_writer(os.path.join(write_root, OPTIMIZER_DIRNAME)),
                    save_async=save_async,
                    save_to_lowest_rank=save_to_lowest_rank,
                    slot=OPTIMIZER_DIRNAME,
                )
        except BaseException:
            if stage_path is not None and _local_rank() == 0:
                shutil.rmtree(stage_path, ignore_errors=True)
            raise

        if stage_path is not None:
            _promote_staged_checkpoint(stage_path, model_root, step_root=checkpoint_dir)

        logger.info_rank0(f"Saved checkpoint to {model_root}")

    @classmethod
    def _remove_dcp_markers(cls, checkpoint_dir: str, module: str) -> None:
        """Delete the ``.metadata`` files vouching for what this save overwrites.

        The unstaged path's invalidation, run before the first byte lands. A
        staged save has ``_promote_staged_checkpoint`` do the same job at
        promotion time, because until then it has overwritten nothing.

        Two places to look. This module's ``ckpt/`` and ``optimizer/``, and the
        step root, where an older VeOmni's fused save left the same file. Both
        are DCP's own marker, which is why both are dropped here rather than by
        the callback: this class is the one that puts them back.

        Only this module's. A sibling module is not being rewritten and its
        markers still hold. The step's ``checkpoint_manifest.json`` is not ours
        either -- ``GlobalStateCallback`` writes it and clears it.

        One rank owns these files, so only that rank deletes them, and the
        reduction turns a failure there into one every rank sees.
        """
        # The step being overwritten may predate this layout, in which case DCP
        # left its marker at the step root. Delete this import (and
        # veomni/checkpoint/legacy_v0_1_12.py) to drop that layout.
        from .legacy_v0_1_12 import drop_marker

        is_coordinator = (not dist.is_initialized()) or dist.get_rank() == 0
        error: Optional[Exception] = None
        if is_coordinator:
            try:
                drop_marker(checkpoint_dir)
                for marker in dcp_markers(checkpoint_dir, [module]):
                    if os.path.exists(marker):
                        os.remove(marker)
            except Exception as e:  # noqa: BLE001 - raised once every rank has agreed
                error = e
        raise_if_any_rank_failed(error, f"removing the old DCP markers under {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        module: str = "",
        trainable_only: bool = False,
        parallel_state=None,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint

        Mirrors :meth:`save`: weights from ``model/<module>/ckpt``, optimizer from
        ``model/<module>/optimizer``, scheduler from the sidecar beside them. A
        checkpoint written before the split has no ``model/`` at all and keeps
        both in one directory; that shape is detected and read as-is.

        args:
            path: step directory to load from
            state: state to load, "model" is required; "optimizer" and "lr_scheduler" are optional
            process_group: process group for loading checkpoint
            module: name of the model to load, for a job that trains several.
                Empty for a single-model job. See :meth:`save`.
            trainable_only: when True, ``set_model_state_dict`` runs in non-strict
                mode (``StateDictOptions(strict=False)``). Use this for LoRA / PEFT
                resumes where the DCP only contains trainable adapter weights and the
                frozen base must come from ``model.model_path``. Safe to enable when
                the DCP is full (extra strictness is just dropped).

        return:
            state: state loaded
        """
        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        def _model_state() -> "ModelState":
            return ModelState(state["model"], trainable_only=trainable_only, parallel_state=parallel_state)

        def _optimizer_state() -> "OptimizerState":
            return OptimizerState(
                model=state["model"],
                optimizer=state["optimizer"],
                parallel_state=parallel_state,
                load=True,
            )

        wants_optimizer = state.get("optimizer") is not None

        fused_dir = cls._legacy_fused_dir(path, module)
        if fused_dir is not None:
            load_state: Dict[str, Any] = {"model": _model_state()}
            if wants_optimizer:
                load_state["optimizer"] = _optimizer_state()
            dcp.load(
                state_dict=load_state,
                storage_reader=cls._create_storage_reader(fused_dir),
                process_group=process_group,
                planner=_ModelStrictLoadPlanner(strict_model=not trainable_only),
            )
            cls._load_lr_scheduler(checkpoint_dir=fused_dir, state=state)
            logger.info_rank0(f"Loaded pre-split checkpoint from {fused_dir}")
            return state

        model_root = model_dir(path, module)
        dcp.load(
            state_dict={"model": _model_state()},
            storage_reader=cls._create_storage_reader(weights_dir(path, module)),
            process_group=process_group,
            planner=_ModelStrictLoadPlanner(strict_model=not trainable_only),
        )
        if wants_optimizer:
            dcp.load(
                state_dict={"optimizer": _optimizer_state()},
                storage_reader=cls._create_storage_reader(optimizer_dir(path, module)),
                process_group=process_group,
                # The strict check looks for missing ``model`` keys; there are none
                # in an optimizer-only load, so strictness has nothing to say here.
                planner=_ModelStrictLoadPlanner(strict_model=False),
            )

        cls._load_lr_scheduler(checkpoint_dir=model_root, state=state)

        logger.info_rank0(f"Loaded checkpoint from {model_root}")

        return state

    @classmethod
    def _legacy_fused_dir(cls, path: str, module: str) -> Optional[str]:
        """Directory holding a pre-split checkpoint, or ``None`` for a current one.

        Before weights and optimizer were separated there was one DCP directory
        per model, marked by its own ``.metadata``: at the step root for a
        single-model job, and at ``<step>/<module>/`` for SeedOmni V2. Neither has
        a ``model/`` subtree, which is what tells the two apart.
        """
        if os.path.exists(os.path.join(weights_dir(path, module), DCP_MARKER_FILENAME)):
            return None
        legacy_dir = os.path.join(path, module) if module else path
        if os.path.exists(os.path.join(legacy_dir, DCP_MARKER_FILENAME)):
            return legacy_dir
        return None

    @classmethod
    def wait_for_pending_save(cls) -> None:
        """Block until every pending async DCP save completes.

        Safe to call when no save is pending (no-op).  Every rank ends up
        raising if any rank's save failed, and the reduction that decides
        that is also the synchronization callers rely on before starting a
        new collective.

        This is the single entrypoint for all async-save coordination —
        prefer calling this over poking ``_save_futures`` directly.

        Every slot is drained even when one raises, so a failed weights write
        cannot leave the optimizer write running into the next step's
        collectives. That is also why the catch below is ``BaseException``:
        DCP reports a failed save as ``CheckpointException``, which derives
        from ``BaseException`` rather than ``Exception``, so catching the
        latter would let the very failures this drains for escape the loop --
        leaving the remaining slots running and their futures unreachable.
        """
        if not cls._save_futures:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        futures = cls._save_futures
        cls._save_futures = {}
        error: Optional[BaseException] = None
        for slot, future in futures.items():
            try:
                logger.info(f"[RANK {rank}] waiting for pending DCP save ({slot}) to end...")
                future.result()
            except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
                logger.error(f"[RANK {rank}] pending async DCP save ({slot}) raised; propagating", exc_info=True)
                if error is None:
                    error = e
        raise_if_any_rank_failed(error, "a pending async DCP save")

    @classmethod
    def _drain_slot(cls, slot: str) -> None:
        """Finish this slot's previous save before its process group is reused.

        DCP surfaces an async failure through the future of the rank that hit
        it, and the save's own process group does not reduce that across the
        group. Raising here on the failing rank alone would leave its peers in
        the reduction below with nobody to meet -- which is also why the catch
        is ``BaseException``: DCP's ``CheckpointException`` does not derive
        from ``Exception``, and letting it past the reduction is exactly the
        hang this method exists to prevent.
        """
        future = cls._save_futures.pop(slot, None)
        if future is None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        error: Optional[BaseException] = None
        try:
            logger.info(f"[RANK {rank}] waiting for previous DCP save ({slot}) to end...")
            future.result()
        except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {rank}] previous async DCP save ({slot}) raised; propagating", exc_info=True)
            error = e
        raise_if_any_rank_failed(error, f"the previous async DCP save ({slot})")

    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
        save_to_lowest_rank: bool = False,
        slot: str = WEIGHTS_DIRNAME,
    ) -> None:
        """Execute DCP save with optional async support.

        ``save_to_lowest_rank`` is forwarded to ``DefaultSavePlanner``; the default
        (False) preserves DCP's load-balanced write assignment across replica holders.

        ``slot`` names the concurrent async save this call belongs to — one per
        directory a step writes. Only the *same* slot's previous save is drained,
        so the weights and the optimizer overlap within a step while neither can
        outlive its own next write.
        """
        planner = DefaultSavePlanner(dedup_save_to_lowest_rank=save_to_lowest_rank)
        if save_async:
            # Lazily create this slot's dedicated Gloo process group. Creating a
            # group is itself collective, and every rank runs the same save
            # sequence, so every rank creates the same groups in the same order.
            if slot not in cls._async_process_groups:
                cls._async_process_groups[slot] = dist.new_group(backend="gloo")

            cls._drain_slot(slot)

            cls._save_futures[slot] = dcp.async_save(
                state_dict=save_state,
                storage_writer=storage_writer,
                process_group=cls._async_process_groups[slot],
                planner=planner,
            )
        else:
            dcp.save(
                state_dict=save_state,
                storage_writer=storage_writer,
                planner=planner,
            )
            if dist.is_initialized():
                dist.barrier()
            gc.collect()
            empty_cache()
            synchronize()

    # Private helper methods
    @classmethod
    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:
        """Create checkpoint directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)

    @classmethod
    def _create_storage_reader(cls, checkpoint_dir: str) -> FileSystemReader:
        """Create storage reader for DCP."""
        return FileSystemReader(checkpoint_dir)

    @classmethod
    def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
        """Create storage writer for DCP."""
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=16,
            single_file_per_rank=True,
            sync_files=False,
        )

    @classmethod
    def _save_lr_scheduler(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Pickle ``lr_scheduler.state_dict`` into a single ``lr_scheduler.pt``.

        The scheduler is replicated across ranks, so only rank 0 writes. Every
        rank still joins the reduction afterwards: a failed write must not let
        peers enter the DCP collective alone.
        """
        error: Optional[Exception] = None
        is_writer = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_writer:
            try:
                if _LR_SCHEDULER_KEY not in state:
                    logger.warning_rank0("lr_scheduler not found in state, skipping lr_scheduler save")
                else:
                    lr_scheduler = state[_LR_SCHEDULER_KEY]
                    if lr_scheduler is not None:
                        torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, _LR_SCHEDULER_FILENAME))
            except Exception as e:  # noqa: BLE001 - raised once every rank has agreed
                error = e
        if any_rank_failed(error is not None):
            raise error or RuntimeError("another rank could not save lr_scheduler")

    @classmethod
    def _load_lr_scheduler(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Load ``lr_scheduler.pt`` into ``lr_scheduler``. Every rank reads the same file."""
        if _LR_SCHEDULER_KEY not in state:
            logger.warning_rank0("lr_scheduler not found in state, skipping lr_scheduler load")
            return
        lr_scheduler = state[_LR_SCHEDULER_KEY]
        if lr_scheduler is None:
            return

        lr_scheduler_path = os.path.join(checkpoint_dir, _LR_SCHEDULER_FILENAME)
        if os.path.exists(lr_scheduler_path):
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_path, weights_only=False))
            return

        # Delete this import (and veomni/checkpoint/legacy_v0_1_12.py) to drop 0.1.12 extra_state resume.
        from .legacy_v0_1_12 import apply_legacy_lr_scheduler

        if apply_legacy_lr_scheduler(checkpoint_dir, lr_scheduler):
            return

        raise FileNotFoundError(
            f"lr_scheduler sidecar not found at {lr_scheduler_path}. "
            "This layout writes lr_scheduler.pt next to the DCP shards "
            "(see docs/usage/checkpoint.md)."
        )


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return size in bytes for a given dtype."""
    return torch.empty((), dtype=dtype).element_size()


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
    - "model.base_model.*" -> "base_model.*" (PEFT LoRA adapter case;
      ``save_lora_adapter_with_dcp`` re-prefixes already-PEFT-prefixed keys
      with ``model.`` so DCP keeps them, and we strip that here on read)
    - Other "model.*" keys -> log warning and strip "model." prefix
    """
    if not key.startswith("model."):
        return None

    if key.startswith("model.model."):
        # Standard case: model.model.* -> model.*
        return key[6:]  # Remove first "model." prefix
    elif key == "model.lm_head.weight":
        # Special case: model.lm_head.weight -> lm_head.weight
        return "lm_head.weight"
    elif key.startswith("model.base_model."):
        # PEFT LoRA adapter save: ``save_lora_adapter_with_dcp`` writes keys
        # of the form ``model.base_model.model.<...>.lora_A.weight`` so the
        # DCP-side ``model.`` filter keeps them. The HF-side adapter file is
        # the standard PEFT layout ``base_model.model.<...>.lora_A.weight``,
        # which is exactly ``key[6:]``. This is a known, expected pattern
        # — silent strip, no warning.
        return key[6:]
    else:
        # Other keys with single "model." prefix - log and strip prefix
        logger.warning(
            f"Found key with single 'model.' prefix that doesn't match expected patterns: '{key}'. "
            f"Converting to '{key[6:]}' by stripping 'model.' prefix."
        )
        return key[6:]


def _get_sharding_plan(
    checkpoint_path: Union[str, os.PathLike],
    shard_size: int = None,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    Create sharding plan from checkpoint metadata without loading weights.

    Returns:
        shards: List of {hf_key: dcp_key} dicts per shard
        total_size: Total size in bytes
        all_dcp_keys: All valid DCP model keys
    """
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {checkpoint_path}")

    # Collect model tensors and calculate sizes
    tensor_infos = []
    all_dcp_keys = []

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key:
            # Determine dtype for size calculation
            if not hasattr(tensor_meta.properties, "dtype"):
                raise ValueError(
                    f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                )
            source_dtype = tensor_meta.properties.dtype
            if save_dtype and source_dtype.is_floating_point:
                dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            else:
                dtype = source_dtype

            # Calculate tensor size in bytes
            numel = 1
            for dim in tensor_meta.size:
                numel *= dim

            byte_size = numel * get_dtype_size(dtype)

            tensor_infos.append({"dcp_key": key, "hf_key": hf_key, "size": byte_size, "metadata": tensor_meta})
            all_dcp_keys.append(key)

    # Sort by key name for deterministic output
    tensor_infos.sort(key=lambda x: x["hf_key"])

    # Pack tensors into shards
    shards = []
    current_shard = {}
    current_shard_size = 0
    total_size = 0

    for info in tensor_infos:
        size = info["size"]
        total_size += size

        # Start new shard if adding this tensor exceeds shard_size (unless current shard is empty)
        if shard_size is not None and current_shard and (current_shard_size + size > shard_size):
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[info["hf_key"]] = info["dcp_key"]
        current_shard_size += size

    if current_shard:
        shards.append(current_shard)
    if shard_size is None:
        assert len(shards) == 1, "Shard size None should result in a single shard"
        shards = shards[0]
    return shards, total_size, all_dcp_keys


def _process_shard(
    shard_keys: Dict[str, str],
    checkpoint_path: str,
    save_dtype: Optional[Union[str, torch.dtype]] = None,
) -> str:
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = OrderedDict()
    dcp_keys_to_load = list(shard_keys.values())

    for dcp_key in dcp_keys_to_load:
        tensor_metadata = metadata.state_dict_metadata[dcp_key]
        if not hasattr(tensor_metadata.properties, "dtype"):
            raise ValueError(
                f"Cannot determine dtype for tensor '{dcp_key}': metadata does not contain dtype information"
            )
        state_dict[dcp_key] = torch.empty(
            tensor_metadata.size,
            dtype=tensor_metadata.properties.dtype,
        )

    # Load partial checkpoint
    load(
        state_dict,
        checkpoint_id=checkpoint_path,
        storage_reader=FileSystemReader(checkpoint_path),
        no_dist=True,
    )

    # Cast and rename tensors
    processed_dict = OrderedDict()
    target_dtype = None
    if save_dtype:
        target_dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype

    for hf_key, dcp_key in shard_keys.items():
        tensor = state_dict[dcp_key]

        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if target_dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype=target_dtype)

        # Explicitly move to CPU and detach to avoid memory retention
        processed_dict[hf_key] = tensor.cpu().detach().clone()
        # Delete the original tensor immediately
        del tensor

    # Clean up state_dict and force garbage collection
    del state_dict
    del metadata
    del reader
    gc.collect()
    empty_cache()
    return processed_dict


def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch state_dict.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    shard, _, _ = _get_sharding_plan(save_checkpoint_path)

    processed_dict = _process_shard(shard, save_checkpoint_path)

    return processed_dict
