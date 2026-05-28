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


import contextlib
import gc
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
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
from ..utils import logging
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX
from ..utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    empty_cache,
    get_device_id,
    get_device_type,
    synchronize,
)
from .checkpointer import CheckpointerBase


logger = logging.get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_EXTRA_STATE_DIR = "extra_state"


# Workaround for a class of DCP save/load hangs on multi-node setups. The
# smoking gun is ranks reporting DIFFERENT first-byte values in
# ``_pickle.UnpicklingError`` after the Metadata broadcast — e.g.
# rank{1,2,3,11,12,13,14,15} see ``'\x00'``, rank4 sees ``'\x0c'``,
# rank6 sees ``'\x9b'`` (observed on production
# Qwen3-VL-30B-A3B-Instruct ep_size=1 H100x16).
#
# If the broadcast had really synchronized, every receiver would see the
# SAME bytes (and either all pass or all fail at the same opcode). Each
# rank seeing different garbage means the broadcast returned to Python
# before NCCL actually delivered the data to that rank's buffer — the
# receivers then call ``_tensor_to_object`` which copies the GPU buffer
# to CPU memory and feeds it to ``_unpickler``. The CPU copy reads
# whatever uninitialized memory the GPU side has at that moment.
#
# This is a known shape of NCCL + caching-allocator stream tracking bug
# that fires on multi-node setups (where the broadcast traverses RDMA)
# but doesn't fire on single-node (where NVLink delivery is effectively
# synchronous). Single-node A100 and H100 verify runs both pass; only
# the multi-node prod jobs reproduce it.
#
# Fix: monkey-patch ``_tensor_to_object`` (the helper PyTorch uses inside
# ``broadcast_object_list`` to read each object out of the broadcast
# buffer) to call ``synchronize()`` first — but ONLY while a DCP save /
# load is in progress on the current thread, gated by the thread-local
# ``_in_dcp_op`` flag. This keeps the patch a no-op for every other
# ``broadcast_object_list`` caller in the process (per-step micro-batch
# broadcast in DiT trainer, weight-metadata broadcasts in
# ``models/module_utils.py`` / ``utils/lora_utils.py``, etc.), so
# importing this module doesn't inject a device sync into unrelated
# training paths. Scope is entered via the ``_dcp_sync_scope`` context
# manager wrapped around the actual ``dcp.save`` / ``dcp.async_save`` /
# ``dcp.load`` calls below. The ``IS_CUDA_AVAILABLE`` guard keeps the
# patch a no-op on CPU-only hosts.
_in_dcp_op = threading.local()


@contextlib.contextmanager
def _dcp_sync_scope():
    """Enable the ``_tensor_to_object`` device-sync workaround for the
    duration of a DCP save/load call on the current thread.

    Re-entrant: nested scopes restore the previous flag value on exit, so
    accidentally nesting (e.g. a save inside a load callback) is safe.
    """
    prev = getattr(_in_dcp_op, "active", False)
    _in_dcp_op.active = True
    try:
        yield
    finally:
        _in_dcp_op.active = prev


try:
    import torch.distributed.distributed_c10d as _c10d  # noqa: E402

    _orig_tensor_to_object = _c10d._tensor_to_object

    def _veomni_tensor_to_object_with_sync(*args, **kwargs):
        # ``*args, **kwargs`` instead of a pinned ``(tensor, tensor_size,
        # group)`` signature: ``_tensor_to_object`` is a private PyTorch
        # API and fair game to add args to in future releases. With a
        # pinned signature, a torch upgrade that added a kwarg would
        # silently install at import time and then crash every DCP save
        # in the process with a ``TypeError``.
        if (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) and getattr(_in_dcp_op, "active", False):
            # Only fires inside ``_dcp_sync_scope``. Wait for any in-flight
            # collective ops to actually deliver bytes into ``tensor``
            # before we read them. On single-node NVLink builds this is
            # essentially a no-op; on multi-node RDMA builds this is the
            # load-bearing fence. Also gated on NPU because HCCL on
            # multi-node has structurally the same async-completion
            # semantics as NCCL — the receive buffer can be readable from
            # CPU before the device-side write is fully delivered — so
            # the same workaround applies. ``synchronize()`` is the
            # device-agnostic helper from ``veomni.utils.device``.
            synchronize()
        return _orig_tensor_to_object(*args, **kwargs)

    # Guard so re-importing the module doesn't stack the wrapper.
    if not getattr(_c10d._tensor_to_object, "_veomni_dcp_sync_patched", False):
        _veomni_tensor_to_object_with_sync._veomni_dcp_sync_patched = True
        _c10d._tensor_to_object = _veomni_tensor_to_object_with_sync
except (ImportError, AttributeError):  # pragma: no cover — defensive
    pass


# Companion patch for ``async_save``: ``torch.distributed.checkpoint.async_save``
# offloads the actual ``save`` call to a ``ThreadPoolExecutor`` worker
# (see ``_async_thread_executor._ThreadBasedAsyncCheckpointExecutor.execute_save``,
# which imports ``state_dict_saver.save`` at call time and ``submit``s
# it). The metadata broadcast — i.e. the ``_tensor_to_object`` call we
# care about — therefore runs on the WORKER thread, not on the main
# thread that entered ``_dcp_sync_scope``. Because the scope flag is
# thread-local, the worker would see ``_in_dcp_op.active`` as False and
# the sync workaround would be silently skipped, leaving async saves
# exposed to exactly the multi-node broadcast corruption this patch
# fixes. Wrap ``state_dict_saver.save`` itself so the scope is entered
# on whichever thread ends up running it — this also makes the sync-path
# explicit scope below redundant-but-safe (the scope is re-entrant).
try:
    import torch.distributed.checkpoint as _dcp_pkg  # noqa: E402
    import torch.distributed.checkpoint.state_dict_saver as _sds  # noqa: E402

    _orig_state_dict_saver_save = _sds.save

    def _veomni_state_dict_saver_save_with_scope(*args, **kwargs):
        with _dcp_sync_scope():
            return _orig_state_dict_saver_save(*args, **kwargs)

    # Guard so re-importing the module doesn't stack the wrapper.
    if not getattr(_sds.save, "_veomni_dcp_save_scoped", False):
        _veomni_state_dict_saver_save_with_scope._veomni_dcp_save_scoped = True
        _sds.save = _veomni_state_dict_saver_save_with_scope
        # ``torch.distributed.checkpoint.__init__.py`` does
        # ``from .state_dict_saver import save`` at import time, so
        # ``dcp.save`` is a separate binding from ``_sds.save`` after
        # that import has run. Patching only ``_sds.save`` misses every
        # direct ``dcp.save(...)`` caller (e.g.
        # ``save_lora_adapter_with_dcp`` in
        # ``veomni/utils/save_safetensor_utils.py``). Rebind the public
        # name too so the multi-node broadcast-corruption fix covers all
        # callers, not just the async path that dispatches through
        # ``state_dict_saver.save`` at call time.
        if getattr(_dcp_pkg, "save", None) is _orig_state_dict_saver_save:
            _dcp_pkg.save = _veomni_state_dict_saver_save_with_scope
except (ImportError, AttributeError):  # pragma: no cover — defensive
    pass


def _torch_save_with_no_progress_timeout(
    obj: Any,
    path: str,
    no_progress_timeout: Optional[float] = None,
    poll_interval: float = 10.0,
) -> None:
    """``torch.save(obj, path)`` guarded by a no-progress watchdog.

    HDFS-FUSE has a "soft hang" failure mode where the write syscall
    blocks indefinitely without raising — observed in production as a
    rank going silent until the 10-min NCCL Watchdog fires. The
    watchdog converts this into a ``TimeoutError`` so the calling code
    can surface a clear error and the cross-rank coordination layer
    can clean up uniformly.

    Raises ``TimeoutError`` only when ``path``'s size has not increased
    for ``no_progress_timeout`` seconds — distinguishing a stuck FUSE
    syscall from a legitimately slow but progressing write.
    """
    import queue as _queue
    import threading as _threading
    import time as _time

    if no_progress_timeout is None:
        no_progress_timeout = float(os.environ.get("VEOMNI_DCP_COPY_NO_PROGRESS_SEC", "180"))

    result_q: "_queue.Queue[tuple[str, Optional[BaseException]]]" = _queue.Queue()

    def _worker():
        try:
            torch.save(obj, path)
            result_q.put(("ok", None))
        except BaseException as exc:
            result_q.put(("err", exc))

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()

    last_size = -1
    last_progress_t = _time.monotonic()
    while True:
        t.join(poll_interval)
        if not t.is_alive():
            break
        try:
            cur_size = os.path.getsize(path)
        except OSError:
            cur_size = 0
        if cur_size > last_size:
            last_size = cur_size
            last_progress_t = _time.monotonic()
        elif _time.monotonic() - last_progress_t > no_progress_timeout:
            # Leaked thread is acceptable here — the FUSE syscall is
            # blocked in the kernel and can't be interrupted from Python
            # anyway.
            raise TimeoutError(
                f"torch.save({path!r}) made no progress for "
                f"{no_progress_timeout:.0f}s at size {cur_size} (HDFS FUSE soft hang)"
            )
    try:
        status, exc = result_q.get_nowait()
    except _queue.Empty:
        raise RuntimeError(f"torch.save({path!r}) worker thread exited without reporting a result") from None
    if status == "err":
        raise exc  # type: ignore[misc]


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

    def __init__(self, model, trainable_only: bool = False):
        self.model = model
        self.trainable_only = trainable_only

        # Determine whether this is ExtraParallel+FSDP2 case
        # If so, we need to restore Para(e.g. EP)-dim before saving to DCP
        self.parallel_state = get_parallel_state()
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
        extra_parallel_fqn2spec_info = self.extra_parallel_fqn2spec_info
        assert extra_parallel_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        extra_parallel_mesh = {
            para: self.parallel_state.extra_parallel_fsdp_device_mesh[para][para]
            if self.parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
            else None
            for para in self.parallel_state.extra_parallel_names
        }

        assert any(para_mesh is not None for para_mesh in extra_parallel_mesh.values()), (
            "At least one extra_parallel mesh should be not None"
        )

        global_extra_parallel_device_mesh = {
            para: self.parallel_state.extra_parallel_fsdp_device_mesh[para]
            for para in self.parallel_state.extra_parallel_names
        }
        assert any(
            global_para_device_mesh is not None and global_para_device_mesh.ndim == 2
            for global_para_device_mesh in global_extra_parallel_device_mesh.values()
        ), "At least one extra_parallel fsdp_device_mesh should be not None"

        assert action in ["restore", "drop"]

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in extra_parallel_fqn2spec_info and isinstance(
                extra_parallel_fqn2spec_info[name].placement, Shard
            ):
                cur_spec_info = extra_parallel_fqn2spec_info[name]
                assert cur_spec_info.para_fsdp_mesh is not None, (
                    f"ExtraParallel spec {name} must have either ExtraParallel FSDP mesh"
                )

                tensor = state_dict[name]

                if action == "drop":
                    tensor = drop_extra_parallel_dim(
                        tensor, cur_spec_info.para_fsdp_mesh[f"{cur_spec_info.para_name}_fsdp"]
                    )
                else:
                    tensor = restore_extra_parallel_dim(
                        tensor,
                        cur_spec_info.para_fsdp_mesh,
                        cur_spec_info.para_fsdp_mesh[f"{cur_spec_info.para_name}_fsdp"],
                    )
                state_dict[name] = tensor

        return state_dict


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
    cannot be scoped to optimizer-only).  Model-weight integrity is still
    enforced by ``set_model_state_dict(strict=True)`` inside
    ``ModelState.load_state_dict`` for non-LoRA loads.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.parallel_state = get_parallel_state()
        self.extra_parallel_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_extra_parallel_aware = (
            self.extra_parallel_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"
        )

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
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state_from_dcp_load,
        )

    def get_state_dict_with_extra_parallel_dim_preprocess(self, state_dict, action):
        extra_parallel_fqn2spec_info = self.extra_parallel_fqn2spec_info
        assert extra_parallel_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        extra_parallel_mesh = {
            para: self.parallel_state.extra_parallel_fsdp_device_mesh[para][para]
            if self.parallel_state.extra_parallel_fsdp_device_mesh[para] is not None
            else None
            for para in self.parallel_state.extra_parallel_names
        }

        assert any(para_mesh is not None for para_mesh in extra_parallel_mesh.values()), (
            "At least one extra_parallel mesh should be not None"
        )

        global_extra_parallel_device_mesh = {
            para: self.parallel_state.extra_parallel_fsdp_device_mesh[para]
            for para in self.parallel_state.extra_parallel_names
        }
        assert any(
            global_para_device_mesh is not None and global_para_device_mesh.ndim == 2
            for global_para_device_mesh in global_extra_parallel_device_mesh.values()
        ), "At least one extra_parallel fsdp_device_mesh should be not None"

        assert action in ["drop", "restore"]

        keys = list(state_dict.keys())
        extra_parallel_keys = list(extra_parallel_fqn2spec_info.keys())

        for name in sorted(keys):
            # Find ExtraParallel spec whose FQN appears in the state_dict key
            # e.g. name = "state.model.layers.0.mlp.experts.gate_proj.step"
            #      extra_parallel_key = "model.layers.0.mlp.experts.gate_proj"
            matches = [extra_parallel_key for extra_parallel_key in extra_parallel_keys if extra_parallel_key in name]
            if not matches:
                # ignore non-extra_parallel tensor
                continue

            # each tensor in the state dict should only belong to one ExtraParallel entry
            assert len(matches) == 1, f"Ambiguous ExtraParallel spec match for state key '{name}': {matches}"

            extra_parallel_key = matches[0]
            cur_spec_info = extra_parallel_fqn2spec_info[extra_parallel_key]

            # skip non-extra_parallel params which has Replicate placement in model spec info
            if not isinstance(cur_spec_info.placement, Shard):
                continue

            tensor = state_dict[name]
            if not torch.is_tensor(tensor):
                # we skip param-group hyperparams like `param_groups.model.layers.0.mlp.experts.down_proj.amsgrad`
                continue
            # Skip scalars (0-D tensors) – cannot be sharded on dim 0
            if tensor.ndim == 0:
                continue

            assert cur_spec_info.para_fsdp_mesh is not None, (
                f"ExtraParallel spec {name} must have either ExtraParallel FSDP mesh"
            )

            if action == "drop":
                tensor = drop_extra_parallel_dim(
                    tensor, cur_spec_info.para_fsdp_mesh[f"{cur_spec_info.para_name}_fsdp"]
                )
            elif action == "restore":
                tensor = restore_extra_parallel_dim(
                    tensor,
                    cur_spec_info.para_fsdp_mesh,
                    cur_spec_info.para_fsdp_mesh[f"{cur_spec_info.para_name}_fsdp"],
                )
            state_dict[name] = tensor

        return state_dict


def drop_extra_parallel_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """
    Drop ExtraParallel dims after loading from DCP so that ExtraParallel-FSDP would not be confused
    """

    if len(loaded_tensor.placements) == 2:
        tensor_to_put = DTensor.from_local(loaded_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(1)])
    elif len(loaded_tensor.placements) == 1:
        tensor_to_put = loaded_tensor.to_local()
    else:
        raise RuntimeError(
            f"Expect ExtraParallel paramters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (ExtraParallel+FSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def restore_extra_parallel_dim(
    orgin_tensor: torch.Tensor, fsdp_mesh: DeviceMesh, extra_parallel_fsdp_mesh: DeviceMesh
):
    """
    Restore ExtraParallel dim so that DCP can be aware about ExtraParallel ranks

    args:
        orgin_tensor (torch.Tensor): The orgin tensor.
        fsdp_mesh (DeviceMesh): The extra_parallel fsdp device mesh.
        shard (Shard): The shard info, default Shard(0).

    """
    assert fsdp_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {fsdp_mesh.ndim}"

    if isinstance(orgin_tensor, DTensor):
        # ExtraParallel+FSDP2
        dtensor = DTensor.from_local(
            orgin_tensor._local_tensor, device_mesh=fsdp_mesh, placements=[Shard(0), Shard(1)]
        )
    elif torch.is_tensor(orgin_tensor):
        # If there is no FSDP but only ExtraParallel
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=extra_parallel_fsdp_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {orgin_tensor} is not a tensor!")

    return dtensor


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    save_future: Optional[Any] = None
    # Checkpoint directory associated with ``save_future`` — used by the
    # cross-rank failure-coordination cleanup so we can ``rmtree`` the
    # actually-broken directory when the previous async save raised on
    # drain (instead of the *current* iteration's directory, which is
    # what a naive cleanup would target).
    _async_checkpoint_dir: Optional[str] = None
    # Dedicated process group for async saves and extra_state failure
    # coordination (created on first use, gloo backend preferred).
    _async_process_group: Optional[Any] = None
    # Sticky flag: if we tried to create the gloo group and it failed
    # (e.g. on HCCL/NPU deployments without gloo support), don't keep
    # retrying on subsequent saves — fall back to the default backend.
    _gloo_creation_failed: bool = False

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: int = None,
        storage_writer: Optional[FileSystemWriter] = None,
        trainable_only: bool = False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            save_async: whether to save asynchronously
            global_steps: global steps
            storage_writer: storage writer backend for dcp.save and dcp.async_save. If None, will use FileSystemWriter
            trainable_only: when True, only persist parameters with ``requires_grad=True``
                (LoRA / PEFT path). Frozen base weights are skipped on save and must be
                re-materialised from ``model.model_path`` at resume time. The optimizer
                state is already trainable-only by construction (the optimizer is built
                from ``filter(lambda p: p.requires_grad, ...)``), so this flag only
                affects the model state dump.
        return:
            None
        """
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        cls._create_checkpoint_dir(checkpoint_dir)

        # Eagerly create the gloo process group so the lazy creation
        # inside ``_coord_save_failures`` cannot interleave with a slow
        # rank's ``_save_extra_state`` retry chain. Without this, the
        # first save of a job would have the fast ranks blocking
        # silently inside ``dist.new_group(backend="gloo")`` (itself a
        # collective) for as long as the slowest rank's retry loop
        # takes — potentially minutes, with no log line explaining the
        # apparent hang. Same try/except + sticky-flag pattern as
        # ``_coord_save_failures``: gloo-unavailable deployments (some
        # HCCL/NPU envs) flip the flag and fall through to the default
        # backend later.
        if dist.is_initialized() and cls._async_process_group is None and not cls._gloo_creation_failed:
            try:
                cls._async_process_group = dist.new_group(backend="gloo")
            except Exception as e:
                cls._gloo_creation_failed = True
                logger.warning(
                    f"Could not create gloo process group at save() entry ({e}); "
                    f"_coord_save_failures will fall back to the default backend."
                )

        # Decide WHEN to drain the previous async-save future:
        #
        # - If the previous async save is writing into the SAME directory
        #   we're about to write into (rare — happens only when the caller
        #   reuses ``path`` / passes ``global_steps=None`` while the prior
        #   future is still in flight), drain FIRST. Otherwise
        #   ``_save_extra_state`` could write its file while the worker
        #   thread is still flushing model/optimizer shards into the same
        #   dir → mismatched checkpoint where extra_state is from
        #   iteration N+1 but model data is from iteration N.
        #
        # - Otherwise (typical case: ``global_step`` monotonically
        #   increases → unique dir per iteration), drain AFTER
        #   ``_save_extra_state`` so per-rank local I/O can overlap with
        #   the previous future for the common (no-failure) case.
        prev_async_exc: Optional[BaseException] = None
        prev_async_dir: Optional[str] = None
        drain_first = cls.save_future is not None and cls._async_checkpoint_dir == checkpoint_dir
        if drain_first:
            prev_async_exc, prev_async_dir = cls._drain_previous_async_save()

        # saving extra_state first to guarantee that every saved model/optimizer ckpts have their extra_state saved before them
        #
        # IMPORTANT: ``_save_extra_state`` is per-rank (no collective inside),
        # but ``execute_save`` below IS a collective. If ``_save_extra_state``
        # raises on only a subset of ranks (e.g. the no-progress watchdog
        # fires in the local→HDFS fallback for one rank, or direct retries
        # exhaust for another), those ranks would bail out before reaching
        # ``execute_save`` and the surviving ranks would hang on the NCCL
        # collective until the 10-min Watchdog fires — the exact deadlock
        # this patch is supposed to prevent.
        #
        # Strategy: catch the IO failure modes locally, then do a
        # cross-rank all_reduce on the failure flag. If ANY rank failed,
        # all ranks skip ``execute_save`` together (so no ``.metadata``
        # is published — ``dcp_get_last_iteration`` won't pick this dir
        # for auto-resume) and rank 0 best-effort cleans up the partial
        # directory. The failing rank's exception is then re-raised on
        # all ranks (surfaced on the failing one, synthesized on the
        # others so the caller sees a uniform "checkpoint not saved"
        # signal). If NO rank failed, all proceed to ``execute_save``
        # normally.
        extra_state_exc: Optional[BaseException] = None
        try:
            cls._save_extra_state(checkpoint_dir=checkpoint_dir, state=state)
        except Exception as e:
            # Catch broadly. ``torch.save`` underneath can raise
            # ``pickle.PicklingError``, ``TypeError`` (unpicklable object
            # in ``extra_state``: lambda in ``lr_scheduler``, unhashable
            # dataloader state, etc.), ``AttributeError`` (cloudpickle
            # quirk), ``MemoryError`` on top of the I/O families
            # (``TimeoutError`` / ``OSError`` / ``RuntimeError``). Any
            # uncaught raise here would let this rank bail out before
            # the cross-rank coordination, and peers would hang on the
            # next collective in ``execute_save`` for 10 min until the
            # NCCL Watchdog fires — the exact deadlock the rest of this
            # code is engineered to prevent. The downstream cleanup path
            # handles every kind of failure uniformly, so a broad catch
            # is safe.
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.error(
                f"[rank {rank}] _save_extra_state raised {type(e).__name__}: {e}. "
                f"Coordinating with peer ranks; if any rank failed, dcp.save will be skipped "
                f"and the partial checkpoint dir will be cleaned up to avoid silent corruption."
            )
            extra_state_exc = e

        # Drain any previous async-save future (deferred case) BEFORE the
        # coordination ``all_reduce`` — which shares ``cls._async_process_group``
        # with the in-flight ``dcp.async_save`` worker thread. Two threads
        # issuing overlapping collectives on one PG can deadlock or corrupt
        # the next save. If we already drained above (same-dir case), this
        # is a no-op via the ``cls.save_future is None`` guard.
        #
        # ``_drain_previous_async_save`` does NOT raise on per-rank
        # failures — it captures the local exception and returns it,
        # along with the directory that was being written by the failed
        # future (so the cleanup below targets the right dir).
        if not drain_first:
            prev_async_exc, prev_async_dir = cls._drain_previous_async_save()

        any_es_failed, any_prev_failed = cls._coord_save_failures(
            local_extra_state_failure=extra_state_exc is not None,
            local_prev_async_failure=prev_async_exc is not None,
        )
        any_failed = any_es_failed or any_prev_failed

        if any_failed:
            # Skip the dcp.save collective entirely → no ``.metadata`` lands
            # on disk for the current iteration → ``dcp_get_last_iteration``
            # cannot mis-pick it as the latest checkpoint. Rank 0 best-
            # effort cleans up partial files.
            #
            # We may need to clean up TWO different directories:
            #
            # 1. ``prev_async_dir`` — the previous iteration's async-save
            #    directory, which the failed future was writing into. Only
            #    cleaned when ``any_prev_failed`` (the previous save
            #    actually failed). If only the current extra_state failed,
            #    the previous async-save dir is a *valid* checkpoint and
            #    must be preserved.
            # 2. ``checkpoint_dir`` — the current iteration's directory.
            #    Always cleaned when ``any_failed`` (we skip dcp.save, so
            #    it has at most some extra_state files and no .metadata).
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                import shutil

                dirs_to_clean = []
                if any_prev_failed and prev_async_dir:
                    dirs_to_clean.append(("previous async-save dir", prev_async_dir))
                dirs_to_clean.append(("current iteration dir", checkpoint_dir))
                for label, dir_path in dirs_to_clean:
                    try:
                        shutil.rmtree(dir_path, ignore_errors=True)
                        logger.error(
                            f"Removed partial checkpoint dir {dir_path} ({label}) after save failure; "
                            f"dcp.save was NOT run for the current iteration."
                        )
                    except OSError as cleanup_exc:
                        logger.warning(f"Could not clean up {dir_path}: {cleanup_exc}")
            if dist.is_initialized():
                dist.barrier(group=cls._async_process_group)
            # Surface the earliest-causally-relevant local exception. If
            # this rank itself failed (either previous async drain OR
            # current extra_state), raise that. Otherwise synthesize a
            # generic peer-failure error.
            if prev_async_exc is not None:
                raise prev_async_exc
            if extra_state_exc is not None:
                raise extra_state_exc
            raise RuntimeError(
                "Save failed on a peer rank; skipped dcp.save and removed "
                "partial checkpoint dir(s) to avoid silent corruption."
            )

        save_state = {"model": ModelState(state["model"], trainable_only=trainable_only)}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])

        if storage_writer is None:
            storage_writer = cls._create_storage_writer(checkpoint_dir)

        cls.execute_save(save_state=save_state, storage_writer=storage_writer, save_async=save_async)

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
        trainable_only: bool = False,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            process_group: process group for loading checkpoint
            storage_reader: storage reader backend for dcp.load. If None, will use FileSystemReader
            trainable_only: when True, ``set_model_state_dict`` runs in non-strict
                mode (``StateDictOptions(strict=False)``). Use this for LoRA / PEFT
                resumes where the DCP only contains trainable adapter weights and the
                frozen base must come from ``model.model_path``. Safe to enable when
                the DCP is full (extra strictness is just dropped).

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        load_state = {"model": ModelState(state["model"], trainable_only=trainable_only)}
        if "optimizer" in state:
            load_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if storage_reader is None:
            storage_reader = cls._create_storage_reader(checkpoint_dir)

        # DCP load also uses broadcast_object_list internally to share the
        # Metadata; gate the sync workaround the same way as save.
        with _dcp_sync_scope():
            dcp.load(
                state_dict=load_state,
                storage_reader=storage_reader,
                process_group=process_group,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )

        cls._load_extra_state(checkpoint_dir=checkpoint_dir, state=state)

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state

    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
    ) -> None:
        """Execute DCP save with optional async support."""
        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP
            # saves. ``dcp.async_save`` REQUIRES a CPU backend in the
            # process group (it asserts ``torch.device("cpu") in pg._device_types``
            # internally), so unlike the soft-fallback in
            # ``_coord_save_failures`` we cannot silently degrade to the
            # default backend here — async save just doesn't work without
            # gloo. ``_gloo_creation_failed`` is the same sticky flag that
            # ``_coord_save_failures`` sets; honor it so we don't retry a
            # collective ``new_group`` call that already failed once.
            if cls._async_process_group is None:
                if cls._gloo_creation_failed:
                    raise RuntimeError(
                        "dcp.async_save requires a gloo process group, but a prior "
                        "dist.new_group(backend='gloo') already failed in this process. "
                        "Use save_async=False, or initialize the default process group with "
                        "a CPU backend (e.g. 'cpu:gloo,cuda:nccl')."
                    )
                try:
                    cls._async_process_group = dist.new_group(backend="gloo")
                except Exception as e:
                    cls._gloo_creation_failed = True
                    raise RuntimeError(
                        f"dcp.async_save requires a gloo process group, but "
                        f"dist.new_group(backend='gloo') failed: {e}. "
                        f"Use save_async=False, or initialize the default process group with "
                        f"a CPU backend (e.g. 'cpu:gloo,cuda:nccl')."
                    ) from e

            # ``save()`` already drained-and-coordinated at its entry, so
            # for the common path this is a no-op via the ``save_future is
            # None`` guard inside the helper. The helper variant is used
            # (rather than the bare drain) so that direct ``execute_save``
            # callers — i.e. anyone bypassing ``save()`` — also get the
            # coordinated raise-uniformly-or-not-at-all semantics; without
            # it a stranded previous-future failure would be silently
            # swallowed here for those callers.
            cls.drain_and_coordinate_previous_async_save()

            # Scope the _tensor_to_object sync workaround to just this DCP
            # call. async_save's metadata broadcast happens synchronously
            # during the in-thread stage phase (before the future is
            # returned); the background file-write thread doesn't broadcast,
            # so thread-local scoping is sufficient.
            with _dcp_sync_scope():
                cls.save_future = dcp.async_save(
                    state_dict=save_state,
                    storage_writer=storage_writer,
                    process_group=cls._async_process_group,
                )
            # Track the dir associated with this in-flight future so the
            # next save's failure-coordination cleanup can ``rmtree`` it
            # (rather than the wrong / current iteration's dir) if the
            # future ends up raising on drain. Pulled off ``storage_writer``
            # so this works for direct ``execute_save`` callers too —
            # ``FileSystemWriter.path`` is the public attribute.
            cls._async_checkpoint_dir = str(getattr(storage_writer, "path", None) or "")
            if not cls._async_checkpoint_dir:
                cls._async_checkpoint_dir = None  # type: ignore[assignment]
        else:
            with _dcp_sync_scope():
                dcp.save(
                    state_dict=save_state,
                    storage_writer=storage_writer,
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
    def _drain_previous_async_save(cls) -> tuple:
        """Block until any in-flight ``dcp.async_save`` future completes,
        returning ``(local_exc, prev_dir)``.

        - ``local_exc`` is the exception raised by ``save_future.result()``
          on this rank, or ``None`` if it succeeded.
        - ``prev_dir`` is the checkpoint directory that was being written
          by the in-flight future (captured at submission time in
          ``execute_save`` from ``storage_writer.path``), or ``None`` if
          there was no future. The caller uses this to ``rmtree`` the
          actually-broken directory in the failure-coordination cleanup;
          using the current iteration's ``checkpoint_dir`` instead (what
          the previous version of this code did) would leave the broken
          previous-iteration directory on disk for ``dcp_get_last_iteration``
          to mis-pick on the next auto-resume.

        Idempotent: if ``cls.save_future`` is ``None`` returns
        ``(None, None)``.

        IMPORTANT — does not raise: the previous async save can raise on
        only a subset of ranks (e.g. writer-thread I/O failure on one
        host), and propagating that here would let those ranks bail out
        before reaching the subsequent ``_coord_save_failures`` all_reduce
        while the survivors blocked there forever. Instead we capture the
        local exception and let the caller coordinate cross-rank.

        Critical for correctness: ``_coord_save_failures``' all_reduce
        shares ``cls._async_process_group`` with the ``dcp.async_save``
        worker thread; without this drain the two threads could issue
        overlapping collectives on one PG and deadlock or corrupt the
        save. The drain itself no longer issues a ``barrier`` — the
        subsequent all_reduce serves as the cross-rank synchronization
        point.
        """
        if cls.save_future is None:
            return None, None
        if dist.is_initialized():
            logger.info(f"[RANK {dist.get_rank()}] waiting for previous DCP saving session to end...")
        local_exc: Optional[BaseException] = None
        try:
            cls.save_future.result()
        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.error(
                f"[rank {rank}] previous dcp.async_save future raised on drain: "
                f"{type(e).__name__}: {e}. Will coordinate with peer ranks; the current "
                f"save will be aborted."
            )
            local_exc = e
        prev_dir = cls._async_checkpoint_dir
        cls.save_future = None
        cls._async_checkpoint_dir = None
        return local_exc, prev_dir

    @classmethod
    def drain_and_coordinate_previous_async_save(cls) -> None:
        """Public coordinated drain for callers outside of ``save()``.

        Use this anywhere code would previously have done a raw
        ``cls.save_future.result()`` (e.g. ``CheckpointerCallback._load_checkpoint``
        before ``load()``, the HF safetensor export, etc.). It captures
        the local exception via ``_drain_previous_async_save``, coordinates
        cross-rank via ``_coord_save_failures``, best-effort ``rmtree``s
        the failed directory on rank 0, and raises a uniform error on all
        ranks if any rank's previous save failed.

        Why this matters: a raw ``.result()`` re-raises the worker
        thread's exception synchronously, but distributed save failures
        are typically asymmetric (one rank's HDFS write fails, others
        succeed). Without coordination the failing rank exits the
        function while peers continue into the next collective (``load``,
        ``barrier``, HF export's collective save), and the whole job
        deadlocks. The coordinated wrapper guarantees that either all
        ranks raise or none do.

        No-op if there's no in-flight async save.
        """
        prev_exc, prev_dir = cls._drain_previous_async_save()
        _, any_prev_failed = cls._coord_save_failures(
            local_extra_state_failure=False,
            local_prev_async_failure=prev_exc is not None,
        )
        if not any_prev_failed:
            return
        # Best-effort cleanup of the broken previous-iteration dir on rank 0.
        if prev_dir and ((not dist.is_initialized()) or dist.get_rank() == 0):
            import shutil

            try:
                shutil.rmtree(prev_dir, ignore_errors=True)
                logger.error(f"Removed broken previous async-save dir {prev_dir} after a peer-rank failure.")
            except OSError as cleanup_exc:
                logger.warning(f"Could not clean up {prev_dir}: {cleanup_exc}")
        if dist.is_initialized():
            dist.barrier(group=cls._async_process_group)
        if prev_exc is not None:
            raise prev_exc
        raise RuntimeError("Previous dcp.async_save failed on a peer rank.")

    @classmethod
    def _coord_save_failures(cls, local_extra_state_failure: bool, local_prev_async_failure: bool) -> tuple:
        """All-reduce two local failure flags across ranks in a single
        collective.

        Returns ``(any_extra_state_failed, any_prev_async_failed)``.
        Distinguishing the two failure kinds is necessary because the
        cleanup logic differs: ``any_prev_async_failed`` triggers
        ``rmtree`` of the *previous iteration's* dir; ``any_extra_state_failed``
        only triggers ``rmtree`` of the *current* dir.

        Tries to use a gloo process group (lazily created here) instead
        of NCCL/HCCL so this collective is not subject to the NCCL/HCCL
        Watchdog timeout — it can follow a long-running
        ``_save_extra_state`` retry chain that may itself approach or
        exceed the default 10-min Watchdog.

        On deployments where gloo is not available (notably some HCCL/NPU
        environments — see the codex review #7 finding), we silently fall
        back to the default process group. The all_reduce still works,
        just under the default backend's watchdog policy. Lazy creation
        is safe because EVERY rank goes through this helper on every
        save (regardless of which ranks failed), so ``dist.new_group``
        is called in lockstep.

        Single-process / dist-not-initialized path just returns the local
        flags unchanged.
        """
        if not dist.is_initialized():
            return local_extra_state_failure, local_prev_async_failure
        if cls._async_process_group is None and not cls._gloo_creation_failed:
            try:
                cls._async_process_group = dist.new_group(backend="gloo")
            except Exception as e:
                # Gloo backend not available — fall through to default group.
                # Cached so subsequent saves don't retry the new_group call
                # (which would itself be a redundant collective).
                cls._gloo_creation_failed = True
                logger.warning(
                    f"Could not create gloo process group for save failure coordination ({e}); "
                    f"falling back to the default backend. NCCL/HCCL Watchdog may fire on this "
                    f"all_reduce if _save_extra_state took an unusually long time on some ranks."
                )
        # ``cls._async_process_group`` is either a gloo group, or None (use default).
        # Gloo all_reduce wants a CPU tensor; NCCL needs CUDA; HCCL needs NPU.
        # Pick the device by which path we're on so the fallback (default
        # PG = NCCL/HCCL on accelerator deployments) doesn't crash with a
        # backend/device mismatch — that would defeat the whole point of
        # the fallback (which exists specifically for HCCL/NPU envs).
        if cls._async_process_group is None and (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
            flag_device = torch.device(f"{get_device_type()}:{get_device_id()}")
        else:
            flag_device = torch.device("cpu")
        # Single tensor, two slots → one collective, both kinds decoded.
        flag = torch.tensor(
            [1 if local_extra_state_failure else 0, 1 if local_prev_async_failure else 0],
            dtype=torch.long,
            device=flag_device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.SUM, group=cls._async_process_group)
        return flag[0].item() > 0, flag[1].item() > 0

    @classmethod
    def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Save extra_state to checkpoint directory.

        Direct ``torch.save`` to the HDFS path with retry-and-backoff plus
        a no-progress watchdog. The watchdog (``_torch_save_with_no_progress_timeout``)
        converts an HDFS-FUSE soft-hang (write syscall blocks forever
        without raising) into a ``TimeoutError`` so the rank doesn't
        wedge silently until the 10-min NCCL Watchdog fires.

        Recovery strategy:

        1. Try the watchdog-wrapped ``torch.save`` directly to the HDFS
           path. On success: return.
        2. On ``TimeoutError`` (soft hang): propagate immediately with
           NO cleanup and NO retry — the leaked worker thread still
           owns the path and any retry would race with it. The outer
           ``save()`` catches it, coordinates cross-rank via
           ``_coord_save_failures``, and all ranks raise uniformly.
        3. On ``RuntimeError`` / ``OSError`` (zip-writer enforce-fail,
           errno-5/75/107, etc.): clean up the half-written file,
           sleep with exponential backoff (1s, 2s), retry up to
           ``_DIRECT_RETRIES`` total attempts. On exhaustion: re-raise
           the last error. ``save()``'s cross-rank coordination handles
           the rest (any-rank failure → all ranks skip ``dcp.save`` and
           clean up the partial dir).

        The trainer's next checkpoint iteration provides the recovery
        path: a single FUSE flake fails this save, but training
        continues and the next save typically succeeds.
        """
        if "extra_state" not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state save")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(rank))

        _DIRECT_RETRIES = 3  # 1 initial + 2 retries
        _DIRECT_BACKOFF_BASE_SEC = 1.0  # 1s, 2s
        import time

        last_exc: Optional[BaseException] = None

        for attempt in range(_DIRECT_RETRIES):
            try:
                # Watchdog wrap: soft hang surfaces as TimeoutError
                # rather than wedging the rank for the NCCL Watchdog
                # window.
                _torch_save_with_no_progress_timeout(state["extra_state"], extra_state_path)
                if attempt > 0:
                    logger.info(
                        f"[rank {rank}] _save_extra_state: torch.save succeeded on retry "
                        f"attempt {attempt + 1}/{_DIRECT_RETRIES} after earlier error ({last_exc!r})"
                    )
                return
            except TimeoutError as e:
                # Worker thread is still blocked in the FUSE syscall
                # and STILL OWNS ``extra_state_path``. Retry would race
                # with the leaked thread. Propagate immediately; outer
                # save() coordinates cross-rank.
                logger.error(
                    f"[rank {rank}] _save_extra_state: torch.save soft-hang on attempt "
                    f"{attempt + 1}/{_DIRECT_RETRIES} for {extra_state_path}: {e}. "
                    f"NOT retrying — leaked write thread still holds the destination."
                )
                raise
            except (RuntimeError, OSError) as e:
                last_exc = e
                is_last = attempt == _DIRECT_RETRIES - 1
                backoff = _DIRECT_BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    f"[rank {rank}] _save_extra_state: torch.save attempt "
                    f"{attempt + 1}/{_DIRECT_RETRIES} failed with {type(e).__name__}: {e}."
                    + ("" if is_last else f" Cleaning up partial file and sleeping {backoff:.1f}s before retry.")
                )
                # Safe to clean up: torch.save raised synchronously (not
                # via the watchdog), so no leaked thread owns the path.
                try:
                    if os.path.exists(extra_state_path):
                        os.remove(extra_state_path)
                except OSError as cleanup_exc:
                    logger.warning(
                        f"[rank {rank}] _save_extra_state: cleanup of {extra_state_path} "
                        f"failed: {cleanup_exc}; subsequent retry may inherit junk bytes."
                    )
                if not is_last:
                    time.sleep(backoff)

        # All direct retries exhausted — re-raise. save()'s cross-rank
        # coordination takes it from here.
        raise last_exc if last_exc is not None else RuntimeError("_save_extra_state exhausted")

    @classmethod
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        """Load extra_state from checkpoint directory.

        Hardened against HDFS-FUSE flakiness and against asymmetric raise
        across ranks. A bare ``torch.load`` failure on a single rank's
        HDFS-FUSE path would let that rank bail out while peer ranks
        proceed past this call into the next collective (e.g. the next
        ``dist.barrier`` inside the load callback) and hang on the NCCL
        Watchdog — the same shape of deadlock the save path is engineered
        to avoid.

        Retry policy (mirrors ``_save_extra_state``'s direct retry):
        - Try ``torch.load`` up to ``_DIRECT_RETRIES`` times. Errno-5 /
          errno-75 / errno-107 / zip-writer enforce-fails seen on save
          can also intermittently affect read.
        - On exhaustion: capture the local exception, coordinate across
          ranks via ``_coord_save_failures``, raise uniformly.
        """
        if "extra_state" not in state:
            logger.warning_rank0("extra_state not found in state, skipping extra_state load")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(rank))

        _DIRECT_RETRIES = 3  # 1 initial + 2 retries
        _DIRECT_BACKOFF_BASE_SEC = 1.0
        import time

        last_exc: Optional[BaseException] = None
        loaded = False
        for attempt in range(_DIRECT_RETRIES):
            try:
                state["extra_state"] = torch.load(extra_state_path, weights_only=False)
                if attempt > 0:
                    logger.info(
                        f"[rank {rank}] _load_extra_state: torch.load succeeded on retry "
                        f"attempt {attempt + 1}/{_DIRECT_RETRIES} after earlier error ({last_exc!r})"
                    )
                loaded = True
                break
            except Exception as e:
                # Broad catch for the same reasons documented in save()'s
                # ``_save_extra_state`` wrapper — any uncaught raise here
                # would be asymmetric across ranks and hang the next
                # collective.
                last_exc = e
                is_last = attempt == _DIRECT_RETRIES - 1
                backoff = _DIRECT_BACKOFF_BASE_SEC * (2**attempt)
                logger.warning(
                    f"[rank {rank}] _load_extra_state: torch.load attempt "
                    f"{attempt + 1}/{_DIRECT_RETRIES} failed with {type(e).__name__}: {e}."
                    + ("" if is_last else f" Sleeping {backoff:.1f}s before retry to let the FUSE mount settle.")
                )
                if not is_last:
                    time.sleep(backoff)

        # Coordinate cross-rank. Reuse ``_coord_save_failures``'s
        # "prev_async_failure" slot to carry the load failure flag — the
        # semantics (any-rank-failed → all raise) are the same; only the
        # name is save-flavored. Keep the local-extra-state-failure flag
        # at False since that's a save-only concept.
        _, any_failed = cls._coord_save_failures(
            local_extra_state_failure=False,
            local_prev_async_failure=not loaded,
        )
        if any_failed:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                f"_load_extra_state failed on a peer rank (rank {rank} succeeded). "
                f"Checkpoint at {checkpoint_dir} is unloadable."
            )


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return size in bytes for a given dtype."""
    size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return size_map.get(dtype, 4)


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert DCP key to HuggingFace format. Returns None for non-model weights.

    Conversion rules:
    - "model.model.*" -> "model.*" (remove first "model." prefix)
    - "model.lm_head.weight" -> "lm_head.weight" (special case)
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
            if save_dtype:
                dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
            else:
                if not hasattr(tensor_meta.properties, "dtype"):
                    raise ValueError(
                        f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                    )
                dtype = tensor_meta.properties.dtype

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

        if target_dtype:
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
