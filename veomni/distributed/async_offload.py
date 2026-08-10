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


# Adapted from MindSpeed-MM's async_offload.py.
# Key adaptations for VeOmni:
#   - Uses VeOmni's device-agnostic stream/event helpers (veomni.utils.device)
#   - Uses VeOmni's logger (veomni.utils.logging) instead of print_rank
#   - Uses segment-aware module-name matching instead of mindspeed's str_match
#   - Supports wildcard pattern ``{*}`` for sequential module groups (e.g. model.layers.{*})

import fnmatch
from typing import List
from weakref import WeakValueDictionary

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..utils import logging
from ..utils.device import create_event, create_stream, get_current_stream, switch_to_specified_stream


logger = logging.get_logger(__name__)


def _module_name_match(pattern: str, name: str) -> bool:
    """Match module paths without allowing wildcards to cross ``.`` separators."""
    pattern_segments = pattern.split(".")
    name_segments = name.split(".")
    return len(pattern_segments) == len(name_segments) and all(
        fnmatch.fnmatchcase(name_segment, pattern_segment)
        for pattern_segment, name_segment in zip(pattern_segments, name_segments)
    )


def base_check_fn(tensor) -> bool:
    if tensor.device.type not in ("cuda", "npu"):
        return False
    if isinstance(tensor, torch.nn.parameter.Parameter) or tensor._base is not None:
        return False
    storage_nbytes = tensor.untyped_storage().nbytes()
    if storage_nbytes <= 0 or storage_nbytes != tensor.numel() * tensor.element_size() or not tensor.is_contiguous():
        return False
    return True


class PinnedBufferPool:
    """Reuse host buffers so async offload does not allocate pinned memory per tensor."""

    def __init__(self):
        self._free_buffers = {}
        self.allocations = 0
        self.reuses = 0

    @staticmethod
    def _key(tensor):
        # Strides are part of the key even though the current offload
        # predicate accepts contiguous tensors only; this keeps the pool safe
        # if another dense layout is supported later.
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)

    def acquire(self, tensor):
        key = self._key(tensor)
        free_buffers = self._free_buffers.get(key)
        if free_buffers:
            self.reuses += 1
            return free_buffers.pop(), key

        pin_memory = tensor.device.type in ("cuda", "npu")
        try:
            buffer = torch.empty_strided(
                tensor.shape,
                tensor.stride(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
        except RuntimeError:
            if not pin_memory:
                raise
            logger.warning_rank0(
                "Pinned host allocation for async activation offload failed; "
                "falling back to pageable host memory."
            )
            buffer = torch.empty_strided(
                tensor.shape,
                tensor.stride(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=False,
            )
        self.allocations += 1
        return buffer, key

    def release(self, buffer, key):
        if buffer is not None:
            self._free_buffers.setdefault(key, []).append(buffer)


class GetCnt:
    def __init__(self):
        self._block_idx = -1
        self._block_tensor_nums = {}

    def get_cnt(self, block_idx):
        after_block = False
        if block_idx > self._block_idx:
            if block_idx in self._block_tensor_nums:
                self._block_tensor_nums[block_idx] += 1
            else:
                self._block_tensor_nums[block_idx] = 1
            if self._block_idx >= 0:
                after_block = True
            self._block_idx = block_idx
        elif block_idx == self._block_idx:
            self._block_tensor_nums[block_idx] += 1
        else:
            if block_idx in self._block_tensor_nums:
                self._block_tensor_nums[block_idx] += 1
            else:
                self._block_tensor_nums[block_idx] = 1
            self._block_idx = block_idx

        offload_tensor_key = "{}_{}".format(self._block_idx, self._block_tensor_nums[self._block_idx] - 1)
        return offload_tensor_key, after_block

    def reset(self):
        self._block_idx = -1
        self._block_tensor_nums = {}

    def get_prefetch_keys(self, block_idx):
        prefetch_block_idx = max((idx for idx in self._block_tensor_nums.keys() if idx < block_idx), default=None)

        if prefetch_block_idx is None:
            return []

        block_tensor_nums = self._block_tensor_nums[prefetch_block_idx]
        prefetch_idxs = list(range(0, block_tensor_nums))
        return ["{}_{}".format(prefetch_block_idx, prefetch_idx) for prefetch_idx in prefetch_idxs]

    def get_layer_tensor_nums(self, block_idx):
        return self._block_tensor_nums[block_idx]


class SwapTensor:
    def __init__(self, tensor, key, pool):
        if (
            tensor._base is not None
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            or not tensor.is_contiguous()
        ):
            raise ValueError("Async activation offload requires a tensor with private, dense storage.")

        self.tensor = tensor
        self.size = tensor.size()
        self.storage_nbytes = tensor.untyped_storage().nbytes()
        self.pool = pool
        self.tensor_cpu, self._host_buffer_key = pool.acquire(tensor)
        self._host_buffer_released = False

        self.stat = "device"
        self.key = key

        self.d2h_event = create_event()
        self.h2d_event = create_event()

    def release_host_buffer(self):
        if self._host_buffer_released or self.tensor_cpu is None:
            return
        if self.stat == "d2h":
            self.d2h_event.synchronize()
        elif self.stat == "host":
            self.d2h_event.synchronize()
        elif self.stat == "h2d":
            self.h2d_event.synchronize()
        self.pool.release(self.tensor_cpu, self._host_buffer_key)
        self.tensor_cpu = None
        self._host_buffer_released = True

    def __del__(self):
        try:
            self.release_host_buffer()
        except Exception:
            pass

    def launch_d2h(self, stream):
        if self.stat != "device":
            return

        forward_event = create_event()
        forward_event.record()
        with torch.no_grad():
            with switch_to_specified_stream(stream):
                stream.wait_event(forward_event)
                self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                self.d2h_event.record()
                self.stat = "d2h"

    def wait_d2h_finished(self):
        if self.stat != "d2h":
            return
        get_current_stream().wait_event(self.d2h_event)
        self.tensor.untyped_storage().resize_(0)
        self.stat = "host"

    def launch_h2d(self, h2d_stream):
        if self.stat == "h2d":
            return
        if self.stat != "host":
            return
        backward_event = create_event()
        backward_event.record()

        with torch.no_grad():
            with switch_to_specified_stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                self.tensor.untyped_storage().resize_(self.storage_nbytes)
                self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                self.h2d_event.record()
                self.stat = "h2d"


class OffloadManager:
    def __init__(self):
        # Autograd owns packed SwapTensor objects until backward. Weak references
        # let abandoned graphs release stale entries after a failed step.
        self.items = WeakValueDictionary()
        self.getcnt = GetCnt()
        self.swap_stream = create_stream()

        self.host_buffer_pool = PinnedBufferPool()

    def get_cnt(self, block_idx):
        return self.getcnt.get_cnt(block_idx)

    def reset_getcnt(self):
        self.getcnt.reset()

    def reset(self):
        self.items.clear()
        self.getcnt.reset()

    def assert_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} does not exist in items")

    def exist(self, key):
        return key in self.items

    def put(self, key, act):
        if key in self.items:
            raise RuntimeError(f"Key {key} already exists in items")
        self.items[key] = act

    def wait_d2h_finished(self, prefix_key):
        for key in list(self.items.keys()):
            if key.startswith(prefix_key):
                self.items[key].wait_d2h_finished()

    def get_layer_items_keys(self, block_idx):
        block_tensor_nums = self.getcnt.get_layer_tensor_nums(block_idx)
        layer_items_keys = []

        for tensor_id in range(block_tensor_nums):
            key = f"{block_idx}_{tensor_id}"
            if key in self.items.keys():
                layer_items_keys.append(key)
        return layer_items_keys

    def get(self, key):
        self.assert_exist(key)
        return self.items[key]

    def prefetch_get(self, block_idx, h2d_stream, d2h_stream):
        prefetch_keys = self.getcnt.get_prefetch_keys(block_idx)
        for prefetch_key in prefetch_keys:
            if self.exist(prefetch_key):
                prefetch_swap_tensor = self.get(prefetch_key)
                d2h_stream.wait_stream(h2d_stream)
                prefetch_swap_tensor.launch_h2d(h2d_stream)

    def clear(self, key=None):
        if key is None:
            self.items.clear()
        else:
            self.assert_exist(key)
            self.items.pop(key)


def reset_async_activation_offload(model):
    """Reset all async-offload managers attached to ``model``.

    The saved-tensor graph normally releases ``SwapTensor`` objects after
    backward.  A failed loss/backward step can leave the graph alive, though,
    so the next training step must explicitly discard its keys and counters.
    The host pool remains attached to the model and can still reuse buffers
    once the old graph is collected.
    """
    managers = {}
    for module in model.modules():
        manager = getattr(module, "_veomni_offload_manager", None)
        if manager is not None:
            managers[id(manager)] = manager

    for manager in managers.values():
        manager.reset()


class async_save_on_cpu(saved_tensors_hooks):
    def __init__(
        self,
        block_idx,
        depth,
        manager,
        custom_check_fn=None,
        prefetch=True,
    ) -> None:
        def _pack_to_cpu(tensor):
            if not base_check_fn(tensor):
                return tensor

            if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                return tensor

            key, after_block = manager.get_cnt(block_idx)
            d2h_stream = manager.swap_stream

            if after_block:
                manager.wait_d2h_finished("{}_".format(block_idx - 1))

            if block_idx == depth - 1:
                return tensor

            swap_tensor = SwapTensor(tensor, key, manager.host_buffer_pool)

            if block_idx < depth - 1:
                swap_tensor.launch_d2h(d2h_stream)

            manager.put(key, swap_tensor)
            return swap_tensor

        def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
            if isinstance(swap_tensor, torch.Tensor):
                return swap_tensor

            d2h_stream = manager.swap_stream
            h2d_stream = manager.swap_stream
            swap_tensor.launch_h2d(h2d_stream)

            get_current_stream().wait_event(swap_tensor.h2d_event)
            swap_tensor.release_host_buffer()
            swap_tensor.stat = "device"

            block_idx, tensor_idx = swap_tensor.key.split("_")
            manager.clear(swap_tensor.key)

            if prefetch:
                manager.prefetch_get(int(block_idx), h2d_stream, d2h_stream)
            return swap_tensor.tensor

        super().__init__(_pack_to_cpu, _unpack_from_cpu)


def get_offload_modules(model, plan: List[str]):
    matched_submodules = []
    offload_layers = 0
    for plan_name in plan:
        if "{*}" in plan_name:
            prefix = plan_name.split("{*}")[0].rstrip(".")
            parent_module = model
            try:
                for sub_name in prefix.split("."):
                    parent_module = getattr(parent_module, sub_name)
                if parent_module is None:
                    continue
                depth = len(parent_module)
                for layer_idx, module in enumerate(parent_module):
                    full_module_name = f"{prefix}.{layer_idx}"
                    matched_submodules.append([full_module_name, module, offload_layers, depth])
                    offload_layers += 1
            except AttributeError as e:
                logger.info_rank0(f"Skip plan {plan_name}: {e}")
        else:
            depth = 1
            for name, module in model.named_modules():
                if _module_name_match(plan_name, name):
                    if not any(item[0] == name for item in matched_submodules):
                        matched_submodules.append([name, module, offload_layers, depth])
                        offload_layers += 1

    for matched_submodule in matched_submodules:
        matched_submodule[-1] = offload_layers

    return matched_submodules


def _get_no_split_offload_modules(model):
    target_classes = set(getattr(model, "_no_split_modules", None) or [])
    if not target_classes:
        raise ValueError(
            "Async activation offload auto-discovery requires model._no_split_modules when "
            "activation_offload_modules is empty."
        )

    matched_submodules = [
        [name, module, layer_idx, 0]
        for layer_idx, (name, module) in enumerate(
            (item for item in model.named_modules() if type(item[1]).__name__ in target_classes)
        )
    ]
    if not matched_submodules:
        raise ValueError(
            "Async activation offload did not match any modules listed in model._no_split_modules: "
            f"{sorted(target_classes)!r}"
        )

    depth = len(matched_submodules)
    for matched_submodule in matched_submodules:
        matched_submodule[-1] = depth
    return matched_submodules


def async_offload_modules(modules, prefetch=True, hidden_states_idx=0):
    """Apply async activation offload via selected-instance ``__call__`` patching.

    For each module, per-instance attributes (``_veomni_offload_layer_idx``,
    ``_veomni_offload_depth``, ``_veomni_offload_hidden_states_idx``,
    ``_veomni_offload_prefetch``) are set, then the instance is rebound to a
    private subclass whose ``__call__`` wraps it with ``async_save_on_cpu``.

    A per-instance subclass is required because Python's dunder-method dispatch
    bypasses instance-level assignments. It avoids mutating the original class,
    so unselected modules and other models in the process remain unchanged.

    More importantly, class-level ``__call__`` patching places the
    ``async_save_on_cpu`` context **outside** the ``GradientCheckpointingLayer``
    checkpoint boundary.  This matches MindSpeed-MM's behavior where
    ``recompute_wrapper`` is applied first and ``with_async_save_on_cpu``
    wraps it second — so the ``saved_tensors_hooks`` stack order during
    forward is:

        [async_save_on_cpu (outer), _checkpoint_hook (inner)]

    With this order, ``_NoopSaveInputs.apply`` (called by checkpoint BEFORE
    ``_checkpoint_hook`` is pushed) saves input tensors through
    ``async_save_on_cpu.pack``, allowing hidden_states to be offloaded to CPU.
    Intermediate activations are then handled by ``_checkpoint_hook`` (GC
    recomputation), requiring zero persistent GPU memory.

    **Key uniqueness across multiple forward passes**: VLM models may invoke
    the same visual-block sequence twice per training step (image forward +
    video forward / FSDP dummy_forward).  ``GetCnt`` must produce unique keys
    across these passes; the original MindSpeed-MM implementation reset
    ``_block_tensor_nums`` when ``block_idx`` decreased, causing key collisions
    (e.g. ``"26_0"`` generated twice) that made ``OffloadManager.clear`` fail
    on the second unpack.  The VeOmni fix increments existing counts instead
    of resetting, so the second pass produces ``"0_1"``, ``"1_1"``, … rather
    than repeating ``"0_0"``, ``"1_0"``, ….
    """
    manager = OffloadManager()
    for name, module, layer_idx, depth in modules:
        logger.info_rank0(
            f"Applying activation offload to module: {name}, offload idx: {layer_idx}, offload_layers_num: {depth}"
        )
        module._veomni_offload_layer_idx = layer_idx
        module._veomni_offload_depth = depth
        module._veomni_offload_hidden_states_idx = hidden_states_idx
        module._veomni_offload_prefetch = prefetch
        module._veomni_offload_manager = manager
        _patch_instance_call(module)


def _patch_instance_call(module):
    """Rebind one module to a private subclass wrapping ``__call__``.

    This keeps saved-tensor hooks outside gradient checkpointing without
    permanently mutating the original module class.
    """
    cls = type(module)
    if getattr(cls, "_veomni_async_offload_instance_class", False):
        return

    original_call = cls.__call__

    def patched_call(self, *args, **kwargs):
        hidden_states_idx = getattr(self, "_veomni_offload_hidden_states_idx", 0)
        prefetch = getattr(self, "_veomni_offload_prefetch", True)
        layer_idx = self._veomni_offload_layer_idx
        depth = self._veomni_offload_depth
        manager = self._veomni_offload_manager

        if layer_idx == 0 and not manager.items:
            manager.reset_getcnt()

        try:
            hidden_states = args[hidden_states_idx] if len(args) > hidden_states_idx else None
            if hidden_states is None:
                hidden_states = kwargs.get("hidden_states")
            if hidden_states is not None and not hasattr(hidden_states, "data_ptr"):
                hidden_states = None
        except (IndexError, AttributeError):
            hidden_states = None

        if hidden_states is not None:
            target_ptr = hidden_states.data_ptr()

            def check_fn(x):
                return x.data_ptr() == target_ptr
        else:
            check_fn = None

        ctx = async_save_on_cpu(
            block_idx=layer_idx,
            depth=depth,
            manager=manager,
            custom_check_fn=check_fn,
            prefetch=prefetch,
        )

        try:
            with ctx:
                return original_call(self, *args, **kwargs)
        except BaseException:
            manager.reset()
            raise

    async_offload_cls = type(
        f"AsyncOffload{cls.__name__}",
        (cls,),
        {
            "__call__": patched_call,
            "_veomni_async_offload_instance_class": True,
        },
    )
    module.__class__ = async_offload_cls


def apply_async_activation_offload(model, activation_offload_modules: List[str]):
    """Apply async activation offload to matched submodules.

    Uses a selected-instance ``__call__`` subclass so that
    ``saved_tensors_hooks`` wraps the ``GradientCheckpointingLayer`` checkpoint
    boundary without mutating the shared decoder class.

    Compatible with both GC-enabled and GC-disabled training:
      - With GC: async offload intercepts hidden_states (via _NoopSaveInputs),
        GC handles intermediate activations via recomputation.
      - Without GC: async offload intercepts hidden_states directly,
        intermediate activations stay on GPU.
    """
    if activation_offload_modules:
        matched_modules = get_offload_modules(model, activation_offload_modules)
        if not matched_modules:
            raise ValueError(
                f"activation_offload_modules did not match any model modules: {activation_offload_modules!r}"
            )
    else:
        matched_modules = _get_no_split_offload_modules(model)
    async_offload_modules(matched_modules)
