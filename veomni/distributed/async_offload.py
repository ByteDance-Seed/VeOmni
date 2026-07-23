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
#   - Uses VeOmni's Singleton metaclass (veomni.utils.singleton)
#   - Uses VeOmni's TrainingContext (veomni.distributed.training_context)
#   - Uses VeOmni's logger (veomni.utils.logging) instead of print_rank
#   - Uses VeOmni's module_match (veomni.utils.module_match) instead of mindspeed's str_match
#   - Supports wildcard pattern ``{*}`` for sequential module groups (e.g. model.layers.{*})

from typing import List

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..utils import logging
from ..utils.device import create_event, create_stream, get_current_stream, switch_to_specified_stream
from ..utils.module_match import module_name_match
from ..utils.singleton import Singleton
from .training_context import TrainingContext


logger = logging.get_logger(__name__)


def base_check_fn(tensor) -> bool:
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    return True


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

    def get_prefetch_keys(self, block_idx, tensor_idx):
        prefetch_block_idx = max((idx for idx in self._block_tensor_nums.keys() if idx < block_idx), default=None)

        if prefetch_block_idx is None:
            return []

        block_tensor_nums = self._block_tensor_nums[block_idx]
        prefetch_idxs = list(range(0, block_tensor_nums))
        return ["{}_{}".format(block_idx - 1, prefetch_idx) for prefetch_idx in prefetch_idxs]

    def get_layer_tensor_nums(self, block_idx):
        return self._block_tensor_nums[block_idx]


class SwapTensor:
    def __init__(self, tensor, key):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu")

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stat = "device"
        self.key = key

        self.d2h_event = create_event()
        self.h2d_event = create_event()

    def launch_d2h(self, stream):
        if self.stat != "device":
            return

        forward_event = create_event()
        forward_event.record()
        with torch.no_grad():
            with switch_to_specified_stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.d2h_event.record()
                self.stat = "host"

    def wait_d2h_finished(self):
        if self.stat != "host":
            return
        get_current_stream().wait_event(self.d2h_event)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    def launch_h2d(self, h2d_stream):
        if self.stat != "host":
            return
        backward_event = create_event()
        backward_event.record()

        with torch.no_grad():
            with switch_to_specified_stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                self.tensor.storage().resize_(self.storage_size)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"


class OffloadItem:
    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt
        self.event = event


class OffloadManager(metaclass=Singleton):
    def __init__(self):
        self.items = {}
        self.npu_item = []
        self.getcnt = GetCnt()
        self.swap_stream = create_stream()

    def get_cnt(self, block_idx):
        return self.getcnt.get_cnt(block_idx)

    def reset_getcnt(self):
        self.getcnt.reset()

    def assert_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} does not exist in items")

    def exist(self, key):
        return key in self.items

    def put(self, key, act, event=None):
        if key in self.items:
            self.items[key].act = act
            self.items[key].ref_cnt += 1
            self.items[key].event = event
        else:
            self.items[key] = OffloadItem(act, 1, event)

    def put_npu_tensor(self, act):
        self.npu_item.append(act)

    def pop_npu_tensor(self):
        return self.npu_item.pop()

    def del_npu_tensor(self, prefix_key):
        for key in self.items.keys():
            if key.startswith(prefix_key):
                self.items[key].act.wait_d2h_finished()

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
        item = self.items[key]

        act = item.act
        if item.event is not None:
            item.get_event().wait()

        item.ref_cnt -= 1
        return act

    def prefetch_get(self, block_idx, tensor_idx, h2d_stream, d2h_stream):
        prefetch_keys = self.getcnt.get_prefetch_keys(block_idx, tensor_idx)
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


class async_save_on_cpu(saved_tensors_hooks):
    def __init__(
        self,
        block_idx,
        depth,
        custom_check_fn=None,
        prefetch=True,
    ) -> None:
        def _pack_to_cpu(tensor):
            if not base_check_fn(tensor):
                return tensor

            if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                return tensor

            key, after_block = OffloadManager().get_cnt(block_idx)
            d2h_stream = OffloadManager().swap_stream

            if after_block:
                OffloadManager().del_npu_tensor("{}_".format(block_idx - 1))

            if block_idx == depth - 1:
                return tensor

            swap_tensor = SwapTensor(tensor, key)

            if block_idx < depth - 1:
                swap_tensor.launch_d2h(d2h_stream)

            OffloadManager().put(key, swap_tensor)
            return swap_tensor

        def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
            if isinstance(swap_tensor, torch.Tensor):
                return swap_tensor

            d2h_stream = OffloadManager().swap_stream
            h2d_stream = OffloadManager().swap_stream
            swap_tensor.launch_h2d(h2d_stream)

            get_current_stream().wait_event(swap_tensor.h2d_event)

            block_idx, tensor_idx = swap_tensor.key.split("_")
            OffloadManager().clear(swap_tensor.key)

            TrainingContext().set_layer_index(int(block_idx))

            if prefetch:
                OffloadManager().prefetch_get(int(block_idx), int(tensor_idx), h2d_stream, d2h_stream)
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
                if module_name_match(plan_name, name):
                    if not any(item[0] == name for item in matched_submodules):
                        matched_submodules.append([name, module, offload_layers, depth])
                        offload_layers += 1

    for matched_submodule in matched_submodules:
        matched_submodule[-1] = offload_layers

    return matched_submodules


def async_offload_modules(modules, prefetch=True, hidden_states_idx=0):
    """Apply async activation offload via class-level ``__call__`` patching.

    For each module, per-instance attributes (``_veomni_offload_layer_idx``,
    ``_veomni_offload_depth``, ``_veomni_offload_hidden_states_idx``,
    ``_veomni_offload_prefetch``) are set, then the module's class's
    ``__call__`` is patched to wrap it with ``async_save_on_cpu`` context.

    Class-level patching is required because Python's dunder-method dispatch
    bypasses instance-level assignments: ``module(args)`` always resolves to
    ``type(module).__call__``, never to ``module.__call__``.

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

    Each unique class is patched only once (guarded by
    ``_veomni_async_offload_patched``).  Instances of the same class that lack
    ``_veomni_offload_layer_idx`` (i.e., not in the offload plan) fall through
    to the original ``__call__`` transparently.

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
    for name, module, layer_idx, depth in modules:
        logger.info_rank0(
            f"Applying activation offload to module: {name}, offload idx: {layer_idx}, offload_layers_num: {depth}"
        )
        module._veomni_offload_layer_idx = layer_idx
        module._veomni_offload_depth = depth
        module._veomni_offload_hidden_states_idx = hidden_states_idx
        module._veomni_offload_prefetch = prefetch
        _patch_class_call(type(module))


def _patch_class_call(cls):
    """Patch ``cls.__call__`` to wrap with ``async_save_on_cpu`` context.

    The patched ``__call__`` reads per-instance config from ``self`` at call
    time, so instances of the same class can have different offload settings.
    Only patches each class once (tracked via ``_veomni_async_offload_patched``).
    """
    if getattr(cls, "_veomni_async_offload_patched", False):
        return

    original_call = cls.__call__

    def patched_call(self, *args, **kwargs):
        if not hasattr(self, "_veomni_offload_layer_idx"):
            return original_call(self, *args, **kwargs)

        hidden_states_idx = getattr(self, "_veomni_offload_hidden_states_idx", 0)
        prefetch = getattr(self, "_veomni_offload_prefetch", True)
        layer_idx = self._veomni_offload_layer_idx
        depth = self._veomni_offload_depth

        if layer_idx == 0 and len(OffloadManager().items) == 0:
            OffloadManager().reset_getcnt()

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
            custom_check_fn=check_fn,
            prefetch=prefetch,
        )

        TrainingContext().set_model_depth(depth)
        TrainingContext().set_layer_index(layer_idx)

        with ctx:
            return original_call(self, *args, **kwargs)

    cls.__call__ = patched_call
    cls._veomni_async_offload_patched = True


def apply_async_activation_offload(model, activation_offload_modules: List[str]):
    """Apply async activation offload to matched submodules.

    Uses class-level ``__call__`` patching so that ``saved_tensors_hooks``
    wraps the ``GradientCheckpointingLayer`` checkpoint boundary, correctly
    intercepting hidden states saved for backward recomputation.

    Compatible with both GC-enabled and GC-disabled training:
      - With GC: async offload intercepts hidden_states (via _NoopSaveInputs),
        GC handles intermediate activations via recomputation.
      - Without GC: async offload intercepts hidden_states directly,
        intermediate activations stay on GPU.
    """
    if not activation_offload_modules:
        return
    matched_modules = get_offload_modules(model, activation_offload_modules)
    if not matched_modules:
        raise ValueError(
            "activation_offload_modules did not match any model modules: "
            f"{activation_offload_modules!r}"
        )
    async_offload_modules(matched_modules)
