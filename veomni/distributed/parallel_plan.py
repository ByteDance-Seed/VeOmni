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


from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from ..utils import logging
from .utils import check_fqn_match, get_module_from_path, set_module_from_path


logger = logging.get_logger(__name__)


@dataclass
class SpecInfo:
    para_name: str  # name of the ExtraParallel this fqn belongs to
    placement: Union[Shard, Replicate]
    fqn: str
    para_fsdp_mesh: DeviceMesh
    # Tensor dim that FSDP2 shards this param along (set in ``torch_parallelize``
    # after the ``shard_placement_fn`` is decided: 1 by default, 0 for the
    # Muon zero-comm layout). Used by the checkpointer to build the correct
    # DTensor placements when (re)storing the ExtraParallel dim.
    fsdp_shard_dim: int = 1

    @property
    def para_mesh(self):
        if self.para_fsdp_mesh is not None:
            return self.para_fsdp_mesh[self.para_name]
        else:
            return None


class ParallelPlan:
    def __init__(self, extra_parallel_plan: Dict[str, Dict[str, Shard]]):
        self.extra_parallel_plan = extra_parallel_plan
        self.extra_parallel_fsdp_no_shard_module = {
            para_name: {".".join(list(plan.keys())[0].split(".")[:-1])}
            for para_name, plan in self.extra_parallel_plan.items()
        }

    def apply(self, model: nn.Module, extra_parallel_fsdp_device_mesh: Dict[str, DeviceMesh]):
        """
        xxx_fsdp_mesh: [replicate, replicate, ... , shard]
        """
        extra_parallel_mesh = {
            para: para_fsdp_mesh[para] if para_fsdp_mesh is not None else None
            for para, para_fsdp_mesh in extra_parallel_fsdp_device_mesh.items()
        }

        fqn2spec_info = {}
        for para, para_plan in self.extra_parallel_plan.items():
            para_mesh = extra_parallel_mesh[para]
            para_fsdp_mesh = extra_parallel_fsdp_device_mesh[para]
            if para_plan and para_mesh is not None:
                para_size = para_mesh.size(-1)
                para_replicate = [Replicate() for _ in range(para_mesh.ndim)]
                for fqn, param in model.named_parameters():
                    for fqn_pattern, shard in para_plan.items():
                        if check_fqn_match(fqn_pattern, fqn):
                            assert param.size(shard.dim) % para_size == 0
                            para_placement = para_replicate[:-1] + [shard]
                            logger.info_rank0(
                                f"{para} sharding: slicing param {fqn} along {para}_mesh with placement {para_placement}"
                            )
                            dtensor = DTensor.from_local(
                                local_tensor=param.data, device_mesh=para_mesh, placements=para_replicate
                            )
                            dtensor = dtensor.redistribute(device_mesh=para_mesh, placements=para_placement)
                            local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                            local_chunk.spec_info = SpecInfo(
                                para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=shard, fqn=fqn
                            )
                            set_module_from_path(model, fqn, local_chunk)
                            fqn2spec_info[fqn] = SpecInfo(
                                para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=shard, fqn=fqn
                            )
                            break
                    if fqn not in fqn2spec_info:  # not sharded
                        param.spec_info = SpecInfo(
                            para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=Replicate(), fqn=fqn
                        )
                        fqn2spec_info[fqn] = SpecInfo(
                            para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=Replicate(), fqn=fqn
                        )

        for fqn, param in model.named_parameters():
            assert hasattr(param, "spec_info"), f"Internal Error: {fqn=} with {param=} is omitted"

        return fqn2spec_info

    def get_fsdp_no_shard_info(self, model: nn.Module):
        if self.extra_parallel_fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        fsdp_no_shard_states_fqn_to_para = {}
        for fqn, _param in model.named_modules():
            for para, no_shard_patterns in self.extra_parallel_fsdp_no_shard_module.items():
                for no_shard_pattern in no_shard_patterns:
                    if check_fqn_match(no_shard_pattern, fqn):
                        fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
                        fsdp_no_shard_states_fqn_to_para[fqn] = para
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, (
            "no module in model match `extra_parallel_fsdp_no_shard_module`"
        )

        return fsdp_no_shard_states_fqn_to_module, fsdp_no_shard_states_fqn_to_para

    def get_extra_parallel_fsdp_no_shard_info(self, model: nn.Module, para_name: str):
        if self.extra_parallel_fsdp_no_shard_module[para_name] is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, _param in model.named_modules():
            for no_shard_pattern in self.extra_parallel_fsdp_no_shard_module[para_name]:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, (
            "no module in model match `extra_parallel_fsdp_no_shard_module`"
        )

        return fsdp_no_shard_states_fqn_to_module

    def update_prefix(self, prefix: str):
        """
        Update extra_parallel_plan when model is wrappered.
        """
        self.extra_parallel_plan = {
            para_name: {prefix + "." + k: v for k, v in plan.items()}
            for para_name, plan in self.extra_parallel_plan.items()
        }
        self.extra_parallel_fsdp_no_shard_module = {
            para_name: {prefix + "." + no_shard_pattern for no_shard_pattern in para_fsdp_no_shard_module}
            for para_name, para_fsdp_no_shard_module in self.extra_parallel_fsdp_no_shard_module.items()
        }

    def shard_tensor(self, tensor: "torch.Tensor", full_param_name: str, target_shape: tuple) -> "torch.Tensor":
        """
        Shard tensor for one extra_parallel parallelism if needed.
        In the future, we may add other tensor slicing in this function to determine TP parameter and its sharding.

        Args:
            tensor: The tensor to potentially shard
            full_param_name: The full parameter name (e.g., "model.layers.0.mlp.experts.gate_proj.weight")
            target_shape: The expected shape of the target parameter

        Returns:
            The original tensor or a sliced version for one extra_parallel parallelism
        """
        shard_group = self._get_shard_parameter_groupname(full_param_name)
        if shard_group:
            return self._slice_shard_tensor(tensor, full_param_name, target_shape, shard_group)
        return tensor

    def _get_shard_parameter_groupname(self, parameter_name: str) -> bool:
        # note that parameter_name should be full name
        for para_name, para_plan in self.extra_parallel_plan.items():
            for fqn_pattern in para_plan.keys():
                if check_fqn_match(fqn_pattern, parameter_name):
                    return para_name
        return None

    def _slice_shard_tensor(
        self, tensor: "torch.Tensor", parameter_name: str, target_shape: tuple, shard_group: str
    ) -> "torch.Tensor":
        """Slice shard tensor for extra_parallel parallelism."""
        try:
            from .parallel_state import get_parallel_state

            parallel_state = get_parallel_state()

            # Check if we need to slice based on tensor vs target shape mismatch
            if len(tensor.shape) >= 1 and len(target_shape) >= 1:
                # If tensor has more feature than target, we need to slice
                if tensor.shape[0] > target_shape[0] and tensor.shape[0] % target_shape[0] == 0:
                    para_size = tensor.shape[0] // target_shape[0]
                    para_rank = (
                        parallel_state.extra_parallel_rank(shard_group)
                        if parallel_state.extra_parallel_enabled(shard_group)
                        else 0
                    )

                    start_idx = para_rank * target_shape[0]
                    end_idx = start_idx + target_shape[0]

                    sliced_tensor = tensor[start_idx:end_idx]

                    logger.info_rank0(
                        f"{shard_group} parameter {parameter_name}: sliced {tensor.shape} -> {sliced_tensor.shape} "
                        f"for {shard_group} rank {para_rank}/{para_size}"
                    )

                    return sliced_tensor

            # No slicing needed
            return tensor

        except Exception as e:
            # Fallback: if anything fails, return original tensor
            logger.warning(f"Failed to slice extra_parallel tensor {parameter_name}: {e}")
            return tensor


def get_runtime_parallel_plan(model: nn.Module) -> "ParallelPlan":
    """Return ``model.get_parallel_plan()`` with PEFT + MoE-LoRA bridges applied.

    PEFT wraps the model so every ``named_parameters()`` /
    ``named_modules()`` FQN gets a ``base_model.model.`` prefix, but
    ``model.get_parallel_plan()`` is forwarded straight through to the
    bare base model (PEFT's ``__getattr__``). The plan it returns
    therefore uses prefix-less FQNs (e.g.
    ``model.layers.*.mlp.experts.gate_up_proj``); without bridging,
    every pattern-matching call site (param sharding in
    :meth:`ParallelPlan.apply`, no-shard module discovery in
    :meth:`ParallelPlan.get_fsdp_no_shard_info`, EP slicing in
    :meth:`ParallelPlan.shard_tensor` from the rank-0 broadcast load
    path) finds zero matches and the EP+PEFT path either silently
    skips sharding or asserts.

    Two bridges, in order:

    1. **PEFT prefix.** If any param FQN starts with ``base_model.``,
       call :meth:`ParallelPlan.update_prefix` to rewrite every plan
       pattern with the ``base_model.model.`` prefix. Detection is by
       parameter-name probe rather than ``isinstance(model, PeftModel)``
       to avoid importing PEFT here.

    2. **MoE-LoRA wrapper rewrite.** For each VeOmni
       ``LoraSharedExperts`` / ``LoraIndependentExperts`` wrapper in
       the model, rewrite the bare base-param patterns
       (``...experts.gate_up_proj``, ``...experts.down_proj``) to the
       PEFT-aligned wrapped FQNs (``...experts.gate_up_proj.base_layer.weight``
       etc.) so :meth:`ParallelPlan.apply` keeps EP-sharding the base
       experts after wrapping. Additionally, for ``LoraIndependentExperts``
       only, register every per-expert LoRA tensor
       (``...experts.<spec>.lora_A.<adapter>.weight`` /
       ``...experts.<spec>.lora_B.<adapter>.weight``, all 3-D with
       leading dim ``num_experts``) as ``Shard(0)`` under the same
       parallel group. ``LoraSharedExperts`` LoRA tensors are
       2-D / replicated and don't enter the EP plan.

    Centralising both bridges here means all four call sites
    (``parallelize_model_fsdp1`` / ``parallelize_model_fsdp2`` /
    ``rank0_load_and_broadcast_weights`` / ``load_model_weights``)
    pick up the same prefixed + extended plan via this helper.
    """
    get_plan = getattr(model, "get_parallel_plan", None)
    if get_plan is None:
        return None
    parallel_plan = get_plan()
    if parallel_plan is None:
        return parallel_plan

    is_peft_model = any(n.startswith("base_model.") for n, _ in model.named_parameters())
    if is_peft_model:
        parallel_plan.update_prefix("base_model.model")

    _rewrite_plan_for_moe_lora_wrappers(model, parallel_plan)
    return parallel_plan


def _rewrite_plan_for_moe_lora_wrappers(model: nn.Module, parallel_plan: "ParallelPlan") -> None:
    """Mutate ``parallel_plan`` in place to match the MoE-LoRA wrapped FQN layout.

    Two-step rewrite, both walked off the *concrete* model so the resulting
    plan uses ``check_fqn_match``-friendly literal patterns (no ``*``):

    1. For each LoRA wrapper, find the EP plan entries that target a base
       parameter under it (the bare-Param patterns like
       ``model.layers.*.mlp.experts.gate_up_proj``). Replace each such
       wildcard pattern with one concrete entry per matched wrapper, at
       the wrapped FQN ``<wrapper_fqn>.<base_param>.base_layer.weight``.
       The Shard placement is preserved verbatim.

    2. For :class:`~veomni.lora.moe_layers.LoraIndependentExperts` only,
       enumerate each per-expert LoRA tensor under the wrapper
       (``<spec>.lora_A.<adapter>.weight`` /
       ``<spec>.lora_B.<adapter>.weight``, leading dim = ``num_experts``)
       and add them to the same EP group as ``Shard(0)``.

    Imports of the MoE-LoRA module are lazy to avoid the
    ``veomni.lora.moe_layers`` ↔ ``veomni.distributed`` import cycle.
    """
    from ..lora.moe_layers import LoraIndependentExperts, _is_lora_param_name, is_lora_moe_experts

    if not any(is_lora_moe_experts(m) for _, m in model.named_modules()):
        return

    target_group: str | None = None
    for group_name, group_plan in parallel_plan.extra_parallel_plan.items():
        if any("experts.gate_up_proj" in p or "experts.down_proj" in p for p in group_plan):
            target_group = group_name
            break
    if target_group is None:
        return

    plan = parallel_plan.extra_parallel_plan[target_group]

    # Step 1: rewrite bare-Param patterns to ``<wrapper_fqn>.<base>.base_layer.weight``.
    # We iterate the wildcard patterns once, and for each emit one concrete entry
    # per matching wrapper (one per layer in practice). Patterns that don't match
    # any wrapper pass through unchanged, so non-LoRA EP plans (e.g. mixed
    # mod-some-layers-only setups) are unaffected.
    rewritten: Dict[str, Shard] = {}
    consumed_patterns: set[str] = set()
    for fqn_pattern, shard in plan.items():
        wrapper_path, _, base_name = fqn_pattern.rpartition(".")
        for fqn, module in model.named_modules():
            if not is_lora_moe_experts(module):
                continue
            if not check_fqn_match(wrapper_path, fqn):
                continue
            if not hasattr(module, base_name):
                continue
            spec = getattr(module, base_name)
            if not getattr(spec, "has_base_layer", False):
                continue
            new_fqn = f"{fqn}.{base_name}.base_layer.weight"
            rewritten[new_fqn] = shard
            consumed_patterns.add(fqn_pattern)

    for old_pattern in consumed_patterns:
        plan.pop(old_pattern, None)
    plan.update(rewritten)

    # Step 2: per-expert LoRA tensors for ``LoraIndependentExperts`` (Mode 1 only).
    lora_entries: Dict[str, Shard] = {}
    for fqn, module in model.named_modules():
        if not isinstance(module, LoraIndependentExperts):
            continue
        for n, _p in module.named_parameters(recurse=True):
            if not _is_lora_param_name(n):
                continue
            full_fqn = f"{fqn}.{n}" if fqn else n
            lora_entries[full_fqn] = Shard(0)
    plan.update(lora_entries)

    if rewritten or lora_entries:
        logger.info_rank0(
            f"Rewrote ExtraParallel plan group {target_group!r} for MoE-LoRA wrappers: "
            f"{len(rewritten)} base-param FQN(s) rewritten, "
            f"{len(lora_entries)} per-expert LoRA tensor(s) added"
        )
