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


import types
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import create_selective_checkpoint_contexts, noop_context_fn

from ..models import load_model_weights
from ..utils import logging
from ..utils.import_utils import is_torch_version_greater_than
from .checkpoint import CheckpointFunction
from .fsdp import (
    clip_grad_norm_,
    init_fsdp_fn,
    parallel_init_fsdp_fn,
    parallel_load_safetensors,
    register_checkpoint_extension,
)
from .parallel_state import get_parallel_state
from .utils import get_module_from_path, set_module_from_path


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)


def verbose_fsdp_grouping(model, prefix="", depth=0):
    indent = "    " * depth

    for name, child in model.named_children():
        if isinstance(child, FullyShardedDataParallel):
            module_names = [m_name for m_name, _ in child.named_modules()][1:]  # [1:] 排除自身
            strategy = child.sharding_strategy
            logger.debug_rank0(f"{indent}├── [FSDP Group] {prefix}{name}")
            logger.debug_rank0(
                f"{indent}│   ├── Sharding Strategy: {strategy}, Mixed Precision: {child.mixed_precision}"
            )
            logger.debug_rank0(f"{indent}│   └── Contains Modules: {module_names}")

            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth + 1)
        else:
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth)


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()
    fsdp_no_shard_states = None

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") != "cuda":
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
        if kwargs.pop("enable_fsdp_offload", False):
            raise ValueError("Only FSDP training supports `enable_fsdp_offload`.")

    if enable_mixed_precision:  # upcast to float32 before feed it to optimizer
        model = model.float()

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        ops_to_save = kwargs.pop("ops_to_save", None)
        context_fn = (
            partial(create_selective_checkpoint_contexts, ops_to_save) if ops_to_save is not None else noop_context_fn
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant, "context_fn": context_fn}
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        ep_param_suffix = parallel_plan.ep_param_suffix

        fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        fsdp_no_shard_states_fqn_to_module = parallel_plan.get_fsdp_no_shard_info(model)

        fsdp_no_shard_states = list(fsdp_no_shard_states_fqn_to_module.values())
        fsdp_no_shard_states_fqn = list(fsdp_no_shard_states_fqn_to_module.keys())
        logger.info_rank0(f"Apply expert parallel to the model successfully.\nEP modules: {fsdp_no_shard_states_fqn}.")
    else:
        fqn2spec_info = None
        ep_param_suffix = None
        fsdp_no_shard_states = None
        fsdp_no_shard_states_fqn = None

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }
            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
                fsdp_kwargs["mp_policy"] = mp_policy

            ignore_modules_in_mixed_precision = tuple()
            if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                ignore_modules_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()

            def apply_fsdp_to_decoder_blocks(module: "nn.Module") -> None:
                if module.__class__.__name__ in basic_modules or module.__class__ in ignore_modules_in_mixed_precision:
                    logger.debug(f"Apply FSDP2 to {module.__class__.__name__}.")
                    if module.__class__ in ignore_modules_in_mixed_precision:
                        fully_shard(module, **{k: v for k, v in fsdp_kwargs.items() if k != "mp_policy"})
                    else:
                        fully_shard(module, **fsdp_kwargs)

            if basic_modules:
                model.apply(apply_fsdp_to_decoder_blocks)

            fully_shard(model, **fsdp_kwargs)

            if kwargs.get("init_device") == "meta":
                if weights_path is None:
                    # shard init empty model with fsdp2
                    model.to_empty(device="cuda")
                    model.init_weights()
                else:
                    from torch.distributed.tensor import distribute_tensor

                    load_model_weights(model, weights_path, "cuda", dtensor_factory=distribute_tensor)

        elif parallel_state.dp_mode == "fsdp1":
            wrap_policy = partial(
                lambda_auto_wrap_policy, lambda_fn=lambda module: module.__class__.__name__ in basic_modules
            )

            # set fsdp/hsdp sharding strategy
            if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
                strategy = ShardingStrategy.HYBRID_SHARD
            else:
                strategy = ShardingStrategy.FULL_SHARD

            fsdp_kwargs = {
                "auto_wrap_policy": wrap_policy,
                "ignored_states": fsdp_no_shard_states,
                "device_id": torch.cuda.current_device(),
                "sharding_strategy": strategy if enable_full_shard else ShardingStrategy.NO_SHARD,
                "use_orig_params": True,
            }

            fsdp_kwargs["device_mesh"] = parallel_state.fsdp_mesh

            fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))

            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                    mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()

                fsdp_kwargs["mixed_precision"] = mixed_precision

            if kwargs.get("init_device") == "cpu":
                logger.info_rank0("Enable rank0-only initialization.")
                fsdp_kwargs["sync_module_states"] = True
                if parallel_state.global_rank != 0:
                    fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device="cuda")
            elif kwargs.get("init_device") == "meta":
                # assert weights_path is not None, "`weights_path` must be provided when `init_device=meta` for fsdp1."

                logger.info_rank0("Enable meta initialization.")
                if weights_path is None:
                    logger.info_rank0("weights_path is None during meta initialization.")

                ignore_param_names = (
                    [".".join([fqn, k]) for fqn in fsdp_no_shard_states_fqn for k in ep_param_suffix]
                    if fsdp_no_shard_states_fqn is not None
                    else None
                )
                shard_states = (
                    parallel_load_safetensors(weights_path, ignore_param_name=ignore_param_names)
                    if weights_path
                    else kwargs.get("state_dict", {})
                )
                fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                    model, shard_states, ignore_param_name=ignore_param_names
                )

            if kwargs.pop("enable_fsdp_offload", False):
                logger.info_rank0("Enable offloading for parameters & gradients & optimizer states.")
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

            if kwargs.pop("enable_forward_prefetch", False):
                fsdp_kwargs["forward_prefetch"] = True
            else:
                fsdp_kwargs["forward_prefetch"] = False
                fsdp_kwargs["backward_prefetch"] = None

            # FULLY_SHARD first
            model = FullyShardedDataParallel(model, **fsdp_kwargs)

            if fsdp_no_shard_states is not None:
                # apply NO_SHARD the ignored_states, but wrap into DDP
                if parallel_state.ep_fsdp_mesh["ep_fsdp"].size() == 1:
                    moe_sharding_strategy = ShardingStrategy.NO_SHARD
                    ep_fsdp_device_mesh = parallel_state.fsdp_mesh
                else:
                    moe_sharding_strategy = ShardingStrategy.FULL_SHARD
                    ep_fsdp_device_mesh = parallel_state.ep_fsdp_mesh["ep_fsdp"]

                logger.info_rank0(f"Apply {moe_sharding_strategy} states on '{fsdp_no_shard_states_fqn}'.")
                fsdp_kwargs.pop("ignored_states", None)
                fsdp_kwargs.pop("auto_wrap_policy", None)
                fsdp_kwargs["sharding_strategy"] = moe_sharding_strategy
                fsdp_kwargs["device_mesh"] = ep_fsdp_device_mesh
                logger.info_rank0(f"{ep_fsdp_device_mesh=}")
                for fqn in fsdp_no_shard_states_fqn:
                    no_shard_module = get_module_from_path(model, fqn)
                    if kwargs.get("init_device") == "meta":
                        specific_param_name = [".".join([fqn, k]) for k in ep_param_suffix]
                        shard_states = (
                            parallel_load_safetensors(weights_path, specific_param_name=specific_param_name)
                            if weights_path
                            else {}
                        )
                        if weights_path:
                            for suffix in ep_param_suffix:
                                shard_states[suffix] = shard_states.pop(".".join([fqn, suffix]))
                        fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                            no_shard_module, shard_states, specific_param_name=ep_param_suffix
                        )
                    fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
                    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
                    fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
                    set_module_from_path(model, fqn, fsdp_module)

            _lazy_init(model, model)

            # Apply fsdp extension to FSDP model
            save_hook_mesh = parallel_state.ep_fsdp_device_mesh if parallel_state.ep_enabled else None
            logger.info_rank0("Register Checkpoints Extension hook to the model")
            register_checkpoint_extension(
                fsdp_model=model,
                save_hook_mesh=save_hook_mesh,
                fqn2spec_info=fqn2spec_info,
            )

            if parallel_state.ep_enabled:
                model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)

            verbose_fsdp_grouping(model)
        else:
            ddp_kwargs = {"device_ids": [parallel_state.local_rank]}
            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
                ddp_kwargs["mixed_precision"] = mixed_precision

            model = DDP(model, **ddp_kwargs)

    return model
