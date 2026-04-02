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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import CPUOffload, FSDPModule, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import noop_context_fn

from ..models import load_model_weights, rank0_load_and_broadcast_weights
from ..models.module_utils import _convert_weight_key as convert_weight_key
from ..utils import logging
from ..utils.device import IS_NPU_AVAILABLE, get_device_id, get_device_type
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
from .utils import get_module_from_path, is_same_module_from_path, set_module_from_path, sort_fqn_by_submodule_first


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)


def _reset_hf_initialized_flag(module: nn.Module) -> None:
    if hasattr(module, "_is_hf_initialized"):
        module._is_hf_initialized = False
    for child in module.children():
        _reset_hf_initialized_flag(child)


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


def _convert_state_dict_keys(state_dict, model):
    return {convert_weight_key(key, model): value for key, value in state_dict.items()}


def parallelize_model_fsdp1(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_full_shard: bool = True,
    enable_shard_grad_op: bool = False,
    enable_mixed_precision: bool = True,
    use_orig_params: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_no_shard_states=None,
    fsdp_no_shard_states_fqn=None,
    fqn2spec_info=None,
    **kwargs,
) -> "nn.Module":
    """
    Apply ExtraParallel (e.g. Expert Parallel + Embed Parallel + ...) + FSDP1 parallel strategy to the model.
    """
    parallel_state = get_parallel_state()

    if parallel_state.any_extra_parallel_enabled:
        parallel_plan = model.get_parallel_plan()
        fqn2spec_info = parallel_plan.apply(model, parallel_state.extra_parallel_fsdp_device_mesh)

        fsdp_no_shard_states_fqn_to_module, fsdp_no_shard_states_fqn_to_para = parallel_plan.get_fsdp_no_shard_info(
            model
        )
        fsdp_no_shard_states = [
            module
            for fqn, module in fsdp_no_shard_states_fqn_to_module.items()
            if parallel_state.extra_parallel_enabled(fsdp_no_shard_states_fqn_to_para[fqn])
        ]
        fsdp_no_shard_states_fqn = [
            fqn
            for fqn, module in fsdp_no_shard_states_fqn_to_module.items()
            if parallel_state.extra_parallel_enabled(fsdp_no_shard_states_fqn_to_para[fqn])
        ]
        root_model_wrap_module_names = []
        for module_name in basic_modules:
            is_same_module = False
            for fqn in fsdp_no_shard_states_fqn:
                if is_same_module_from_path(model, module_name, fqn):
                    is_same_module = True
                    break
            if not is_same_module:
                root_model_wrap_module_names.append(module_name)

        logger.info_rank0(
            f"Apply extra_parallel parallel to the model successfully.\nExtraParallel modules: {fsdp_no_shard_states_fqn}."
        )
    else:
        fqn2spec_info = None
        fsdp_no_shard_states = None
        fsdp_no_shard_states_fqn = None
        root_model_wrap_module_names = basic_modules

    wrap_policy = partial(
        lambda_auto_wrap_policy, lambda_fn=lambda module: module.__class__.__name__ in root_model_wrap_module_names
    )

    # set fsdp/hsdp sharding strategy
    if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
        strategy = ShardingStrategy.HYBRID_SHARD if enable_full_shard else ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        strategy = ShardingStrategy.FULL_SHARD if enable_full_shard else ShardingStrategy.SHARD_GRAD_OP
    fsdp_kwargs = {
        "auto_wrap_policy": wrap_policy,
        "ignored_states": fsdp_no_shard_states,
        "device_id": get_device_id(),
        "sharding_strategy": strategy if enable_full_shard or enable_shard_grad_op else ShardingStrategy.NO_SHARD,
        "use_orig_params": use_orig_params,
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
            fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device=get_device_type())
    elif kwargs.get("init_device") == "meta":
        logger.info_rank0("Enable meta initialization.")
        if weights_path is None:
            logger.info_rank0("weights_path is None during meta initialization.")

        shard_states = kwargs.get("shard_states", {})
        if weights_path:
            shard_states = parallel_load_safetensors(
                weights_path, cpu_load_param_name=kwargs.get("cpu_load_param_name", None)
            )

        fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
            model,
            shard_states.copy(),
            ignore_states=fsdp_no_shard_states,
            strict=kwargs.pop("strict", False),
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
        _extra_parallel_sharding_strategy = {}
        _extra_parallel_fsdp_device_mesh = {}
        for para in parallel_state.extra_parallel_names:
            if parallel_state.extra_parallel_fsdp_device_mesh[para] is None:
                _extra_parallel_sharding_strategy[para] = ShardingStrategy.NO_SHARD
                _extra_parallel_fsdp_device_mesh[para] = parallel_state.fsdp_mesh
            else:
                # Always use FULL_SHARD with ep_fsdp mesh, even when ep_fsdp size is 1.
                # FULL_SHARD on a size-1 mesh is a no-op for communication but still
                # applies mixed precision and gradient postdivide factor correctly.
                _extra_parallel_sharding_strategy[para] = ShardingStrategy.FULL_SHARD
                _extra_parallel_fsdp_device_mesh[para] = parallel_state.extra_parallel_fsdp_mesh(para)[f"{para}_fsdp"]

            logger.info_rank0(
                f"Apply sharding_strategy={_extra_parallel_sharding_strategy[para]} on '{fsdp_no_shard_states_fqn}' at parallel {para}."
            )
            logger.info_rank0(f"{para}_fsdp_device_mesh={_extra_parallel_fsdp_device_mesh[para]}.")

        fsdp_kwargs.pop("ignored_states", None)
        fsdp_kwargs.pop("auto_wrap_policy", None)

        for fqn in fsdp_no_shard_states_fqn:
            if not parallel_state.extra_parallel_enabled(fsdp_no_shard_states_fqn_to_para[fqn]):
                continue
            no_shard_module = get_module_from_path(model, fqn)
            para = fsdp_no_shard_states_fqn_to_para[fqn]

            assert para is not None, f"para is None for fqn {fqn}"

            fsdp_kwargs["sharding_strategy"] = _extra_parallel_sharding_strategy[para]
            fsdp_kwargs["device_mesh"] = _extra_parallel_fsdp_device_mesh[para]

            if kwargs.get("init_device") == "meta":
                key_prefix = fqn + "."
                para_shard_states = {
                    k[len(key_prefix) :]: v for k, v in shard_states.items() if k.startswith(key_prefix)
                }
                fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                    no_shard_module,
                    para_shard_states,
                    strict=kwargs.pop("strict", False),
                )

            fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
            fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
            fsdp_state._gradient_postdivide_factor *= parallel_state.extra_parallel_sizes[para]
            set_module_from_path(model, fqn, fsdp_module)

    _lazy_init(model, model)

    # Apply fsdp extension to FSDP model
    save_hook_mesh = {
        para: parallel_state.extra_parallel_fsdp_device_mesh[para]
        if parallel_state.extra_parallel_enabled(para)
        else None
        for para in parallel_state.extra_parallel_names
    }
    logger.info_rank0("Register Checkpoints Extension hook to the model")
    register_checkpoint_extension(
        fsdp_model=model,
        save_hook_mesh=save_hook_mesh,
        fqn2spec_info=fqn2spec_info,
    )

    if parallel_state.any_extra_parallel_enabled:
        model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)

    verbose_fsdp_grouping(model)

    # FSDP1 path keeps its own clipper already above

    return model


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_reshard_after_forward: bool = True,
    enable_mixed_precision: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    Apply ExtraParallel (e.g. Expert Parallel or Embed Parallel) + FSDP2 parallel strategy to the model.

    For Expert Parallel, the flow is as follows:
        1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
        2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
        3. Apply FSDP2 to regular modules: Standard dim-0 sharding
        4. Result: Expert params [32,H/fsdp_size,I], regular params use standard FSDP2

    For ExtraParallel, see test_clip_grad_norm_fsdp2_ep2_emb4 with Expert Parallel + Embed Parallel, where
        ToyMoeAndEmbedModel(
            (embed_tokens): ToyEmbed()
            (decoder): ToyMoeAndEmbedDecoderLayer(
                (embed_tokens): ToyEmbed()
                (moe): ToyMoeExperts()
            )
        )
        ToyMoeAndEmbedModel._no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"embed_tokens.weight": Shard(0), "decoder.embed_tokens.weight": Shard(0)}
        ep_size, emb_size = 2, 4
    We will use this model for illustration of Expert Parallel + Embed Parallel below.
    """
    parallel_state = get_parallel_state()

    # Step 0: Get target classes to shard later
    model_no_split_modules = getattr(model, "_no_split_modules", None) or []
    target_classes = set(model_no_split_modules) | set(basic_modules or [])

    # Make a list of tuples that contains target classes' name and module
    # Note that all target classes should include all ExtraParallel modules.
    #   e.g. `ToyEmbed` and `ToyMoeAndEmbedDecoderLayer` include `embed_tokens.weight` and `decoder.embed_tokens.weight`
    # Note that target class A is allowed to include target class B:
    #   e.g. `ToyMoeAndEmbedDecoderLayer` includes target class `ToyEmbed`
    # Thus, target module A could include target module B.
    #   e.g. `decoder` includes `decoder.embed_tokens`
    target_modules: List[Tuple[str, nn.Module]] = [
        (fqn, mod) for fqn, mod in model.named_modules() if mod.__class__.__name__ in target_classes
    ]
    logger.info_rank0(f"target classes to shard: {target_classes}")

    # Step 1: Apply ExtraParallel
    #   e.g. Apply expert parallelism (slice expert tensors [128,H,I] -> [16,H,I])
    #        Apply embed parallelism (slice embed tensors [64,H] -> [16,H])
    if parallel_state.any_extra_parallel_enabled:
        parallel_plan = model.get_parallel_plan()
        assert parallel_plan is not None, (
            "ExtraParallel needs parallel plan defined in the model! \
            Please see veomni/models/transformers/qwen3_moe/parallel_plan.py for example of expert parallelism. \
            Please see tests/utils/test_extra_parallel_clip_grad_norm.py::test_clip_grad_norm_fsdp2_ep2_emb4 \
            for example of expert parallelism + embed parallelism."
        )
        # Add SpecInfo to extra_parallel modules,
        #   e.g. embed_tokens.weight, decoder.regular_mlp, decoder.embed_tokens.weight, and decoder.moe.experts
        fqn2spec_info = parallel_plan.apply(model, parallel_state.extra_parallel_fsdp_device_mesh)

        # Attach spec mapping for checkpoint load-time reconstruction
        model._fqn2spec_info = fqn2spec_info
        # ExtraParallel mesh does not really exist in ExtraParallel parameters' device mesh.
        # ExtraParallel parameters are loaded as local tensors to be later sharded by fully_shard.
        # We store each extra_parallel's fqn to module mapping in _extra_parallel_map.
        #  e.g. {'emb': {'embed_tokens': ToyEmbed, 'decoder.embed_tokens': ToyEmbed}, 'ep': {'decoder.moe': ToyMoeExperts}}
        _extra_parallel_mesh = {}
        _extra_parallel_map = {}
        for para in parallel_state.extra_parallel_names:
            if parallel_state.extra_parallel_enabled(para):
                _extra_parallel_mesh[para] = parallel_state.extra_parallel_fsdp_device_mesh[para]
                _extra_parallel_map[para] = parallel_plan.get_extra_parallel_fsdp_no_shard_info(model, para)
            else:
                _extra_parallel_mesh[para] = None
                _extra_parallel_map[para] = None

            logger.info_rank0(
                f"Applied {para}: tensors sliced along dimension ({para} mesh: {_extra_parallel_mesh[para]})"
            )
            logger.info_rank0(f"{para} Map: {_extra_parallel_map[para]}")

    else:
        fqn2spec_info = None
        _extra_parallel_mesh = None
        _extra_parallel_map = None

    # Extract ExtraParallel modules from the target classes if any, then pair them.
    # Regard each target module as a layer.
    # Note that all target modules should include ExtraParallel modules.
    #     If we have ToyMoeAndEmbedModel like the above, then,
    #         layer_pairs_list = [
    #            ('decoder.embed_tokens', (ToyEmbed, {'emb': ToyEmbed, 'ep': None})),
    #            ('embed_tokens', (ToyEmbed, {'emb': ToyEmbed, 'ep': None})),
    #            ('decoder', (ToyMoeAndEmbedDecoderLayer, {'emb': ToyEmbed, 'ep': ToyMoeExperts}))
    #         ]
    layer_pairs = {}
    for layer_fqn, layer_mod in target_modules:
        layer_pair = [layer_mod]
        extra_parallel_mod = {}

        if parallel_state.any_extra_parallel_enabled:
            for para in parallel_state.extra_parallel_names:
                if _extra_parallel_map[para] is not None:
                    para_mod = next(
                        (
                            para_mod
                            for para_mod_fqn, para_mod in _extra_parallel_map[para].items()
                            if para_mod_fqn.startswith(layer_fqn)
                        ),
                        None,
                    )
                else:
                    para_mod = None
                extra_parallel_mod[para] = para_mod
        layer_pair.append(extra_parallel_mod)
        layer_pairs[layer_fqn] = tuple(layer_pair)

    logger.info_rank0(f"extra_parallel layer pairs: {layer_pairs}")

    # Step 2: Update fsdp2 kwargs
    fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh, "reshard_after_forward": enable_reshard_after_forward}
    # prepare mp_policy kwargs
    if enable_mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs["mp_policy"] = mp_policy

    if hasattr(model, "get_ignore_modules_in_mixed_precision"):
        modules_to_ignore_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()
    else:
        modules_to_ignore_in_mixed_precision = None

    if modules_to_ignore_in_mixed_precision:
        assert isinstance(modules_to_ignore_in_mixed_precision, tuple), (
            "modules_to_ignore_in_mixed_precision needs to be a tuple!"
        )
        mp_ignored_classes = modules_to_ignore_in_mixed_precision
        fsdp_kwargs_without_mp = dict(fsdp_kwargs)
        fsdp_kwargs_without_mp.pop("mp_policy", None)
        # for high-precision modules, we do not reshard them after forward to avoid all-gather them in backward
        # these modules will stay in GPU memory so please ensure high-precision modules do not contain too many parameters
        fsdp_kwargs_without_mp["reshard_after_forward"] = False
    else:
        mp_ignored_classes = None
        fsdp_kwargs_without_mp = fsdp_kwargs

    # prepare extra_parallel_fsdp2 kwargs
    extra_parallel_fsdp_kwargs = {}
    for para in parallel_state.extra_parallel_names:
        if parallel_state.extra_parallel_enabled(para):
            para_fsdp_mesh = parallel_state.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"]
            para_fsdp_kwargs = dict(fsdp_kwargs)
            para_fsdp_kwargs["mesh"] = para_fsdp_mesh
            # Prefer dim-1 sharding for expert/embed weights when composing with expert/embed shard on dim-0
            para_fsdp_kwargs["shard_placement_fn"] = lambda param: Shard(1)
            extra_parallel_fsdp_kwargs[para] = para_fsdp_kwargs
        else:
            extra_parallel_fsdp_kwargs[para] = None

    # Here we have a basic assumption for target module (e.g. embed_tokens, decoder) hierarchy:
    # | -- target module A (e.g. decoder)
    #   | -- target module B (e.g. decoder.embed_tokens)
    #   | -- extra parallel module C (e.g. decoder.moe)
    #     | -- no more target module or extra parallel module
    #   | -- mp modules
    #     | -- no more target module or extra parallel module
    #   | -- other module (e.g. attention, if provided)
    # e.g. Decoder Layer
    # | -- layers that are sharded by fully_shard(decode_layer) (e.g., Attention)
    # | -- experts layer (apply fully_shard separately in order to shard across EP groups on the same EP rank instead of sharding globally)
    # | -- layers (declared in model.modules_to_ignore_in_mixed_precision) that need to apply fully_shard separately due to different mp policy as the decoder layer
    #      (e.g., some models requires MoE TopK gate layer to have parameters in higher FP32 precision in forward).
    # NPU currently does not support the PreSumMul operation, so this operation is supported through the apply_hccl_premul_sum_patch.
    # TODO(https://github.com/ByteDance-Seed/VeOmni/issues/241):
    # NPU is missing PreSumMul ReduceOp. Need to remove this condition after the issue is resolved.
    if IS_NPU_AVAILABLE and parallel_state.any_extra_parallel_enabled:
        from veomni.ops.npu_patch.hccl_premul_sum import apply_hccl_premul_sum_patch

        apply_hccl_premul_sum_patch()

    # Sort layer_pairs by fqn by submodule order, as fully_shard should starts from bottom modules to top modules
    #   e.g. sorted_fqn_list = ['decoder.embed_tokens', 'embed_tokens', 'decoder']
    sorted_fqn_list = sort_fqn_by_submodule_first(list(layer_pairs.keys()))
    layer_pairs_list = [(fqn, layer_pairs[fqn]) for fqn in sorted_fqn_list]

    for layer_fqn, (layer_mod, extra_parallel_mod) in layer_pairs_list:
        # register all the FSDPModule inside this decoder layer for the convenience of manual prefetching configuration
        layer_mod._fsdp_modules = []

        for para in parallel_state.extra_parallel_names:
            # para (e.g. ep, emb) enabled and this layer contains the para (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens) module
            if (
                parallel_state.extra_parallel_enabled(para)
                and extra_parallel_mod[para] is not None
                and not isinstance(extra_parallel_mod[para], FSDPModule)
            ):
                # shard para module (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens)
                fully_shard(extra_parallel_mod[para], **extra_parallel_fsdp_kwargs[para])
                # average para (e.g. ep) grads across para (e.g. ep) ranks
                # NOTE: in torch 2.8 and later we should use
                # experts_mod.set_gradient_divide_factor(parallel_state.ep_size)
                # but for torch 2.7 we still use set_reduce_scatter_divide_factor(parallel_state.ep_size)
                gradient_divide_factor = parallel_state.extra_parallel_gradient_divide_factor(para)
                logger.info(f"setting grad divide factor for {para} module to {gradient_divide_factor}")
                if IS_NPU_AVAILABLE:
                    # NPU is using torch 2.7
                    extra_parallel_mod[para].set_reduce_scatter_divide_factor(gradient_divide_factor)
                else:
                    # from torch 2.8
                    extra_parallel_mod[para].set_gradient_divide_factor(gradient_divide_factor)
                layer_mod._fsdp_modules.append(extra_parallel_mod[para])

        # shard module that needs to ignore mixed precision control
        if mp_ignored_classes:
            for sub_mod in layer_mod.modules():
                if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                    fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                    layer_mod._fsdp_modules.append(sub_mod)

        # Shard everything else in the module:
        #   Note:
        #      if we have a model and layer_pairs_list like the above,
        #      when layer_mod (also called as target module, e.g. decoder.embed_tokens),
        #      is the parent of or equal to extra_parallel_mod[para] (e.g. ToyEmbed),
        #      no need to shard layer_mod again.
        if not isinstance(layer_mod, FSDPModule):
            fully_shard(layer_mod, **fsdp_kwargs)
            layer_mod._fsdp_modules.append(layer_mod)
        logger.info_rank0(f"{layer_fqn=}, {layer_mod._fsdp_modules=}")

    # shard root model
    fully_shard(model, **fsdp_kwargs)

    # configure manual prefetching when needed
    need_manual_prefetch = parallel_state.any_extra_parallel_enabled or mp_ignored_classes is not None
    if need_manual_prefetch:
        blocks = [pair[1][0] for pair in layer_pairs_list]  # all target modules
        next_blocks = blocks[1:] + [None]
        for current_block, next_block in zip(blocks, next_blocks):
            if next_block is not None:
                prefetch_modules = next_block._fsdp_modules
                # prefetch in order of attn, gate, experts
                current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

        # configure backward prefetch
        rev_blocks = list(reversed(blocks))
        prev_blocks = rev_blocks[1:] + [None]
        for current_block, prev_block in zip(rev_blocks, prev_blocks):
            if prev_block is not None:
                prefetch_modules = prev_block._fsdp_modules
                current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))

    # Handle meta initialization for FSDP2 (fallback if pre-load not done)
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"

    if weights_path is None:
        model.to_empty(device=get_device_type())
        _reset_hf_initialized_flag(model)
        model.init_weights()
    else:
        from torch.distributed.tensor import distribute_tensor

        logger.info_rank0("starting to load model weights...")
        if kwargs.get("broadcast_model_weights_from_rank0"):
            logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
            rank0_load_and_broadcast_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)
        else:
            logger.info_rank0("Every rank would read weights from disk and expect this to be slow!")
            load_model_weights(model, weights_path, get_device_type(), dtensor_factory=distribute_tensor)

    # Register grad norm clipping method for FSDP2
    from .fsdp2 import clip_grad_norm as clip_grad_norm_fn

    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)

    return model


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_full_shard: bool = True,
    enable_shard_grad_op: bool = False,
    use_orig_params: bool = True,
    enable_reshard_after_forward: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu"]:
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

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": use_reentrant,
                "context_fn": kwargs.pop("recompute_context_fn", noop_context_fn),
            },
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            model = parallelize_model_fsdp2(
                model=model,
                weights_path=weights_path,
                enable_full_shard=enable_full_shard,
                enable_reshard_after_forward=enable_reshard_after_forward,
                enable_mixed_precision=enable_mixed_precision,
                basic_modules=basic_modules,
                **kwargs,
            )
        elif parallel_state.dp_mode == "fsdp1":
            model = parallelize_model_fsdp1(
                model=model,
                weights_path=weights_path,
                enable_full_shard=enable_full_shard,
                enable_shard_grad_op=enable_shard_grad_op,
                use_orig_params=use_orig_params,
                enable_mixed_precision=enable_mixed_precision,
                basic_modules=basic_modules,
                **kwargs,
            )
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
