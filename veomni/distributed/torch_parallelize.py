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
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import noop_context_fn

from ..arguments import MixedPrecisionConfig
from ..models import load_model_weights, rank0_load_and_broadcast_weights
from ..utils import logging
from ..utils.device import IS_NPU_AVAILABLE, get_device_type
from .checkpoint import CheckpointFunction
from .parallel_state import get_parallel_state
from .utils import sort_fqn_by_submodule_first


logger = logging.get_logger(__name__)


def _reset_hf_initialized_flag(module: nn.Module) -> None:
    if hasattr(module, "_is_hf_initialized"):
        module._is_hf_initialized = False
    for child in module.children():
        _reset_hf_initialized_flag(child)


def _resolve_weights_path_mapping(
    model: nn.Module,
    weights_path: Mapping[str, str],
) -> List[Tuple[str, nn.Module, str]]:
    """Resolve a ``{sub_module_name: snapshot_path}`` mapping against the
    model's direct children.

    Strict bijection: the mapping's keys must equal the set of
    ``model.named_children()`` names exactly.  This matches the contract
    OmniModel exposes after the D2.2 small refactor — every declared
    sub-module is a direct attribute, not nested under a ``modules_dict.``
    middle layer.

    Returns
    -------
    ``[(name, child, path), ...]`` in the mapping's declared order.

    Raises
    ------
    KeyError if (a) any mapping key is not a direct child of ``model``, or
    (b) any direct child of ``model`` is missing from the mapping.  Either
    side of the mismatch is named in the error so YAML typos / omissions
    surface immediately — the caller can't accidentally end up with a
    silently un-initialised sub-module.
    """
    children = dict(model.named_children())
    unknown = [name for name in weights_path if name not in children]
    missing = [name for name in children if name not in weights_path]
    if unknown or missing:
        msg_parts = [f"weights_path mapping mismatch on {type(model).__name__}:"]
        if unknown:
            msg_parts.append(f"unknown key(s) {sorted(unknown)!r} (not a direct child)")
        if missing:
            msg_parts.append(f"missing key(s) {sorted(missing)!r} (child has no path)")
        msg_parts.append(f"available children: {sorted(children)}")
        raise KeyError(" ".join(msg_parts))

    return [(name, children[name], weights_path[name]) for name in weights_path]


def _apply_weights_load_step(
    model: nn.Module,
    weights_path: Optional[Union[str, Mapping[str, str]]],
    materialize_device: str,
    *,
    broadcast_from_rank0: bool,
    is_peft_model: bool,
    adapter_path: Optional[str],
    cpu_load_param_name: Optional[List[str]],
    max_load_broadcast_size: float,
    distribute_tensor_fn: Callable[..., Any],
) -> None:
    """Materialise meta-initialised parameters and (optionally) load weights.

    Three behaviours, dispatched by the type of ``weights_path``:

    * ``None`` — call ``model.to_empty(device=...)`` then ``model.init_weights()``
      (HF random init).  Behaviourally identical to the pre-D2.2 inline
      block in ``parallelize_model_fsdp2``.
    * ``str`` — load the entire ``model`` from the single snapshot at the
      given path via ``rank0_load_and_broadcast_weights`` (when broadcast
      is enabled) or ``load_model_weights`` (every-rank-reads fallback).
      Behaviourally identical to the pre-D2.2 inline block; this is the
      path BaseTrainer / VLMTrainer / TextTrainer / DiTTrainer all hit.
    * ``Mapping[str, str]`` — strict bijection over ``model.named_children()``:
      every direct child must have an entry, every entry must name a
      direct child.  Each ``(sub_module_name, sub_path)`` pair is loaded
      via the same ``rank0_load_and_broadcast_weights`` / ``load_model_weights``
      pair as the ``str`` branch, scoped to the corresponding child
      subtree.  This is the path OmniTrainer (D2.3) will use to load
      each named sub-module from its own HF snapshot.

    Notes
    -----
    * The ``str`` / ``None`` branches preserve the pre-D2.2 control flow
      and call sites — ``rank0_load_and_broadcast_weights`` and
      ``load_model_weights`` are invoked with the same args in the same
      order.  A small structural cleanup (eager ``distribute_tensor``
      import; ``kwargs.pop`` of ``is_peft_model`` / ``adapter_path``
      lifted to the caller) is functionally a no-op for those paths.
    * The ``Mapping`` branch deliberately rejects ``is_peft_model=True``:
      the existing PEFT loader (``model_utils.load_model_weights``) maps
      a single base-model checkpoint into a single PEFT-wrapped model,
      and there is no sane interpretation of "PEFT" applied uniformly
      across a heterogeneous set of OmniModel sub-modules.  D-track will
      design per-sub-module PEFT once V2 trainers settle; refusing the
      combination today prevents silent miswiring.
    * Calling ``load_model_weights`` on a child sub-tree of an FSDP-wrapped
      root mirrors the pre-D2.2 ``str`` path's behaviour against
      ``model.to_empty(...)`` on the wrapped root — both rely on
      ``to_empty`` propagating through FSDP DTensor parameters.  This
      composition is exercised end-to-end by D2.3's smoke tests (which
      run real FSDP) — the unit tests here only cover dispatch.
    """
    if weights_path is None:
        model.to_empty(device=materialize_device)
        _reset_hf_initialized_flag(model)
        model.init_weights()
        return

    if isinstance(weights_path, Mapping):
        if is_peft_model:
            raise NotImplementedError(
                "weights_path: Mapping[str, str] is incompatible with is_peft_model=True. "
                "PEFT for V2 OmniModel sub-modules will be designed in a follow-up; "
                "for now, pass a single str weights_path or disable PEFT."
            )
        loaded = _resolve_weights_path_mapping(model, weights_path)
        for name, child, sub_path in loaded:
            logger.info_rank0(f"Loading weights for sub-module {name!r} from {sub_path}...")
            _load_one(
                model=child,
                weights_path=sub_path,
                materialize_device=materialize_device,
                broadcast_from_rank0=broadcast_from_rank0,
                is_peft_model=False,
                adapter_path=None,
                cpu_load_param_name=cpu_load_param_name,
                max_load_broadcast_size=max_load_broadcast_size,
                distribute_tensor_fn=distribute_tensor_fn,
            )
        return

    logger.info_rank0(f"starting to load model weights from {weights_path}...")
    if is_peft_model:
        if adapter_path is not None:
            logger.info_rank0(f"also loading lora adapter weights from {adapter_path}...")
        else:
            logger.info_rank0("also init peft model lora weights...")
    _load_one(
        model=model,
        weights_path=weights_path,
        materialize_device=materialize_device,
        broadcast_from_rank0=broadcast_from_rank0,
        is_peft_model=is_peft_model,
        adapter_path=adapter_path,
        cpu_load_param_name=cpu_load_param_name,
        max_load_broadcast_size=max_load_broadcast_size,
        distribute_tensor_fn=distribute_tensor_fn,
    )


def _load_one(
    *,
    model: nn.Module,
    weights_path: str,
    materialize_device: str,
    broadcast_from_rank0: bool,
    is_peft_model: bool,
    adapter_path: Optional[str],
    cpu_load_param_name: Optional[List[str]],
    max_load_broadcast_size: float,
    distribute_tensor_fn: Callable[..., Any],
) -> None:
    """Materialise + load weights into one (sub-)tree from a single snapshot.

    Centralises the rank0-broadcast vs. every-rank-reads choice so both
    the ``str`` and ``Mapping`` branches above share identical loader
    behaviour.  ``model`` here may be the full top-level model or a
    single named child — both are valid ``nn.Module`` instances.
    """
    if broadcast_from_rank0:
        logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
        rank0_load_and_broadcast_weights(
            model,
            weights_path,
            materialize_device,
            dtensor_factory=distribute_tensor_fn,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=max_load_broadcast_size,
            is_peft_model=is_peft_model,
            adapter_path=adapter_path,
        )
    else:
        logger.info_rank0("Every rank would read weights from disk and expect this to be slow!")
        _dt_local_split = partial(distribute_tensor_fn, src_data_rank=None)
        load_model_weights(
            model,
            weights_path,
            materialize_device,
            dtensor_factory=_dt_local_split,
            is_peft_model=is_peft_model,
            adapter_path=adapter_path,
        )


def _check_extra_parallel_dim0_divisibility(model: "nn.Module", para_name: str, ep_fsdp_size: int) -> bool:
    """Return whether EP-local dim-0 can be evenly sharded by ``ep_fsdp_size``."""
    parallel_plan = getattr(model, "get_parallel_plan", None)
    if parallel_plan is None:
        return False
    plan = parallel_plan()
    if plan is None or plan.extra_parallel_plan is None:
        return False
    para_plan = plan.extra_parallel_plan.get(para_name)
    if not para_plan:
        return False

    for fqn in para_plan.keys():
        param = dict(model.named_parameters()).get(fqn)
        if param is None:
            continue
        if param.ndim < 1:
            continue
        local_n = param.shape[0]
        if local_n % ep_fsdp_size != 0:
            logger.warning_rank0(
                f"[muon_expert_zero_comm] param {fqn!r} dim-0 ({local_n}) is not "
                f"divisible by ep_fsdp_size={ep_fsdp_size}; cannot use Shard(0)."
            )
            return False
    return True


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[Union[str, Mapping[str, str]]] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
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

    Args:
        weights_path: One of three forms:
            * ``None`` — random init (``to_empty + init_weights``).
            * ``str`` — single HF snapshot path; loads the entire ``model``.
              All existing single-model trainers (BaseTrainer / VLMTrainer /
              TextTrainer / DiTTrainer) hit this branch.
            * ``Mapping[str, str]`` — ``{sub_module_name: snapshot_path}``,
              keyed by the model's direct ``named_children()`` names.
              **Strict bijection** — every direct child must appear as a
              key, every key must name a direct child; mismatches raise
              ``KeyError``.  Each named child is loaded from its own
              snapshot via the same loader the ``str`` branch uses.
              Used by V2 OmniModel (D2.3), where each declared sub-module
              owns its own HF folder.  Incompatible with
              ``is_peft_model=True`` — raises ``NotImplementedError``.
    """

    parallel_state = get_parallel_state()

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

        model._fqn2spec_info = fqn2spec_info
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
    if mixed_precision.enable:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=getattr(torch, mixed_precision.param_dtype) if mixed_precision.param_dtype else None,
            reduce_dtype=getattr(torch, mixed_precision.reduce_dtype) if mixed_precision.reduce_dtype else None,
            output_dtype=getattr(torch, mixed_precision.output_dtype) if mixed_precision.output_dtype else None,
            cast_forward_inputs=mixed_precision.cast_forward_inputs,
        )
        fsdp_kwargs["mp_policy"] = mp_policy
    # prepare offload_policy kwargs
    enable_fsdp_cpu_offload = kwargs.pop("enable_fsdp_offload", False)
    model._fsdp_cpu_offload_enabled = enable_fsdp_cpu_offload
    if enable_fsdp_cpu_offload:
        logger.info_rank0("Enable FSDP2 CPU offload for parameters, gradients, and optimizer states.")
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

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
            shard_dim_for_para = 1
            # Muon zero-comm needs whole experts per rank; otherwise keep the
            # default hidden-dim sharding.
            if muon_expert_zero_comm:
                ep_fsdp_size = parallel_state.extra_parallel_fsdp_size(para)
                divisible = _check_extra_parallel_dim0_divisibility(model, para, ep_fsdp_size)
                if divisible:
                    shard_dim_for_para = 0
                    logger.info_rank0(
                        f"[muon_expert_zero_comm] {para}: enabling Shard(0) for "
                        f"the FSDP step (ep_fsdp_size={ep_fsdp_size}); Muon will "
                        "run batched NS locally with zero communication."
                    )
                else:
                    logger.warning_rank0(
                        f"[muon_expert_zero_comm] {para}: divisibility check failed "
                        f"(ep_fsdp_size={ep_fsdp_size}); falling back to default "
                        "Shard(1) layout (Muon will use the all-to-all-gather path)."
                    )
            para_fsdp_kwargs["shard_placement_fn"] = lambda param, _d=shard_dim_for_para: Shard(_d)
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
        from veomni.ops.platform.npu import apply_hccl_premul_sum_patch

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
    need_manual_prefetch = (
        parallel_state.any_extra_parallel_enabled or mp_ignored_classes is not None
    ) and kwargs.pop("enable_forward_prefetch", True)
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

    # Handle meta initialization for FSDP2 (fallback if pre-load not done).
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"
    materialize_device = "cpu" if enable_fsdp_cpu_offload else get_device_type()

    from torch.distributed.tensor import distribute_tensor

    _apply_weights_load_step(
        model=model,
        weights_path=weights_path,
        materialize_device=materialize_device,
        broadcast_from_rank0=bool(kwargs.get("broadcast_model_weights_from_rank0")),
        is_peft_model=kwargs.pop("is_peft_model", False),
        adapter_path=kwargs.pop("adapter_path", None),
        cpu_load_param_name=kwargs.get("cpu_load_param_name", None),
        max_load_broadcast_size=kwargs.get("max_load_broadcast_size", 20.0),
        distribute_tensor_fn=distribute_tensor,
    )

    # Register grad norm clipping method for FSDP2
    from .fsdp2 import clip_grad_norm as clip_grad_norm_fn

    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)

    return model


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[Union[str, Mapping[str, str]]] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
    **kwargs,
) -> "nn.Module":
    """Apply parallel strategies to the model.

    Args:
        weights_path: One of three forms:
            * ``None`` — random init under FSDP2.
            * ``str`` — single HF snapshot for the whole model (the path
              all single-model trainers — BaseTrainer / VLMTrainer /
              TextTrainer / DiTTrainer — hit today).
            * ``Mapping[str, str]`` — ``{sub_module_name: snapshot_path}``
              keyed by ``model.named_children()`` (strict bijection;
              missing or extra keys raise ``KeyError``).  Used by V2
              OmniModel (D2.3) to load each declared sub-module from its
              own folder.  See :func:`parallelize_model_fsdp2` for details.
        muon_expert_zero_comm: Shard ExtraParallel weights on dim-0 when the
            EP-local dim is divisible by ``ep_fsdp_size``.
    """

    parallel_state = get_parallel_state()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu"]:
            raise ValueError("Only FSDP training supports `init_device=meta`.")

    if mixed_precision.enable:  # upcast to float32 before feed it to optimizer
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
                enable_reshard_after_forward=enable_reshard_after_forward,
                mixed_precision=mixed_precision,
                basic_modules=basic_modules,
                muon_expert_zero_comm=muon_expert_zero_comm,
                **kwargs,
            )
        else:
            model = DDP(model, device_ids=[parallel_state.local_rank], process_group=parallel_state.dp_group)

    return model
