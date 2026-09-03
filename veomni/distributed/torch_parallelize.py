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
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import noop_context_fn

from ..arguments import MixedPrecisionConfig
from ..models import load_model_weights, load_model_weights_ep_sharded, rank0_load_and_broadcast_weights
from ..utils import logging
from ..utils.device import IS_NPU_AVAILABLE, get_device_type
from .checkpoint import CheckpointFunction
from .parallel_plan import ParallelPlan, get_runtime_parallel_plan
from .parallel_state import get_parallel_state
from .torch_compile import CompileConfig, compile_decoder_blocks, validate_compile_runtime
from .utils import check_fqn_match, sort_fqn_by_submodule_first


logger = logging.get_logger(__name__)


def _reset_hf_initialized_flag(module: nn.Module) -> None:
    if hasattr(module, "_is_hf_initialized"):
        module._is_hf_initialized = False
    for child in module.children():
        _reset_hf_initialized_flag(child)


def _to_empty_preserving_nonpersistent_buffers(model: nn.Module, device: str) -> None:
    """Materialize parameters without discarding config-derived buffers.

    ``init_empty_weights()`` patches ``register_parameter`` only, so a
    meta-initialized model still holds real buffer values -- and ``to_empty()``
    replaces every one of them with uninitialized memory. Snapshot them across it
    instead, on the random-init path as much as the resume path: distributed
    checkpoints restore ``state_dict()``, which excludes ``persistent=False``,
    and HF's ``_init_weights`` recomputes a rope table only for a module that
    exposes ``original_inv_freq``. Buffers outside that shape -- Gemma3's
    per-layer-type ``{type}_inv_freq`` and its ``embed_scale``, the Omni audio
    tower's sinusoidal ``positional_embedding`` -- have nothing else to restore
    them. Persistent buffers are left to whichever loader runs next.
    """
    buffers = []
    for module in model.modules():
        for name in module._non_persistent_buffers_set:
            buffer = module._buffers.get(name)
            if buffer is None:
                continue
            if buffer.is_meta:
                # A buffer derived from a parameter (``self.weight.detach()``) is
                # on meta like the parameter, and holds nothing to copy out of.
                # No model does this today; say so rather than let ``to_empty()``
                # leave uninitialized memory behind unannounced.
                logger.warning_rank0(
                    f"Non-persistent buffer {name!r} on {type(module).__name__} is on meta and cannot be "
                    "preserved across materialization; it will hold uninitialized memory unless init_weights() sets it."
                )
                continue
            buffers.append((module, name, buffer.detach().clone()))

    model.to_empty(device=device)

    for module, name, buffer in buffers:
        materialized_buffer = module._buffers[name]
        materialized_buffer.copy_(buffer.to(device=materialized_buffer.device, dtype=materialized_buffer.dtype))


def _has_extra_parallel_plan(model: nn.Module) -> bool:
    """True when the model's parallel plan declares ExtraParallel-sharded tensors.

    The question both ExtraParallel guards below actually need to ask. A plan is
    a property of the *model*, unlike ``ParallelState.any_extra_parallel_enabled``,
    which only says an ep dim exists in the mesh — and a mesh dim is inherited by
    every sub-module of a SeedOmni V2 config, including the ones that have no
    experts.
    """
    plan = get_runtime_parallel_plan(model)
    return plan is not None and bool(getattr(plan, "extra_parallel_plan", None))


def _materialize_and_load_weights(
    model: nn.Module,
    weights_path: Optional[str],
    materialize_device: str,
    *,
    should_skip_hf_weight_load: bool,
    is_peft_model: bool,
    adapter_path: Optional[str],
    broadcast_from_rank0: bool,
    cpu_load_param_name: Optional[List[str]] = None,
    max_load_broadcast_size: float = 20.0,
    fqn_to_index_mapping: Optional[dict] = None,
    ep_sharded_stream_load: bool = False,
) -> None:
    """Move meta-initialized parameters onto a real device and fill them in.

    Shared by the FSDP2 and DDP paths: both build the model on ``meta`` and are
    the only place that materializes it, so the choice between random init, an
    HF snapshot and a checkpoint resume has to be made identically for each.
    """
    # A full non-LoRA checkpoint will overwrite the model, so its resume path can
    # skip expensive HF weight materialization. LoRA checkpoints are trainable-only
    # and still need the HF base weights.
    if should_skip_hf_weight_load and is_peft_model:
        raise ValueError(
            "should_skip_hf_weight_load=True is incompatible with LoRA/PEFT models: the checkpoint is "
            "trainable-only and the frozen base must still be loaded from weights_path."
        )

    if weights_path is None or should_skip_hf_weight_load:
        if should_skip_hf_weight_load:
            logger.info_rank0(
                "Skipping pretrained weight load for checkpoint resume; "
                "parameters will be restored from the distributed checkpoint."
            )
        # Preserve non-persistent buffers on the random-init path too: HF's
        # ``_init_weights`` recomputes only rope tables shaped the way it expects,
        # so the rest would train on whatever ``to_empty()`` left behind.
        _to_empty_preserving_nonpersistent_buffers(model, materialize_device)
        _reset_hf_initialized_flag(model)
        # Random init is unnecessary when the checkpoint will overwrite every parameter.
        if not should_skip_hf_weight_load:
            model.init_weights()
        return

    from torch.distributed.tensor import distribute_tensor

    logger.info_rank0(f"starting to load model weights from {weights_path}...")
    if is_peft_model:
        if adapter_path is not None:
            logger.info_rank0(f"also loading lora adapter weights from {adapter_path}...")
        else:
            logger.info_rank0("also init peft model lora weights...")

    if broadcast_from_rank0:
        logger.info_rank0("Loading model weights from disk on rank0 then broadcasting to other ranks...")
        rank0_load_and_broadcast_weights(
            model,
            weights_path,
            materialize_device,
            dtensor_factory=distribute_tensor,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=max_load_broadcast_size,
            is_peft_model=is_peft_model,
            adapter_path=adapter_path,
            fqn_to_index_mapping=fqn_to_index_mapping,
        )
    else:
        _dt_local_split = partial(distribute_tensor, src_data_rank=None)
        if ep_sharded_stream_load and _has_extra_parallel_plan(model):
            # Opt-in fast/low-memory path for large MoE checkpoints: each rank
            # reads only its ExtraParallel dim-0 slice of the expert tensors
            # straight from the checkpoint (see ``load_model_weights_ep_sharded``).
            # Only valid on the every-rank-reads (non-broadcast) path. An
            # unsupported *checkpoint* still raises ``NotImplementedError``, which
            # we surface directly rather than silently falling back -- an opt-in
            # flag that quietly degrades is hard to reason about.
            logger.info_rank0(
                "Loading model weights via per-rank ExtraParallel-slice streaming (ep_sharded_stream_load)..."
            )
            load_model_weights_ep_sharded(
                model,
                weights_path,
                materialize_device,
                dtensor_factory=_dt_local_split,
                is_peft_model=is_peft_model,
                adapter_path=adapter_path,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )
        else:
            if ep_sharded_stream_load:
                # The flag is set once for a whole run, but this helper runs once
                # per model -- once per OmniModule under SeedOmni V2 -- and a model
                # with no ExtraParallel plan (vision encoder, VAE, connector) has
                # no expert tensors to stream. Skipping is not a silent degrade:
                # there was never a fast path to take here, and refusing instead
                # would make the flag unusable for any heterogeneous model whose
                # MoE backbone wants it.
                logger.info_rank0("Ignoring ep_sharded_stream_load for a model with no ExtraParallel parallel_plan.")
            logger.info_rank0("Every rank would read weights from disk and expect this to be slow!")
            load_model_weights(
                model,
                weights_path,
                materialize_device,
                dtensor_factory=_dt_local_split,
                is_peft_model=is_peft_model,
                adapter_path=adapter_path,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )


def _veomni_shard_placement_fn(param: "nn.Parameter") -> Optional[Shard]:
    """``fully_shard(..., shard_placement_fn=...)`` hook: shard on a model-chosen
    dimension instead of the FSDP2 default ``Shard(0)``.

    A model opts a specific parameter in by tagging it with a
    ``_veomni_fsdp_shard_dim`` attribute in its own ``__init__`` (e.g. DeepSeek-V4's
    compressor/indexer ``position_bias`` -- shape ``(compress_rate, head_dim*k)``
    where ``compress_rate`` can be as small as 4, but the trailing dim is a large,
    reliably-divisible power of 2). Sharding such a param on dim-0 across a large
    FSDP world leaves most ranks with a genuinely empty local shard; redirecting it
    to a dimension that divides evenly avoids that at no memory cost. Returning
    ``None`` for untagged params keeps the FSDP2 default (``Shard(0)``).
    """
    dim = getattr(param, "_veomni_fsdp_shard_dim", None)
    return Shard(dim) if dim is not None else None


def _move_buffers_to_device(model: nn.Module, device: str) -> None:
    """Place every buffer on ``device``.

    ``CPUOffloadPolicy`` only offloads parameters, gradients and optimizer states:
    ``fully_shard`` puts buffers on the FSDP device once at wrap time and never stages
    them again. Meta init defers that move, so a CPU materialization leaves buffers such
    as the DeepSeek-V4 hash-router ``tid2eid`` table or rotary ``inv_freq`` on CPU while
    the activations are on the accelerator.
    """
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.device.type != device:
                module._buffers[name] = buffer.to(device)


def _can_shard_extra_parallel_dim0(
    model: "nn.Module",
    parallel_plan: Optional["ParallelPlan"],
    para_name: str,
    para_fsdp_size: int,
) -> bool:
    """Return whether ``para_name``'s weights may take FSDP ``Shard(0)`` for Muon.

    Zero communication is only reachable for a 3D expert stack that
    ``ParallelPlan`` already slices on dim 0, since that is the sole layout
    ``DistributedMuon`` classifies as ``moe_local_3d``. So the group needs at
    least one such stack; a group of only 2D weights (an ``emb`` plan) would
    gain nothing and must keep the default hidden-dim split. Plans that slice a
    dim other than 0 are rejected outright — dim-0 FSDP sharding would not line
    up with them.

    Every matched param, 2D biases included, still has to divide the FSDP mesh
    on dim 0, because ``shard_placement_fn`` applies one dim to the whole
    wrapped module.

    Plan keys are glob patterns such as ``model.layers.*.mlp.experts.gate_up_proj``,
    so they only resolve through ``check_fqn_match``. The plan must be the runtime
    one (``get_runtime_parallel_plan``) for PEFT-prefixed FQNs to match.
    """
    if parallel_plan is None or not parallel_plan.extra_parallel_plan:
        return False
    para_plan = parallel_plan.extra_parallel_plan.get(para_name)
    if not para_plan:
        return False

    matched = 0
    expert_stacks = 0
    for fqn, param in model.named_parameters():
        for fqn_pattern, shard in para_plan.items():
            if not check_fqn_match(fqn_pattern, fqn):
                continue
            matched += 1
            if shard.dim != 0:
                logger.warning_rank0(
                    f"[muon_expert_zero_comm] {para_name}: {fqn!r} is sliced on dim {shard.dim}; "
                    "Shard(0) only lines up with a dim-0 plan."
                )
                return False
            # ParallelPlan.apply already cut dim 0 down to the local expert count.
            if param.shape[0] % para_fsdp_size != 0:
                logger.warning_rank0(
                    f"[muon_expert_zero_comm] {para_name}: {fqn!r} dim-0 ({param.shape[0]}) is not "
                    f"divisible by {para_name}_fsdp size {para_fsdp_size}."
                )
                return False
            if param.ndim == 3:
                expert_stacks += 1
            break

    if matched == 0:
        logger.warning_rank0(
            f"[muon_expert_zero_comm] {para_name}: no parameter matched the plan patterns {sorted(para_plan)}."
        )
        return False
    if expert_stacks == 0:
        logger.warning_rank0(
            f"[muon_expert_zero_comm] {para_name}: the plan holds no 3D expert stack, "
            "so there is no zero-communication path to enable."
        )
        return False
    return True


def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
    compile_config: Optional[CompileConfig] = None,
    should_skip_hf_weight_load: bool = False,
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

    model._veomni_compile_enabled = False
    compile_config = compile_config or CompileConfig()

    # Step 1: Apply ExtraParallel
    #   e.g. Apply expert parallelism (slice expert tensors [128,H,I] -> [16,H,I])
    #        Apply embed parallelism (slice embed tensors [64,H] -> [16,H])
    if parallel_state.any_extra_parallel_enabled:
        parallel_plan = get_runtime_parallel_plan(model)
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
        persistent_extra_parallel_params = {
            param
            for fqn, param in model.named_parameters()
            if fqn2spec_info[fqn].persistent_fsdp_shard_dim is not None
        }
        persistent_para_names = {
            spec_info.para_name
            for spec_info in fqn2spec_info.values()
            if spec_info.persistent_fsdp_shard_dim is not None
        }
        if persistent_extra_parallel_params:
            if parallel_state.dp_replicate_enabled:
                raise NotImplementedError("Persistent ExtraParallel parameters do not support HSDP replicas yet.")
            if parallel_state.sp_enabled:
                raise NotImplementedError(
                    "Persistent ExtraParallel parameters do not support sequence parallelism yet."
                )
            for para in persistent_para_names:
                para_mesh = parallel_state.extra_parallel_fsdp_device_mesh[para]
                flat_mesh = parallel_state.extra_parallel_flat_device_mesh[para]
                if (
                    para_mesh is None
                    or para_mesh.ndim != 2
                    or para_mesh.size() != parallel_state.world_size
                    or flat_mesh is None
                ):
                    raise NotImplementedError(
                        f"Persistent {para} parameters currently require a 2D ({para}_fsdp, {para}) mesh "
                        "covering the complete world."
                    )
                logger.info_rank0(
                    f"Persistent {para} 2D layout: rows={para_mesh.size(1)} ({para}), "
                    f"columns={para_mesh.size(0)} ({para}_fsdp); parameter all-gather is disabled."
                )
        _extra_parallel_mesh = {}
        _extra_parallel_map = {}
        for para in parallel_state.extra_parallel_names:
            # A globally-enabled ExtraParallel (e.g. ``ep``) may not be declared by
            # THIS model's plan -- e.g. a text-encoder / over-encoder has no experts,
            # so ``ep`` shards nothing here even though a MoE sibling module uses it.
            # Treat such a para as a no-op for this model (``None``) instead of
            # crashing in ``get_extra_parallel_fsdp_no_shard_info`` (which requires
            # >=1 matching module). Nothing downstream wraps it (its layer para_mods
            # stay empty).
            if parallel_state.extra_parallel_enabled(para) and para in parallel_plan.extra_parallel_plan:
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
        parallel_plan = None
        fqn2spec_info = None
        persistent_extra_parallel_params = set()
        persistent_para_names = set()
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
                    # Collect ALL extra-parallel (e.g. ep) modules under this layer,
                    # not just the first. A single layer may hold more than one such
                    # module (e.g. multiple experts modules in one decoder layer), and
                    # EACH must be fully_shard'd over the ep_fsdp mesh. If only the
                    # first were wrapped, the others would fall through to the regular
                    # ``fully_shard(layer_mod)`` below and be re-sharded over the
                    # FSDP mesh -- but their params were already EP-sliced (each rank
                    # holds a *different* [E/ep, ...] slice), so that second shard
                    # scrambles them and corrupts the experts.
                    # Match ``layer_fqn + "."`` (not a bare ``startswith``) so e.g.
                    # ``model.layers.1`` does not also swallow ``model.layers.10``.
                    para_mods = [
                        para_mod
                        for para_mod_fqn, para_mod in _extra_parallel_map[para].items()
                        if para_mod_fqn == layer_fqn or para_mod_fqn.startswith(layer_fqn + ".")
                    ]
                else:
                    para_mods = []
                extra_parallel_mod[para] = para_mods
        layer_pair.append(extra_parallel_mod)
        layer_pairs[layer_fqn] = tuple(layer_pair)

    logger.info_rank0(f"extra_parallel layer pairs: {layer_pairs}")

    if compile_config.enable:
        validate_compile_runtime(
            compile_config,
            device_type=get_device_type(),
            fsdp_enabled=parallel_state.fsdp_enabled,
            fsdp_mode=parallel_state.dp_mode,
            any_extra_parallel_enabled=parallel_state.any_extra_parallel_enabled,
            enable_reshard_after_forward=enable_reshard_after_forward,
        )

        compiled_count = compile_decoder_blocks(
            model,
            compile_config,
            sequence_parallel_enabled=parallel_state.sp_enabled,
            async_enabled=parallel_state.async_enabled,
        )
        if compiled_count == 0:
            raise RuntimeError("train.torch_compile.enable found no decoder blocks to compile.")
        model._veomni_compile_enabled = True
        model._veomni_compile_uses_cuda_graphs = compile_config.uses_cuda_graphs()

    # Step 2: Update fsdp2 kwargs
    fsdp_kwargs = {
        "mesh": parallel_state.fsdp_mesh,
        "reshard_after_forward": enable_reshard_after_forward,
        "shard_placement_fn": _veomni_shard_placement_fn,
    }
    if persistent_extra_parallel_params:
        # These params already carry their complete 2D DTensor placement. Every
        # ancestor FSDP group must ignore them or the parent layer/root would
        # absorb them and reintroduce a parameter all-gather.
        fsdp_kwargs["ignored_params"] = persistent_extra_parallel_params
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
        if persistent_extra_parallel_params:
            raise NotImplementedError(
                "FSDP CPU offload is not supported with persistent ExtraParallel parameters: "
                "FSDP does not manage or stage ignored parameters."
            )
        logger.info_rank0("Enable FSDP2 CPU offload for parameters, gradients, and optimizer states.")
        # ``CPUOffloadPolicy`` defaults to ``pin_memory=True``: the offloaded CPU
        # param shards are page-locked, which (a) is accounted as non-reclaimable
        # ``Shmem`` against the container memcg and (b) cannot be swapped/evicted.
        # For a very large model (e.g. the ~1.5 TiB MoT experts) each rank pins
        # its whole shard, so N ranks/host pin ~N * shard_size of Shmem and OOM
        # the cgroup during load. ``fsdp_offload_pin_memory=False`` keeps the
        # shards in ordinary pageable anon memory (what a bespoke manual offload
        # does) at the cost of a non-pinned (slightly slower) H2D per layer.
        offload_pin_memory = kwargs.pop("fsdp_offload_pin_memory", True)
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=offload_pin_memory)

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
            para_mesh = parallel_state.extra_parallel_fsdp_device_mesh[para]
            para_mesh_dim_names = para_mesh.mesh_dim_names
            # exclude para_size dimension:
            # - (ep_fsdp, ep) -> (ep_fsdp,)
            # - (ep_replicate, ep_fsdp, ep) -> (ep_replicate, ep_fsdp)
            para_fsdp_mesh = para_mesh[para_mesh_dim_names[:-1]]
            para_fsdp_kwargs = dict(fsdp_kwargs)
            para_fsdp_kwargs["mesh"] = para_fsdp_mesh
            shard_dim_for_para = 1
            # Muon zero-comm needs whole experts per rank; otherwise keep the
            # default hidden-dim sharding. A para this model's plan does not
            # declare wraps nothing here, so leave it out of the report.
            declared = parallel_plan is not None and para in (parallel_plan.extra_parallel_plan or {})
            if muon_expert_zero_comm and declared:
                para_fsdp_size = para_mesh[f"{para}_fsdp"].size()
                if _can_shard_extra_parallel_dim0(model, parallel_plan, para, para_fsdp_size):
                    shard_dim_for_para = 0
                    logger.info_rank0(
                        f"[muon_expert_zero_comm] {para}: enabling Shard(0) for "
                        f"the FSDP step ({para}_fsdp size={para_fsdp_size}); Muon will "
                        "run batched NS locally with zero communication."
                    )
                else:
                    logger.warning_rank0(
                        f"[muon_expert_zero_comm] {para}: falling back to the default "
                        "Shard(1) layout (Muon will use the all-to-all-gather path)."
                    )
            para_fsdp_kwargs["shard_placement_fn"] = lambda param, _d=shard_dim_for_para: Shard(_d)
            extra_parallel_fsdp_kwargs[para] = para_fsdp_kwargs
            # Record the FSDP shard dim on the spec_info so the checkpointer can
            # build correct DTensor placements (EP dim vs FSDP dim) on save/load.
            for spec_info in getattr(model, "_fqn2spec_info", {}).values():
                if spec_info.para_name == para:
                    spec_info.fsdp_shard_dim = shard_dim_for_para
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
            # para (e.g. ep, emb) enabled and this layer contains the para module(s)
            # (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens). A layer
            # may hold more than one (e.g. multiple experts modules) -- wrap each.
            if not parallel_state.extra_parallel_enabled(para):
                continue
            for _para_mod in extra_parallel_mod[para]:
                if isinstance(_para_mod, FSDPModule):
                    continue
                if any(param in persistent_extra_parallel_params for param in _para_mod.parameters()):
                    # Persistent params remain on the complete 2D mesh. Any
                    # non-persistent siblings are picked up by the enclosing
                    # decoder/root FSDP group through the normal bottom-up walk.
                    continue
                # shard para module (e.g. expert/decoder.moe, embed_tokens/decoder.embed_tokens)
                fully_shard(_para_mod, **extra_parallel_fsdp_kwargs[para])
                # average para (e.g. ep) grads across para (e.g. ep) ranks
                # NOTE: in torch 2.8 and later we should use
                # experts_mod.set_gradient_divide_factor(parallel_state.ep_size)
                # but for torch 2.7 we still use set_reduce_scatter_divide_factor(parallel_state.ep_size)
                gradient_divide_factor = parallel_state.extra_parallel_gradient_divide_factor(para)
                logger.info(f"setting grad divide factor for {para} module to {gradient_divide_factor}")
                if IS_NPU_AVAILABLE:
                    # NPU is using torch 2.7
                    _para_mod.set_reduce_scatter_divide_factor(gradient_divide_factor)
                else:
                    # from torch 2.8
                    _para_mod.set_gradient_divide_factor(gradient_divide_factor)
                layer_mod._fsdp_modules.append(_para_mod)

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

    # Shard root WITHOUT explicit `reshard_after_forward` so FSDP2's
    # auto-no-reshard for the root kicks in (`_fsdp_state.py:_lazy_init`).
    # This keeps the root's params (lm_head + embeddings + final norm)
    # unsharded between forward and backward, which is what fused-linear
    # kernels (chunk_logprobs / chunk_topk_distill) need — they save the
    # lm_head weight via `save_for_backward` and would otherwise hit
    # `setStorage … storage of size 0` when the saved reference points
    # to a freed buffer. Decoder layers reshard normally (their calls
    # above pass `reshard_after_forward` explicitly).
    root_fsdp_kwargs = {k: v for k, v in fsdp_kwargs.items() if k != "reshard_after_forward"}
    fully_shard(model, **root_fsdp_kwargs)
    # Persistent ExtraParallel parameters are ignored by every FSDP group and
    # retain their original 2D DTensor identity. Record them after wrapping so
    # gradient clipping can reduce their unique local shards once over the
    # flattened 2D mesh instead of traversing the two mesh axes separately.
    model._persistent_extra_parallel_param_ids = {id(param) for param in persistent_extra_parallel_params}

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

    # Handle meta initialization for FSDP2 (fallback if pre-load not done)
    assert kwargs.get("init_device") == "meta", "Please use init_device: meta for FSDP2"
    materialize_device = "cpu" if enable_fsdp_cpu_offload else get_device_type()

    _materialize_and_load_weights(
        model,
        weights_path,
        materialize_device,
        should_skip_hf_weight_load=should_skip_hf_weight_load,
        is_peft_model=kwargs.pop("is_peft_model", False),
        adapter_path=kwargs.pop("adapter_path", None),
        broadcast_from_rank0=bool(kwargs.get("broadcast_model_weights_from_rank0")),
        cpu_load_param_name=kwargs.get("cpu_load_param_name", None),
        max_load_broadcast_size=kwargs.get("max_load_broadcast_size", 20.0),
        fqn_to_index_mapping=kwargs.get("fqn_to_index_mapping"),
        ep_sharded_stream_load=bool(kwargs.get("ep_sharded_stream_load")),
    )

    if materialize_device == "cpu":
        _move_buffers_to_device(model, get_device_type())

    # Register grad norm clipping method for FSDP2
    from .fsdp2 import clip_grad_norm as clip_grad_norm_fn

    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_fn, model)

    return model


def parallelize_model_ddp(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    should_skip_hf_weight_load: bool = False,
    **kwargs,
) -> "nn.Module":
    """Replicate the model with DDP, materializing it first under meta-init.

    DDP keeps a full replica of plain tensors per rank: it registers gradient
    hooks and broadcasts rank0's parameters at construction, but it neither
    materializes meta parameters nor loads weights. Under meta-init nothing else
    does either — ``build_model`` returns an empty model and ``BaseTrainer`` has
    no load step of its own — so the wrap used to reach DDP's constructor with
    meta parameters and die there on ``Tensor.item()``. A model that arrives with
    real weights already loaded must not be touched here.
    """
    parallel_state = get_parallel_state()

    # Only fsdp2 applies the ExtraParallel plan that shards expert weights, so a
    # DDP module's experts are whole. Refuse rather than load full tensors into a
    # model whose plan says they are sharded -- until now the meta crash below hid
    # this combination. Keyed on the model's plan, not on
    # ``any_extra_parallel_enabled``: a SeedOmni V2 sub-module inherits the global
    # accelerator's ep dim whether or not it owns any experts, so the mesh alone
    # would refuse a plan-less DDP vision tower that is perfectly fine.
    if parallel_state.any_extra_parallel_enabled and _has_extra_parallel_plan(model):
        raise RuntimeError(
            "ExtraParallel (ep) requires fsdp_mode='fsdp2'; DDP does not apply the parallel plan that shards experts."
        )

    # Ask the parameters, not ``init_device``: the flag states an intent a model
    # builder is free to ignore -- the data tests construct their model eagerly
    # while the config still says ``meta`` -- and materializing a model that
    # already holds real weights would discard them, or raise outright on a plain
    # nn.Module with no ``init_weights``.
    if any(param.is_meta for param in model.parameters()):
        _materialize_and_load_weights(
            model,
            weights_path,
            get_device_type(),
            should_skip_hf_weight_load=should_skip_hf_weight_load,
            is_peft_model=kwargs.pop("is_peft_model", False),
            adapter_path=kwargs.pop("adapter_path", None),
            # The flag was fsdp2-only while DDP loaded nothing at all. Now that
            # this path loads, it applies verbatim: ``rank0_load_and_broadcast_weights``
            # broadcasts over the default (world) group from global rank0, and a
            # DDP replica wants exactly that whole tensor. Honouring it is what
            # keeps a DDP run off the every-rank-reads path, which the flag
            # defaults to avoiding.
            broadcast_from_rank0=bool(kwargs.get("broadcast_model_weights_from_rank0")),
            cpu_load_param_name=kwargs.get("cpu_load_param_name", None),
            max_load_broadcast_size=kwargs.get("max_load_broadcast_size", 20.0),
            fqn_to_index_mapping=kwargs.get("fqn_to_index_mapping"),
        )

        # Anything the loader left behind would otherwise surface as
        # ``Tensor.item() cannot be called on meta tensors`` from inside DDP's
        # constructor, which names neither the parameter nor the cause.
        unmaterialized = [name for name, param in model.named_parameters() if param.is_meta]
        if unmaterialized:
            raise RuntimeError(
                f"DDP received unmaterialized parameters after loading from {weights_path!r}: "
                f"{unmaterialized[:5]}{'...' if len(unmaterialized) > 5 else ''}"
            )

    # ``broadcast_buffers=False`` because FSDP2 syncs no buffers at all, and a
    # module's buffer semantics must not change with the ``fsdp_mode`` a config
    # happened to pick. Nothing is lost: rank0's copy is either identical to the
    # others or, for dynamic-rope ``inv_freq``, wrong for them. See constraint 7a
    # in `.agents/knowledge/constraints.md`.
    return DDP(
        model,
        device_ids=[parallel_state.local_rank],
        process_group=parallel_state.dp_group,
        broadcast_buffers=False,
    )


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_reshard_after_forward: bool = True,
    mixed_precision: MixedPrecisionConfig = MixedPrecisionConfig(enable=True),  # noqa
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    muon_expert_zero_comm: bool = False,
    compile_config: Optional[CompileConfig] = None,
    should_skip_hf_weight_load: bool = False,
    **kwargs,
) -> "nn.Module":
    """Apply parallel strategies to the model.

    Args:
        muon_expert_zero_comm: Shard ExtraParallel weights on dim-0 when the
            EP-local dim is divisible by ``ep_fsdp_size``.
    """
    parallel_state = get_parallel_state()
    compile_config = compile_config or CompileConfig()

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") not in ["cuda", "npu", "mlu"]:
            raise ValueError("Only FSDP training supports `init_device=meta`.")

    if mixed_precision.enable:  # upcast to float32 before feed it to optimizer
        model = model.float()

    checkpoint_early_stop = kwargs.pop("early_stop", True)
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        gradient_checkpointing_kwargs = {
            "use_reentrant": use_reentrant,
            "context_fn": kwargs.pop("recompute_context_fn", noop_context_fn),
        }
        if not use_reentrant:
            gradient_checkpointing_kwargs["early_stop"] = checkpoint_early_stop

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
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
                compile_config=compile_config,
                should_skip_hf_weight_load=should_skip_hf_weight_load,
                **kwargs,
            )
        else:
            if compile_config.enable:
                raise RuntimeError("train.torch_compile.enable requires fsdp_mode='fsdp2'; DDP is not supported.")
            model = parallelize_model_ddp(
                model=model,
                weights_path=weights_path,
                should_skip_hf_weight_load=should_skip_hf_weight_load,
                **kwargs,
            )
    elif compile_config.enable:
        raise RuntimeError("train.torch_compile.enable requires FSDP2; compile without FSDP is not supported.")

    return model
