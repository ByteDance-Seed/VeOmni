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


import functools
import sys
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..arguments.arguments_types import OpsImplementationConfig
from ..distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from ..ops.dispatch import OpsConfigSlot, OpSlot
from ..utils import logging
from ..utils.device import is_torch_npu_available
from .loader import BaseModelLoader, get_loader, get_model_config, get_model_processor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)

# Models whose forward implements context parallelism. An allow-list rather than
# a deny-list so that a model has to be ported deliberately: CP is enabled
# model-agnostically (``ParallelState`` and ``TrainingArguments`` both admit
# ``cp_size > 1`` for anything), and nothing downstream would notice a model that
# has not been. ``SequenceParallelCollator`` shards the sequence on
# ``sp_enabled``, which CP alone turns on, while every unported model gates its
# sequence-parallel collectives on ``ulysses_enabled``, which CP alone leaves
# false — so the shards are never gathered, each rank attends only within its own
# 1/cp_size of the sequence, and the run trains to a plausible loss curve while
# being silently wrong.
CONTEXT_PARALLEL_MODEL_TYPES = frozenset({"deepseek_v4"})

# Models whose forward implements the Lightning Indexer KL objective. An
# allow-list for the same reason as above, and against the same failure: the flag
# is on the model-agnostic ``OpsImplementationConfig``, but only DeepSeek-V4's
# patch config declares the slot and only its forward builds the KL. GLM MoE DSA
# has a Lightning Indexer of its own and declares the two neighbouring DSA slots,
# so ``dsa_indexer_loss: true`` on a GLM run is an entirely reasonable thing for a
# user to write -- and produces no error, no metric and no training. Every other
# unsupported configuration of this objective is refused loudly; this one has to
# be too, and it can only be refused from outside the model, because a GLM run
# never reaches DeepSeek-V4's patched forward where the other two gates live.
INDEXER_LOSS_MODEL_TYPES = frozenset({"deepseek_v4"})


def check_indexer_loss_supported(config: PretrainedConfig) -> None:
    """Raise unless this model type implements the Lightning Indexer KL objective.

    A no-op when ``dsa_indexer_loss`` is off, which is the default and every other
    configuration. Reads the installed ops singleton rather than taking the config
    as an argument, so that every construction path is covered by the same call.

    "Off" includes a non-positive coefficient, matching ``OpsImplementationConfig``
    and the runtime gate in DeepSeek-V4's forward, both of which read the weight
    before anything else and treat zero as the objective being switched off rather
    than as a configuration of it. Without that agreement here, a shared config that
    parks ``dsa_indexer_loss: true`` and disables the term with the documented
    coefficient-only switch would still be refused on every other model type -- and
    refused for enabling something it just turned off.
    """
    from ..ops.config.singleton import get_ops_config

    ops_config = get_ops_config()
    if ops_config is None or not getattr(ops_config, "dsa_indexer_loss", False):
        return
    if getattr(ops_config, "dsa_indexer_loss_coef", 1.0) <= 0:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in INDEXER_LOSS_MODEL_TYPES:
        return

    supported = ", ".join(sorted(INDEXER_LOSS_MODEL_TYPES))
    raise NotImplementedError(
        f"dsa_indexer_loss is not implemented for model type {model_type!r}; only {supported} "
        "builds the indexer KL. Left to run it would train nothing, report nothing and cost "
        "nothing -- a silent no-op rather than an error. Set dsa_indexer_loss=false, or train "
        f"one of: {supported}."
    )


def check_context_parallel_supported(config: PretrainedConfig) -> None:
    """Raise unless this model type implements context parallelism.

    A no-op when context parallelism is off, which is every other configuration.
    """
    # ``build_foundation_model`` runs this for every model, including in processes
    # that never installed a parallel state -- tests that spawn a multi-rank world
    # and build a model directly do exactly that. Asking ``get_parallel_state``
    # there would *construct* a default single-process state, whose ``dp_size=1``
    # contradicts the real world size and raises on the topology check. No state
    # installed means no context parallelism to gate.
    if not is_parallel_state_initialized():
        return

    if not get_parallel_state().cp_enabled:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in CONTEXT_PARALLEL_MODEL_TYPES:
        return

    supported = ", ".join(sorted(CONTEXT_PARALLEL_MODEL_TYPES))
    raise NotImplementedError(
        f"Context parallelism is not implemented for model type {model_type!r}; "
        f"only {supported} supports it. Set cp_size=1 to disable it, or use "
        "ulysses_size for sequence parallelism on this model."
    )


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str, **kwargs) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return get_model_processor(processor_path, padding_side="right", trust_remote_code=True, **kwargs)


def build_config(config_path: str, **config_kwargs) -> "PretrainedConfig":
    """
    Builds the model config.
    """
    trust_remote_code = config_kwargs.pop("trust_remote_code", True)
    return get_model_config(config_path, trust_remote_code=trust_remote_code, **config_kwargs)


def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig) -> bool:
    """Bind every OpSlot in *modeling_module* from *ops_config*.

    Returns ``True`` if at least one OpSlot was found (and bound).
    """
    bound: list[str] = []
    moe_experts_kernel: Optional[str] = None
    for name in dir(modeling_module):
        obj = getattr(modeling_module, name, None)
        if isinstance(obj, OpsConfigSlot):
            obj.bind(ops_config)
            bound.append(f"{obj.field_name} ({obj.value})")
            continue
        if not isinstance(obj, OpSlot):
            continue
        # ``moe_experts`` reads ``moe_implementation`` (not the
        # ``{op}_implementation`` convention) and its values carry a
        # ``fused_`` prefix the registry entries don't. Translate so the
        # registry lookup finds the kernel and the HardwareRequirement
        # check fires.
        if obj.op_name == "moe_experts":
            impl_name = (
                "eager"
                if ops_config.moe_implementation == "eager"
                else ops_config.moe_implementation.removeprefix("fused_")
            )
            if impl_name != "eager" and obj.variant == "standard":
                moe_experts_kernel = impl_name
        else:
            impl_name = getattr(ops_config, f"{obj.op_name}_implementation", "eager")
        obj.bind(impl_name)
        bound.append(f"{obj.op_name} ({impl_name})")

    # OpSlot is just an eager-vs-fused guard; inside the fused branch the
    # generated modeling code dispatches through the module-level pointer
    # ``veomni.ops.kernels.moe._fused_moe_forward``. Keep the pointer in
    # sync with the slot's bound kernel; eager leaves it untouched.
    if moe_experts_kernel is not None:
        from ..ops.kernels.moe import apply_veomni_fused_moe_patch

        apply_veomni_fused_moe_patch(fused_moe_kernel=moe_experts_kernel)

    if bound:
        logger.info_rank0(f"OpSlot dispatch bound: {', '.join(bound)}.")
    return bool(bound)


def _validate_attention_parallelism(attn_implementation: Optional[str]) -> None:
    if attn_implementation not in ("magi_attention", "veomni_magi_attention_with_sp"):
        return

    cp_size = get_parallel_state().cp_size
    if cp_size != 1:
        raise ValueError(
            f"MagiAttention currently requires context parallel size 1 (cp_size == 1), got cp_size={cp_size}."
        )


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    # ``None`` = "no caller preference" — resolved from ``ops_implementation``
    # below. A non-None value is treated as an explicit caller override and
    # preserved as-is (used by callers that pre-installed the ops singleton
    # via ``apply_ops_config`` and only need to pin attn here).
    attn_implementation: Optional[
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "flex_attention",
            "magi_attention",
            "veomni_flex_attention_with_sp",
            "veomni_magi_attention_with_sp",
            "veomni_flash_attention_2_with_sp",
            "veomni_flash_attention_3_with_sp",
            "veomni_flash_attention_4_with_sp",
            "native-sparse",
        ]
    ] = None,
    init_device: Literal["cpu", "cuda", "npu", "mlu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    encoder_data_balance: Optional[bool] = False,
    encoder_data_balance_sorting_algo: Optional[str] = "post_mbs_balancing_greedy_without_pad",
    ops_implementation: Optional[OpsImplementationConfig] = None,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.

    Ops dispatch: callers must pass ``ops_implementation`` *or* pre-install a
    singleton via ``apply_ops_config(...)``. There is no silent all-eager
    fallback — passing neither raises ``ValueError``. Trainers pass
    ``args.model.ops_implementation``; standalone scripts (``tasks/infer/*``)
    construct an explicit ``OpsImplementationConfig``. ``DiTTrainer`` builds a
    condition model first via ``apply_ops_config`` and then calls into here
    without the kwarg, which is fine — the singleton-already-installed branch
    leaves it alone.
    """
    from ..ops import apply_ops_config
    from ..ops.config.singleton import get_ops_config

    if ops_implementation is not None:
        attn_implementation = ops_implementation.attn_implementation
        _validate_attention_parallelism(attn_implementation)
        apply_ops_config(ops_implementation)
    else:
        installed = get_ops_config()
        if installed is None:
            raise ValueError(
                "build_foundation_model requires `ops_implementation` (or a prior "
                "`apply_ops_config(...)` call). Trainers pass "
                "`args.model.ops_implementation`; standalone scripts must "
                "construct an `OpsImplementationConfig` explicitly. "
                "There is no longer a silent all-eager fallback."
            )
        # Caller pre-installed the singleton (e.g. DiTTrainer building a
        # condition model). Honour the installed config's attn unless the
        # caller passed an explicit override — without this, the model
        # silently uses the loader's HF default instead of the SP-aware
        # variant the user selected.
        if attn_implementation is None:
            attn_implementation = installed.attn_implementation
        _validate_attention_parallelism(attn_implementation)

    if config_kwargs is None:
        config_kwargs = {}

    if isinstance(config_path, PretrainedConfig):
        config = config_path
    else:
        config = build_config(config_path, **config_kwargs)

    check_context_parallel_supported(config)
    check_indexer_loss_supported(config)

    if encoder_data_balance:
        if config.model_type == "qwen3_vl_moe":
            if get_parallel_state().sp_enabled:
                logger.warning_rank0(
                    "Warning: Qwen3VLEncoderDataBalance currently does not support sequence parallelism. "
                    "The configuration of 'encoder_data_balance' is reset to False. "
                    "This issue will be addressed in a future release."
                )
                config.encoder_data_balance = False
            else:
                config.encoder_data_balance = encoder_data_balance
                config.encoder_data_balance_sorting_algo = encoder_data_balance_sorting_algo
        else:
            logger.warning_rank0(
                f"Encoder data balance currently supported only for Qwen3-VL MoE, "
                f"current model type: {config.model_type}, reset encoder_data_balance = False"
            )
            config.encoder_data_balance = False
    else:
        config.encoder_data_balance = False

    loader: Optional[BaseModelLoader] = get_loader(config)

    # ── Pre-init: OpSlot binding ──────────────────────────────────────────
    # ``get_loader`` -> ``get_model_class`` -> ``MODELING_REGISTRY[...]()``
    # has already imported the patched modeling module, so ``loader.model_cls``
    # is in ``sys.modules`` and we can resolve OpSlot bindings *before*
    # the model is constructed. This matters for slots consumed inside
    # ``__init__`` (e.g. Qwen3.5's GatedDeltaNet picks between
    # ``Qwen3_5RMSNormGated`` and ``FusedRMSNormGated`` at init time based
    # on ``veomni_rms_norm_gated.use_non_eager_impl``); slots consumed only
    # in ``forward`` would also work post-init, but binding once, here, keeps
    # the timing uniform. Assumes ``loader.model_cls`` is final at this point —
    # i.e. no loader rewrites it between here and ``loader.load_model()`` below.
    model_cls = getattr(loader, "model_cls", None) if loader is not None else None
    modeling_module = sys.modules.get(model_cls.__module__) if model_cls is not None else None
    if modeling_module is not None:
        if _bind_veomni_ops(modeling_module, get_ops_config()):
            logger.info_rank0("OpSlot-based kernel dispatch active.")

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }

    if attn_implementation not in (
        "veomni_flex_attention_with_sp",
        "veomni_magi_attention_with_sp",
        "veomni_flash_attention_2_with_sp",
        "veomni_flash_attention_3_with_sp",
        "veomni_flash_attention_4_with_sp",
    ):
        logger.warning_rank0(
            f"building foundation model with attn_implementation: {attn_implementation}.. you are missing sequence parallelism support. Please use a veomni_*_with_sp attention implementation for SP."
        )

    if (init_device == "cpu" and get_parallel_state().global_rank != 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False

    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
    )

    if is_torch_npu_available():
        # We override the forward method (on NPU devices) instead of passing CPU FA kwargs directly to the model in the trainer,
        # due to the behavior in https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L130
        logger.info_rank0(
            "We override the model’s forward method on NPU devices to ensure that the FA kwargs are on CPU, since the npu_fused_attention requires cpu FA kwargs"
        )
        original_forward = model.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            if "cu_seq_lens_q" in kwargs and kwargs["cu_seq_lens_q"] is not None:
                kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_q"].cpu()
            if "cu_seq_lens_k" in kwargs and kwargs["cu_seq_lens_k"] is not None:
                kwargs["cu_seq_lens_k"] = kwargs["cu_seq_lens_k"].cpu()
            return original_forward(*args, **kwargs)

        model.forward = wrapped_forward

    assert not getattr(model, "use_kernels", False), (
        "Still evaluating HF kernels hub integration with VeOmni patches; keep use_kernels disabled for now "
        "to avoid unexpected kernel loading side effects."
    )

    model_class_path = f"{model.__class__.__module__}.{model.__class__.__name__}"
    logger.info_rank0(f"Built foundation model class: {model_class_path}")

    return model
