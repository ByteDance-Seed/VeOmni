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
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..arguments.arguments_types import OpsImplementationConfig
from ..distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
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


def check_model_build_prerequisites(config: PretrainedConfig) -> None:
    """Let a model config refuse a run it cannot serve, before any weight is read.

    A hook rather than a body: any config class may define
    ``validate_build_prerequisites`` and this calls it. Nothing here knows which
    models do, which is the point -- the alternative is a list of model names in the
    generic builder, and a list is a thing the next model has to be remembered into.

    What a config cannot see on its own is the rest of the run, so the convention is
    that the hook reads whatever singletons it needs (the installed
    ``OpsImplementationConfig``, the parallel state) and takes no arguments. Kept as a
    plain ``getattr`` for the same reason: a config that has nothing to refuse should
    not have to say so.

    ``DeepseekV4Config.validate_build_prerequisites`` is the only implementation
    today. It refuses a Lightning Indexer KL objective configured without the TileLang
    indexer and attention it is defined in terms of -- a disagreement between a model
    field and two kernel selections, which no single dataclass can see.
    """
    validate = getattr(config, "validate_build_prerequisites", None)
    if callable(validate):
        validate()


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

    if is_torch_npu_available():
        raise NotImplementedError("Context parallelism is GPU-only in this release; set cp_size=1 on Ascend/NPU runs.")

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
    check_model_build_prerequisites(config)

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
