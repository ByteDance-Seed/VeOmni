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

"""Build tokenizer / processor / foundation model from models_kernel registries."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from veomni.distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.loader import get_loader
from veomni.models_kernel.registry import get_model_config, get_model_processor
from veomni.utils import logging
from veomni.utils.device import is_torch_npu_available


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from veomni.arguments.arguments_types import OpsImplementationConfig


logger = logging.get_logger(__name__)

CONTEXT_PARALLEL_MODEL_TYPES = frozenset({"deepseek_v4"})

_VEOMNI_SP_ATTN = (
    "veomni_flex_attention",
    "veomni_magi_attention",
    "veomni_flash_attention_2",
    "veomni_flash_attention_3",
    "veomni_flash_attention_4",
)


def check_context_parallel_supported(config: PretrainedConfig) -> None:
    """Raise unless this model type implements context parallelism.

    A no-op when context parallelism is off, which is every other configuration.
    """
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


def build_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """Build a right-padded tokenizer from ``tokenizer_path``."""
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str, **kwargs) -> ProcessorMixin:
    """Build a right-padded processor from ``processor_path``."""
    return get_model_processor(processor_path, padding_side="right", trust_remote_code=True, **kwargs)


def build_config(config_path: str, **config_kwargs) -> PretrainedConfig:
    """Build a model config from ``config_path``."""
    trust_remote_code = config_kwargs.pop("trust_remote_code", True)
    return get_model_config(config_path, trust_remote_code=trust_remote_code, **config_kwargs)


def _validate_attention_parallelism(attn_implementation: str | None) -> None:
    if attn_implementation not in ("magi_attention", "veomni_magi_attention"):
        return

    cp_size = get_parallel_state().cp_size
    if cp_size != 1:
        raise ValueError(
            f"MagiAttention currently requires context parallel size 1 (cp_size == 1), got cp_size={cp_size}."
        )


def build_foundation_model(
    config_path: str | PretrainedConfig,
    weights_path: str | None = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: None
    | (
        Literal[
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "flex_attention",
            "magi_attention",
            "veomni_flash_attention_2",
            "veomni_flash_attention_3",
            "veomni_flash_attention_4",
            "veomni_flex_attention",
            "veomni_magi_attention",
            "veomni_sage_attention",
            "veomni_sdpa",
            "native-sparse",
        ]
    ) = None,
    init_device: Literal["cpu", "cuda", "npu", "mlu", "meta"] = "cuda",
    config_kwargs: dict[str, Any] | None = None,
    encoder_data_balance: bool | None = False,
    encoder_data_balance_sorting_algo: str | None = "post_mbs_balancing_greedy_without_pad",
    kernels_implementation: OpsImplementationConfig | None = None,
) -> PreTrainedModel:
    """Build a foundation model from the models_kernel registry.

    Callers must pass ``kernels_implementation`` or pre-install a config with
    ``set_kernels_config``. There is no silent all-eager fallback.
    """
    if kernels_implementation is not None:
        attn_implementation = kernels_implementation.attn_implementation
        _validate_attention_parallelism(attn_implementation)
        set_kernels_config(kernels_implementation)
    else:
        installed = get_kernels_config()
        if installed is None:
            raise ValueError(
                "build_foundation_model requires `kernels_implementation` (or a prior "
                "`set_kernels_config(...)` call). There is no silent all-eager fallback."
            )
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

    loader = get_loader(config)

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }

    if attn_implementation not in _VEOMNI_SP_ATTN:
        logger.warning_rank0(
            f"building foundation model with attn_implementation: {attn_implementation}.. "
            "you are missing sequence parallelism support. Please use a veomni_* attention implementation for SP."
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
