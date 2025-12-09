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
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import is_torch_npu_available
from .loader import BaseModelLoader, get_loader, get_model_config, get_model_processor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str, force_use_huggingface: bool = False) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return get_model_processor(processor_path, force_use_huggingface, padding_side="right", trust_remote_code=True)


def build_config(config_path: str, force_use_huggingface: bool = False, **config_kwargs) -> "PretrainedConfig":
    """
    Builds the model config.
    """
    return get_model_config(config_path, force_use_huggingface, trust_remote_code=True, **config_kwargs)


def build_foundation_model(
    config_path: Union[str, PretrainedConfig],
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[
        Literal["eager", "sdpa", "flash_attention_2", "native-sparse"]
    ] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "npu", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: Optional[bool] = False,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}

    if isinstance(config_path, PretrainedConfig):
        config = config_path
    else:
        config = build_config(config_path, force_use_huggingface, **config_kwargs)

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    loader: Optional[BaseModelLoader] = get_loader(config, force_use_huggingface)

    if not force_use_huggingface:
        from functools import partial

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from ..ops.attention import flash_attention_forward

        seed_kernel_attn_implementation = os.getenv("SEED_KERNEL_ATTN_IMPLEMENTATION")

        if seed_kernel_attn_implementation == "fa3":
            flash_attention_forward = partial(flash_attention_forward, implementation="fa3")
        elif seed_kernel_attn_implementation == "fa2":
            flash_attention_forward = partial(flash_attention_forward, implementation="fa2")
        elif seed_kernel_attn_implementation == "lego":
            flash_attention_forward = partial(flash_attention_forward, implementation="lego")
        else:
            assert seed_kernel_attn_implementation is None, (
                f"seed_kernel_attn_implementation={seed_kernel_attn_implementation} is not supported"
            )

        ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)

    init_kwargs = {
        "config": config,
        "torch_dtype": getattr(torch, torch_dtype),
        "attn_implementation": attn_implementation,
        "trust_remote_code": True,
    }

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

    return model
