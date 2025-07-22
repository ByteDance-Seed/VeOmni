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


from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .loader import BaseModelLoader, get_loader


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)


def build_foundation_model(
    config_path: str,
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: bool = False,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """
    if config_kwargs is None:
        config_kwargs = {}

    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True, **config_kwargs)

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    loader: Optional[BaseModelLoader] = get_loader(config, force_use_huggingface)

    if not force_use_huggingface:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from ..ops.attention import flash_attention_forward

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
    return model
