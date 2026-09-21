# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""The scheme the released DeepSeek-V4.1 weights are quantized under.

Two modules hold quantized weights and neither should have to ask the other
about it: ``deepseek_v41_llm`` has the projections and the expert slabs, and
``deepseek_v41_ngram`` has the two memory tables. The module boundary between
them is deliberate -- the lookup is the cheap cut -- so the scheme they share
lives here rather than in either one.

What a module config carries is the released ``quantization_config`` verbatim,
under the name ``weight_quantization``. Not ``quantization_config``, which is a
name transformers claims: it reads that off the config in ``get_hf_quantizer``,
and ``quant_method: "fp8"`` is a method it supports, so it would install its own
fp8 quantizer, substitute its own Linear and take over weight loading.
"""

from typing import Optional

import torch


#: ``quantization_config.weight_block_size`` in the released config, and the
#: width of one activation quantization block. Both sides must agree: the
#: kernels index a weight's scale by the same block counter as the activation's.
QUANT_BLOCK_SIZE = 32

#: ``quantization_config.scale_fmt``. Rounds every scale up to a power of two,
#: which is what lets it be stored as a bare exponent.
SCALE_FMT = "ue8m0"

#: The bare exponent that ``SCALE_FMT`` makes possible: no mantissa, one byte.
SCALE_DTYPE = torch.float8_e8m0fnu


def quantized_storage_dtype(config) -> Optional[torch.dtype]:
    """The dtype the released weights are stored in, or ``None`` for bf16.

    ``config.weight_quantization`` is present on the converted files (released
    dtypes). Training clears it so the llm allocates bf16 Parameters and
    ``DeepseekV41DequantConverter`` spends the scales at load.

    The scheme is checked rather than assumed. The kernels index a weight's
    scale by the same block counter as the activation's, so a checkpoint
    quantized on a different block width would not fail to run -- it would read
    the wrong scale for every block and return plausible nonsense.
    """
    declared = getattr(config, "weight_quantization", None)
    if not declared:
        return None

    block_size = declared.get("weight_block_size", [QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE])
    if tuple(block_size) != (QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE):
        raise ValueError(
            f"This checkpoint is quantized in {tuple(block_size)} blocks, but the kernels here are "
            f"built for {(QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE)}. Running it anyway would pair every "
            "block with the wrong scale."
        )
    scale_fmt = declared.get("scale_fmt", SCALE_FMT)
    if scale_fmt != SCALE_FMT:
        raise ValueError(
            f"This checkpoint's scales are {scale_fmt!r}, not {SCALE_FMT!r}. Only the latter rounds "
            f"every scale to a power of two, which is what lets it be stored as the bare exponent "
            f"{SCALE_DTYPE} the kernels read."
        )
    return torch.float8_e4m3fn


__all__ = [
    "QUANT_BLOCK_SIZE",
    "SCALE_DTYPE",
    "SCALE_FMT",
    "quantized_storage_dtype",
]
