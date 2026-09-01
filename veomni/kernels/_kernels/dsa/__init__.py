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
# See the License for the specific language governing limitations
# under the License.

"""DeepSeek Sparse Attention compound kernels.

``dsa_attention`` / ``dsa_indexer`` each have ``deepseek_v4`` and ``glm``
rows. Fused impls are opaque wrappers around TileLang or FlashMLA
``Function.apply``.
"""

from ...registry import register_kernel
from ...requirement import CudaKernelRequirement
from .attention.deepseek_v4 import eager as dsv4_attn_eager
from .attention.deepseek_v4 import tilelang as dsv4_attn_tilelang
from .attention.glm import eager as glm_attn_eager
from .attention.glm import flashmla_cudnn as glm_attn_flashmla
from .indexer.deepseek_v4 import eager as dsv4_indexer_eager
from .indexer.deepseek_v4 import tilelang as dsv4_indexer_tilelang
from .indexer.glm import cudnn as glm_indexer_cudnn
from .indexer.glm import eager as glm_indexer_eager


_TILELANG = CudaKernelRequirement(min_cc=90)
_CUDA = CudaKernelRequirement()

register_kernel("dsa_attention", "deepseek_v4", "eager", wrapper=dsv4_attn_eager.wrapper)

register_kernel(
    "dsa_attention",
    "deepseek_v4",
    "tilelang",
    wrapper=dsv4_attn_tilelang.wrapper,
    requirement=_TILELANG,
)
register_kernel("dsa_attention", "glm", "eager", wrapper=glm_attn_eager.wrapper)

register_kernel(
    "dsa_attention",
    "glm",
    "flashmla_cudnn",
    wrapper=glm_attn_flashmla.wrapper,
    requirement=_CUDA,
)

register_kernel("dsa_indexer", "deepseek_v4", "eager", wrapper=dsv4_indexer_eager.wrapper)

register_kernel(
    "dsa_indexer",
    "deepseek_v4",
    "tilelang",
    wrapper=dsv4_indexer_tilelang.wrapper,
    requirement=_TILELANG,
)
register_kernel("dsa_indexer", "glm", "eager", wrapper=glm_indexer_eager.wrapper)

register_kernel("dsa_indexer", "glm", "cudnn", wrapper=glm_indexer_cudnn.wrapper, requirement=_CUDA)
