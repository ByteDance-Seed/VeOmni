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

"""MiniMax M3 sparse attention (MSA) operators on packed TND tensors.

The operator boundary follows the block-sparse kernel rather than the
Transformers reference code, so a kernel backend can replace the eager
implementation without touching the model:

``indexer(index_q, index_k, cu_seqlens, max_seqlen, *, block_size, topk_blocks, local_blocks)``
    ``index_q`` is ``[T, G, D]`` and ``index_k`` is ``[T, 1, D]`` (MQA), both
    after norm and RoPE. Returns ``[T, G, topk_blocks]`` int32 key-block
    indices, numbered within each sequence, left-packed with ``-1`` padding.

``attention(q, k, v, block_indices, cu_seqlens, max_seqlen, *, block_size, scale, dropout_p)``
    ``q`` is ``[T, Hq, D]`` and ``k``/``v`` are ``[T, G, D]`` without GQA
    repetition; query head ``h`` belongs to KV group ``h // (Hq // G)``.
    ``block_indices`` is the indexer output. Token-level causality inside a
    selected block is enforced by the operator. Returns ``[T, Hq, D]``.

Only sparse layers use these operators. Dense MiniMax layers call the standard
``attention`` family (a varlen FlashAttention backend) from the model instead.

``T`` is the full packed token count seen by this rank after the Ulysses
all-to-all, and ``cu_seqlens`` describes every packed sequence, including the
synthetic SP tail-padding sequence appended by the collator.

Sparse layers bind separate standard indexer and attention ops. The indexer
uses PyTorch. Attention defaults to the fused NPU implementation on Ascend and
to PyTorch elsewhere. Explicit eager selection uses PyTorch on every device.
"""

from ...platform import NpuKernelRequirement
from ...registry import register_op
from .eager import minimax_sparse_attention_eager, minimax_sparse_indexer_eager
from .npu import minimax_sparse_attention_npu


register_op(
    "minimax_sparse_indexer",
    "standard",
    "eager",
    description="Select MiniMax key blocks per KV group using PyTorch",
    wrapper=minimax_sparse_indexer_eager,
)

register_op(
    "minimax_sparse_attention",
    "standard",
    "eager",
    description="PyTorch MiniMax block-sparse attention on packed TND input",
    wrapper=minimax_sparse_attention_eager,
)

register_op(
    "minimax_sparse_attention",
    "standard",
    "torch_npu",
    description="MiniMax block-sparse attention using torch_npu.npu_fusion_attention",
    wrapper=minimax_sparse_attention_npu,
    requirement=NpuKernelRequirement(),
    default=True,
)


__all__ = ["minimax_sparse_attention_eager", "minimax_sparse_attention_npu", "minimax_sparse_indexer_eager"]
