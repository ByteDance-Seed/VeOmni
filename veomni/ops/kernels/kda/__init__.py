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

"""chunk_kda kernel family (Kimi K3 KDA linear attention).

``ascendc`` is the fused AscendC forward+backward path vendored from
flash-linear-attention-npu (MindSpeed-MM kimi_k3 wrapper); it covers the full
training contract (varlen cu_seqlens, use_gate_in_kernel, beta-sigmoid) with
initial_state=None / output_final_state=False. ``eager`` resolves the
modeling-side reference lazily (fla ``chunk_kda`` when flash-linear-attention
is installed, else ``chunk_kda_naive``) to avoid an ops -> models import
cycle. A standalone triton fused entry was removed earlier (EP>1 NaN race
with the MoE alltoall); the ascendc path covers production.
"""

from ...platform import NpuKernelRequirement
from ...registry import register_op


def _eager(q, k, v, g, beta, **kwargs):
    from veomni.models.transformers.kimi_k3.source import modeling_kimi_k3 as src

    fn = getattr(src, "chunk_kda", None)
    if fn is None:
        from veomni.models.transformers.kimi_k3.source.kda_fallback import chunk_kda_naive as fn
    return fn(q=q, k=k, v=v, g=g, beta=beta, **kwargs)


def _ascendc(q, k, v, g, beta, **kwargs):
    from .ascendc.chunk_kda_ascendc_kimi import chunk_kda_ascendc

    return chunk_kda_ascendc(q, k, v, g, beta, **kwargs)


register_op(
    "chunk_kda",
    "kimi_k3",
    "eager",
    wrapper=_eager,
    description="Kimi K3 KDA chunked linear attention reference (fla or naive fallback)",
)

register_op(
    "chunk_kda",
    "kimi_k3",
    "ascendc",
    wrapper=_ascendc,
    description="Kimi K3 KDA chunked linear attention fused AscendC kernel (NPU)",
    requirement=NpuKernelRequirement(),
    default=True,
)
