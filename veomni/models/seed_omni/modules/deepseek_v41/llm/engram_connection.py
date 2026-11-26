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

"""The Engram *connection*: ``wkv`` + the gate that writes it into the stream.

This is the half of the released ``Engram`` module that stays in the backbone. The
other half -- compressing token ids, hashing each position's n-grams and gathering
the two ~384M-row tables -- is the separate ``deepseek_v41_ngram`` OmniModule,
because it depends on nothing but ``input_ids`` and the image mask.

The boundary is the lookup, and it is also the cheap place to cut: the gathered
rows are ``n_hash_cols * head_dim`` = 24 values per token per engram layer
(~24 KB/token in bf16), where the mHC residual stream this file reads is
``hc_mult * hidden_size`` (~102 KB/token). Everything below needs the stream, so
it has to be here.

There is no ``deepseek_v4`` counterpart to reuse -- V4's ``num_hash_layers`` /
``DeepseekV4HashRouter`` is a frozen token-id-to-expert table for MoE routing,
unrelated to this despite the shared word "hash".
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ......utils.device import IS_CUDA_AVAILABLE, get_torch_device
from ......utils.import_utils import is_package_available
from .configuration import DeepseekV41LlmConfig
from .quantized import DeepseekV41Linear, linear_quant_dtype


_HAS_TRITON = is_package_available("triton")
if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_engram_gate_add_kernel(
        Stream_ptr,
        KV_ptr,
        QKW_ptr,
        TokenMask_ptr,
        Out_ptr,
        EPS: tl.constexpr,
        CLAMP_VAL: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        HC: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_TOKEN_MASK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        s_idx = tl.program_id(1).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        stream = tl.load(Stream_ptr + (row * HC + s_idx) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        if HAS_TOKEN_MASK:
            active = tl.load(TokenMask_ptr + row).to(tl.int1)
            if not active:
                tl.store(Out_ptr + (row * HC + s_idx) * D + offs_d, stream.to(Out_ptr.dtype.element_ty), mask=mask_d)
                return

        kv_base = KV_ptr + row * ((HC + 1) * D)
        key = tl.load(kv_base + s_idx * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        val = tl.load(kv_base + HC * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        qkw = tl.load(QKW_ptr + s_idx * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        rstd_s = tl.rsqrt(tl.sum(stream * stream, axis=0) / D + EPS)
        rstd_k = tl.rsqrt(tl.sum(key * key, axis=0) / D + EPS)
        dot = tl.sum(stream * qkw * key, axis=0) * (rstd_s * rstd_k * INV_SQRT_D)
        sqrt_abs = tl.sqrt(tl.maximum(tl.abs(dot), CLAMP_VAL))
        signed_sqrt = tl.where(dot >= 0.0, sqrt_abs, -sqrt_abs)
        gate = tl.sigmoid(signed_sqrt)
        out = (stream + gate * val).to(Out_ptr.dtype.element_ty)
        tl.store(Out_ptr + (row * HC + s_idx) * D + offs_d, out, mask=mask_d)


class DeepseekV41EngramConnection(nn.Module):
    """Write one engram layer's gathered rows into the mHC residual stream.

    ``wkv`` turns the gathered rows into one key per hc copy plus a single shared
    value. The gate is a normalized dot product of stream against key, so a lookup
    only lands where it agrees with what the stream already holds -- an n-gram that
    matches nothing is gated to roughly zero rather than added as noise.

    Three details are load-bearing and easy to get subtly wrong, all reproduced
    from the released ``Engram.forward``:

    * The RMS normalization is per ``(token, hc copy)`` over ``hidden_size``,
      **not** jointly over the copies. Normalizing jointly would couple the copies
      that mHC deliberately keeps separate.
    * ``q_weight`` and ``k_weight`` only ever appear as a product, so they are kept
      as two parameters purely to match the checkpoint; the forward multiplies them
      once.
    * The gate takes a *signed* square root before the sigmoid
      (``copysign(|dot|.clamp_min(1e-6).sqrt(), dot)``), matching the training
      kernel. Dropping the sign flips the gate for every negative dot product.
    """

    #: Floor on |dot| before the square root -- keeps the sqrt's gradient finite at 0.
    clamp_value: float = 1e-6

    def __init__(self, config: DeepseekV41LlmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_hash_index = config.engram_layer_ids.index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps

        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = DeepseekV41Linear(
            n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
            quant_dtype=linear_quant_dtype(config),
            config=config,
            site=f"layers.{layer_idx}.engram_wkv",
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def _get_fp32_qk_weight(self) -> torch.Tensor:
        key = (self.q_weight.data_ptr(), self.q_weight._version, self.k_weight.data_ptr(), self.k_weight._version)
        cached = self.__dict__.get("_qk_weight_cache")
        if cached is None or cached[0] != key:
            cached = (key, (self.q_weight.float() * self.k_weight.float()).contiguous())
            self.__dict__["_qk_weight_cache"] = cached
        return cached[1]

    def forward(
        self,
        hidden_streams: torch.Tensor,
        ngram_embeds: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[B, L, hc_mult, D]`` stream + ``[B, L, n_hash_cols, head_dim]`` rows -> stream.

        ``token_mask`` is ``[B, L]``, ``False`` for positions that take no part in an
        n-gram (image spans). Those positions have their gate zeroed, so they pass
        through untouched -- masking the gate rather than the value keeps the
        ``wkv`` matmul dense and uniform across ranks.
        """
        kv = self.wkv(ngram_embeds.flatten(-2))
        if (
            not torch.is_grad_enabled()
            and IS_CUDA_AVAILABLE
            and _HAS_TRITON
            and hidden_streams.is_cuda
            and hidden_streams.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and 0 < self.hidden_size <= 16384
        ):
            m = hidden_streams.numel() // (self.hc_mult * self.hidden_size)
            out = torch.empty_like(hidden_streams)
            if m > 0:
                qkw = self._get_fp32_qk_weight()
                has_mask = token_mask is not None
                mask_c = token_mask.reshape(-1).contiguous() if has_mask else qkw
                block_d = triton.next_power_of_2(self.hidden_size)
                num_warps = 4 if block_d <= 2048 else 8
                with get_torch_device().device(hidden_streams.device):
                    _triton_engram_gate_add_kernel[(m, self.hc_mult)](
                        hidden_streams.contiguous(),
                        kv.contiguous(),
                        qkw,
                        mask_c,
                        out,
                        EPS=float(self.eps),
                        CLAMP_VAL=float(self.clamp_value),
                        INV_SQRT_D=float(self.hidden_size**-0.5),
                        HC=self.hc_mult,
                        D=self.hidden_size,
                        BLOCK_D=block_d,
                        HAS_TOKEN_MASK=has_mask,
                        num_warps=num_warps,
                    )
            return out

        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))

        weight = self.q_weight.float() * self.k_weight.float()
        stream = hidden_streams.float()
        rstd = torch.rsqrt(stream.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (stream * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_streams.dtype)


def engram_layer_order(config: DeepseekV41LlmConfig) -> List[int]:
    """Engram layer indices in the order the ngram module stacks its tables."""
    return list(config.engram_layer_ids)


__all__ = [
    "DeepseekV41EngramConnection",
    "engram_layer_order",
]
