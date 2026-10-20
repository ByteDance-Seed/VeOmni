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

"""DeepSeek-V4.1 vision tower (``deepseek_v41_vision``): ViT + aligner + span.

Ported from the released ``inference/vision.py``; transformers 5.16.1 has no
``deepseek_v41_vision`` to patch, so the stack is written out here.

This module emits an image's **entire** LLM span, not just the patch rows:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

The three delimiters are learned ``(out_hidden_size,)`` embeddings, and they live
here rather than in the text encoder because they are not tokenizer tokens -- in
the released checkpoint every position of the span carries ``image_token_id`` in
``input_ids`` and only the token *type* tells them apart. Keeping the whole span
on this side means the sibling text encoder never has to know an image layout and
the backbone just splices one contiguous segment.

The released reference runs one image per forward (its attention is full and
bidirectional *within* an image). Here the images of a micro-batch are packed into
one varlen sequence with ``cu_seqlens`` at the image boundaries, which is the same
computation -- an image still attends only to itself -- in one kernel call.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    sp_pad_and_slice,
)
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from veomni.utils.import_utils import is_flash_attn_2_available

from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from .configuration import DeepseekV41VisionEncoderConfig
from .processing import (
    _OMNI_TOKEN_TYPES,
    _OMNI_VIT_GRID,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    DeepseekV41VisionImageProcessor,
    DeepseekV41VisionPreprocessor,
)


# FA2 ships no aarch64 wheel, so a CUDA box is not proof the kernel is there:
# Grace-Blackwell has CUDA and no ``flash_attn``. Dispatch on the package.
USE_FLASH_ATTN_VARLEN = IS_CUDA_AVAILABLE and is_flash_attn_2_available()

if USE_FLASH_ATTN_VARLEN:
    from flash_attn import flash_attn_varlen_func
elif IS_NPU_AVAILABLE:
    import torch_npu


class _VisualOutputSlot(NamedTuple):
    """How ``forward_post`` scatters one encoded image's span back onto the carrier."""

    item: ConversationItem
    span_len: int


@lru_cache(maxsize=32)
def _vision_freqs(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE cos/sin for an ``n_h x n_w`` patch grid, ``(n_h * n_w, 1, dim)`` each.

    The first half of ``dim`` rotates by the row index and the second half by the
    column index, so one rotate-half over the full head dim carries both axes.
    Cached because a training batch repeats a handful of distinct grids.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def _sdpa_varlen_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-segment SDPA fallback for platforms without a varlen kernel."""
    outputs = []
    for i in range(cu_seqlens.numel() - 1):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        qi, ki, vi = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        outputs.append(scaled_dot_product_attention(qi, ki, vi).squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


class DeepseekV41VisionRMSNorm(nn.Module):
    """RMS norm with an fp32 weight, normalizing in fp32.

    Written out rather than imported from ``deepseek_v4``: the backbone's norm is
    tied to that config's ``rms_norm_eps`` (1e-20), while the ViT's is a plain 1e-6
    that the released ``vision.py`` hardcodes and the vision config never mentions.
    """

    def __init__(self, dim: int, eps: float = 1e-6, rms_impl: str | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        if rms_impl is None:
            rms_impl = resolve_op_impl("rms_norm_implementation")
        self.veomni_rms_norm = VeomniOp("rms_norm", "deepseek_v4", rms_impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.veomni_rms_norm(x, self.weight, eps=self.eps)


class DeepseekV41VisionPatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class DeepseekV41VisionAttention(nn.Module):
    """Full bidirectional attention within one image, packed varlen across images."""

    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)
        attn_impl = config.ops_implementation.get(
            "attn_implementation",
            getattr(config, "_attn_implementation", None),
        )
        self.veomni_attn = VeomniOp("attention", "standard", attn_impl) if attn_impl else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        if get_parallel_state().sp_enabled:
            sp_group = get_parallel_state().sp_group
            q = gather_seq_scatter_heads(q, seq_dim=0, head_dim=1, group=sp_group)
            k = gather_seq_scatter_heads(k, seq_dim=0, head_dim=1, group=sp_group)
            v = gather_seq_scatter_heads(v, seq_dim=0, head_dim=1, group=sp_group)

        force_sdpa = self.veomni_attn is not None and self.veomni_attn.impl in ("sdpa", "veomni_sdpa", "eager")
        if not force_sdpa and USE_FLASH_ATTN_VARLEN:
            out = flash_attn_varlen_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
        elif not force_sdpa and IS_NPU_AVAILABLE:
            seqlens = tuple(cu_seqlens[1:].cpu().numpy().tolist())
            out = torch_npu.npu_fusion_attention(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                q.size(1),
                pse=None,
                atten_mask=None,
                scale=1.0 / math.sqrt(self.head_dim),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=seqlens,
                actual_seq_kvlen=seqlens,
            )[0]
        else:
            out = _sdpa_varlen_attn(q, k, v, cu_seqlens)

        if get_parallel_state().sp_enabled:
            out = gather_heads_scatter_seq(out, head_dim=1, seq_dim=0, group=sp_group)
        return self.wo(out.reshape(n, -1).to(x.dtype))


class DeepseekV41VisionMLP(nn.Module):
    """Fused-gate SwiGLU: ``w1`` emits gate and up in one ``2 * intermediate`` matmul."""

    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        rms_impl = config.ops_implementation.get("rms_norm_implementation", resolve_op_impl("rms_norm_implementation"))
        self.norm1 = DeepseekV41VisionRMSNorm(config.hidden_size, rms_impl=rms_impl)
        self.attn = DeepseekV41VisionAttention(config)
        self.norm2 = DeepseekV41VisionRMSNorm(config.hidden_size, rms_impl=rms_impl)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm2(x))


class DeepseekV41ViT(nn.Module):
    """DeepSeek ViT: patch embed -> 32 blocks with 2D RoPE -> final norm."""

    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        rms_impl = config.ops_implementation.get("rms_norm_implementation", resolve_op_impl("rms_norm_implementation"))
        self.patch_embed = DeepseekV41VisionPatchEmbed(config)
        self.blocks = nn.ModuleList([DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = DeepseekV41VisionRMSNorm(config.hidden_size, rms_impl=rms_impl)

    def forward(
        self,
        patches: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, max_seqlen)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
    """``downsample_ratio``-square patch cells -> one backbone-width token each."""

    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        in_dim = config.hidden_size * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.out_hidden_size)
        self.w2 = nn.Linear(config.out_hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        # Pad right/bottom to whole cells; a partial cell at the edge is kept, which
        # is why the token grid is a ceil-divide rather than a floor-divide.
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


def build_deepseek_v41_vit_batch(
    items: list[ConversationItem],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[list[int]], list[torch.Tensor], list[_VisualOutputSlot]]:
    """One packed ViT batch from CPU-preprocessed image items."""
    patches, grids, token_types, slots = [], [], [], []
    for item in items:
        patches.append(item.value.to(device=device, dtype=dtype))
        grids.append(list(item.meta[_OMNI_VIT_GRID]))
        types = item.meta[_OMNI_TOKEN_TYPES]
        token_types.append(types)
        slots.append(_VisualOutputSlot(item=item, span_len=int(types.numel())))
    return torch.cat(patches, dim=0), grids, token_types, slots


def build_deepseek_v41_vit_metadata(
    grids: list[list[int]],
    rope_dim: int,
    rope_theta: float,
    device: torch.device,
) -> dict[str, Any]:
    """``cu_seqlens`` at the image boundaries plus the concatenated 2D RoPE tables."""
    cu: list[int] = [0]
    cos_parts, sin_parts = [], []
    for n_h, n_w in grids:
        cu.append(cu[-1] + n_h * n_w)
        cos, sin = _vision_freqs(int(n_h), int(n_w), rope_dim, float(rope_theta))
        cos_parts.append(cos)
        sin_parts.append(sin)
    cos = torch.cat(cos_parts, dim=0).to(device)
    sin = torch.cat(sin_parts, dim=0).to(device)
    if get_parallel_state().sp_enabled:
        pad = -cu[-1] % get_parallel_state().sp_size
        if pad:
            cu.append(cu[-1] + pad)
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0)
    return {
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "max_seqlen": max((b - a for a, b in zip(cu, cu[1:])), default=0),
        "cos": cos,
        "sin": sin,
    }


def scatter_deepseek_v41_visual_embeds(slots: list[_VisualOutputSlot], spans: torch.Tensor) -> None:
    """Write each image's assembled span back onto its conversation item."""
    for item_span, slot in zip(torch.split(spans, [s.span_len for s in slots], dim=0), slots):
        slot.item.value = item_span


class InferenceMixin:
    """FSM ``generate`` -- encode every user image and scatter its span.

    Listed before :class:`PretrainedOmniModule` in the bases so MRO resolves this
    ``generate`` rather than the base's no-op; this module holds no state across
    FSM steps, so there is nothing to reset.
    """

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        items = [p for p in conversation_list if p.type == "image" and p.role == "user"]
        if not items:
            return {"conversation_list": conversation_list}
        patches, grids, token_types, slots = build_deepseek_v41_vit_batch(items, device=self.device, dtype=self.dtype)
        metadata = build_deepseek_v41_vit_metadata(grids, self.config.rope_dim, self.config.rope_theta, self.device)
        spans = self._encode(patches, grids, token_types, metadata)
        scatter_deepseek_v41_visual_embeds(slots, spans)
        return {"conversation_list": conversation_list}


class DeepseekV41VisionEncoder(InferenceMixin, PretrainedOmniModule):
    """DeepSeek-V4.1 vision tower: ViT + aligner + the three learned span delimiters."""

    config_class = DeepseekV41VisionEncoderConfig
    image_processor_class = DeepseekV41VisionImageProcessor
    preprocessor_class = DeepseekV41VisionPreprocessor
    base_model_prefix = "deepseek_v41_vision"
    main_input_name = "pixel_values"
    _no_split_modules = ["DeepseekV41VisionBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: DeepseekV41VisionEncoderConfig):
        super().__init__(config)
        self.config = config
        self.vision = DeepseekV41ViT(config)
        self.aligner = DeepseekV41Aligner(config)
        # Span delimiters, in backbone width. Named as in the released checkpoint so
        # the convert step moves them across without a rename table.
        self.image_start = nn.Parameter(torch.zeros(config.out_hidden_size))
        self.image_end = nn.Parameter(torch.zeros(config.out_hidden_size))
        self.image_newline = nn.Parameter(torch.zeros(config.out_hidden_size))
        self.post_init()

    def freeze_model(self) -> None:
        """When ``config.freeze``, freeze the ViT but keep the aligner trainable.

        The aligner is the projection into the backbone's embedding space, so it is
        the part that has to move when the backbone does; the ViT itself is a
        general-purpose encoder.
        """
        if self.config.freeze:
            self.vision.requires_grad_(False)
            self.aligner.requires_grad_(True)

    def _assemble_span(self, rows: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Aligner rows + learned delimiters -> one ``(span_len, out_hidden_size)`` block."""
        types = types.to(rows.device)
        span = rows.new_empty(types.numel(), rows.size(-1))
        span[types == IMAGE] = rows
        span[types == IMAGE_START] = self.image_start.to(rows.dtype)
        span[types == IMAGE_END] = self.image_end.to(rows.dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(rows.dtype)
        return span

    def _encode(
        self,
        pixel_values: torch.Tensor,
        grids: list[list[int]],
        token_types: list[torch.Tensor],
        vit_metadata: dict[str, Any],
    ) -> torch.Tensor:
        features = self.vision(
            pixel_values.to(self.dtype),
            vit_metadata["cos"],
            vit_metadata["sin"],
            vit_metadata["cu_seqlens"],
            vit_metadata["max_seqlen"],
        )
        if get_parallel_state().sp_enabled:
            features = gather_outputs(features, gather_dim=0, group=get_parallel_state().sp_group)
        cu = vit_metadata["cu_seqlens"]
        spans = []
        for index, (n_h, n_w) in enumerate(grids):
            patch_rows = features[int(cu[index]) : int(cu[index + 1])]
            spans.append(self._assemble_span(self.aligner(patch_rows, int(n_h), int(n_w)), token_types[index]))
        return torch.cat(spans, dim=0)

    def forward(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor | None = None,
        grids: list[list[int]] | None = None,
        token_types: list[torch.Tensor] | None = None,
        vit_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        return {"image_embeds": self._encode(pixel_values, grids, token_types, vit_metadata)}


__all__ = [
    "DeepseekV41Aligner",
    "DeepseekV41ViT",
    "DeepseekV41VisionEncoder",
    "build_deepseek_v41_vit_batch",
    "build_deepseek_v41_vit_metadata",
    "scatter_deepseek_v41_visual_embeds",
]
