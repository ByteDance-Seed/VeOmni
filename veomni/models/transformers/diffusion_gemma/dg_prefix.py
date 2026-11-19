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
"""Pack encoder cache + current canvas for the DG_PREFIX Triton kernel.

The kernel is varlen and ``batch=1``. It never sees a padding mask: each
packed segment is physically ``H_prefix || current_noisy_canvas``, and
visibility is decided *before* the call.

Layer type is already encoded in the decoder mask that
``create_diffusion_decoder_attention_mask`` builds:

* full layers: every non-pad, non-future encoder key
* sliding layers: the last ``W-1`` of those visible keys (or the already
  truncated sliding cache at generation)

Gathering with that encoder-key axis therefore applies the window without a
second ``history_window`` argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from ....utils import logging


if TYPE_CHECKING:
    from .configuration_diffusion_gemma import DiffusionGemmaTextConfig


logger = logging.get_logger(__name__)

_KERNEL: Callable | None = None


def _load_dg_prefix_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from ....ops.kernels.dllm_attention import dllm_attention

    _KERNEL = dllm_attention
    return _KERNEL


def encoder_visible_from_decoder_mask(
    attention_mask: torch.Tensor | None,
    cache_len: int,
    batch_size: int,
    device: torch.device,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Return ``(batch, cache_len)`` bool keys the current layer may read.

    ``attention_mask`` is the *per-layer* decoder mask already selected by
    ``layer_types`` (full vs sliding). Padding, future canvases, and the
    sliding window have been applied to its encoder half. ``sliding_window``
    is only used when the mask is ``None`` (the unpadded shortcut), so a
    sliding layer still keeps just the last ``W-1`` encoder keys.
    """
    if attention_mask is None:
        if sliding_window is None:
            return torch.ones(batch_size, cache_len, dtype=torch.bool, device=device)
        visible = torch.zeros(batch_size, cache_len, dtype=torch.bool, device=device)
        keep = min(max(sliding_window - 1, 0), cache_len)
        if keep:
            visible[:, -keep:] = True
        return visible
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, 0, 0, :]
    if attention_mask.dim() != 2:
        raise ValueError(
            f"DG_PREFIX needs a 2D or 4D decoder mask to gather encoder keys, got shape {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[-1] < cache_len:
        raise ValueError(
            "Decoder mask is shorter than the encoder cache: "
            f"mask length {attention_mask.shape[-1]} vs cache {cache_len}"
        )
    encoder = attention_mask[:, :cache_len]
    if encoder.dtype == torch.bool:
        return encoder
    # eager additive mask: keep = 0, drop = finfo.min
    return encoder >= 0


def pack_prefix_kv(
    encoder_k: torch.Tensor,
    encoder_v: torch.Tensor,
    encoder_visible: torch.Tensor,
    canvas_groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather visible encoder keys into packed ``(1, heads, total, dim)``.

    Empty prefixes stay in ``cu_seqlens`` as zero-length segments so each
    row still maps to one kernel segment.

    With ``canvas_groups`` above 1, ``encoder_visible`` carries that many rows per
    cached sample -- flat canvas batching gives the decoder K canvases each with its
    own visible prefix -- and segment ``i`` reads cache row ``i // K``. Gathering
    against that row rather than a repeated cache is the point: the copies live only
    in the packed prefix, which holds just the visible keys anyway. K is passed in
    rather than divided out of the two shapes, so a mismatch that is not a fold still
    fails here instead of silently reading the wrong sample's cache.
    """
    if encoder_k.shape != encoder_v.shape:
        raise ValueError("encoder K/V shapes must match")
    if encoder_k.dim() != 4:
        raise ValueError("encoder K/V must have shape (batch, heads, cache_len, dim)")
    batch, heads, cache_len, dim = encoder_k.shape
    segments = batch * canvas_groups
    if tuple(encoder_visible.shape) != (segments, cache_len):
        raise ValueError(
            "encoder_visible must have shape (batch * canvas_groups, cache_len), "
            f"got {tuple(encoder_visible.shape)} vs {(segments, cache_len)}"
        )
    canvases_per_row = canvas_groups
    visible = encoder_visible.bool()
    lengths = visible.sum(dim=-1)
    coords = visible.nonzero(as_tuple=False)
    if coords.numel() == 0:
        packed_k = encoder_k.new_empty(1, heads, 0, dim)
        packed_v = encoder_v.new_empty(1, heads, 0, dim)
    else:
        flat = torch.div(coords[:, 0], canvases_per_row, rounding_mode="floor") * cache_len + coords[:, 1]
        k_flat = encoder_k.permute(0, 2, 1, 3).reshape(batch * cache_len, heads, dim)
        v_flat = encoder_v.permute(0, 2, 1, 3).reshape(batch * cache_len, heads, dim)
        packed_k = k_flat.index_select(0, flat).permute(1, 0, 2).unsqueeze(0).contiguous()
        packed_v = v_flat.index_select(0, flat).permute(1, 0, 2).unsqueeze(0).contiguous()
    cu = torch.zeros(segments + 1, device=encoder_k.device, dtype=torch.int32)
    cu[1:] = lengths.to(torch.int32).cumsum(0)
    return packed_k, packed_v, cu


def pack_canvas_states(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``(batch, heads, canvas, dim)`` into ``(1, heads, batch*canvas, dim)``.

    Every row is assumed to have the same canvas length, which is how the
    collator stacks ``decoder_input_ids``.
    """
    if states.dim() != 4:
        raise ValueError("canvas states must have shape (batch, heads, canvas, dim)")
    batch, heads, canvas, dim = states.shape
    packed = states.permute(1, 0, 2, 3).contiguous().view(1, heads, batch * canvas, dim)
    cu = torch.arange(0, (batch + 1) * canvas, canvas, device=states.device, dtype=torch.int32)
    return packed, cu


def unpack_canvas_output(output: torch.Tensor, batch_size: int, canvas_length: int) -> torch.Tensor:
    """Undo ``pack_canvas_states`` on the kernel's ``(1, total_q, heads, dim)`` output."""
    if output.shape[0] != 1 or output.shape[1] != batch_size * canvas_length:
        raise ValueError(
            "DG_PREFIX output is not packed as (1, batch*canvas, heads, dim): "
            f"got {tuple(output.shape)} for batch={batch_size} canvas={canvas_length}"
        )
    return output.view(batch_size, canvas_length, output.shape[2], output.shape[3])


def should_use_dg_prefix_decoder(
    config: "DiffusionGemmaTextConfig",
    query_states: torch.Tensor,
    dropout: float | int,
) -> bool:
    """Whether this decoder layer should call the packed Triton kernel.

    Sequence parallelism is not asked about here. This path cannot serve it -- it
    gathers each canvas's visible run out of a cache that SP leaves sharded -- but
    refusing inside a predicate put the refusal in the condition of the caller's
    `if`, which made the caller's own refusal unreachable and handed the user a
    message that does not name the path that does work. Answering only "does this
    path apply" leaves the refusal where it can say that.
    """
    if not getattr(config, "attn_dg_prefix", False):
        return False
    if not query_states.is_cuda:
        return False
    if dropout:
        raise ValueError("attn_dg_prefix does not support attention dropout; keep attention_dropout at 0")
    return True


def dg_prefix_decoder_attention(
    query_states: torch.Tensor,
    canvas_key_states: torch.Tensor,
    canvas_value_states: torch.Tensor,
    encoder_key_states: torch.Tensor,
    encoder_value_states: torch.Tensor,
    encoder_visible: torch.Tensor,
    sm_scale: float = 1.0,
    canvas_groups: int = 1,
) -> torch.Tensor:
    """Packed DG_PREFIX attention; returns ``(batch, canvas, heads, dim)``.

    ``batch`` here is the query's, which under flat canvas batching is
    ``canvas_groups`` times the encoder cache's. Only the prefix gather has to know
    that; everything else already works per query row.
    """
    kernel = _load_dg_prefix_kernel()
    batch, _, canvas, _ = query_states.shape
    prefix_k, prefix_v, cu_prefix = pack_prefix_kv(
        encoder_key_states, encoder_value_states, encoder_visible, canvas_groups
    )
    q_packed, cu_q = pack_canvas_states(query_states)
    canvas_k, _ = pack_canvas_states(canvas_key_states)
    canvas_v, _ = pack_canvas_states(canvas_value_states)
    packed_out = kernel(
        q_packed.contiguous(),
        prefix_k,
        prefix_v,
        canvas_k.contiguous(),
        canvas_v.contiguous(),
        cu_q,
        cu_prefix,
        sm_scale=sm_scale,
    )
    output = unpack_canvas_output(packed_out, batch, canvas)
    # An empty prefix never enters the kernel, so the encoder cache would
    # otherwise drop out of the decoder's backward graph. SDPA still has those
    # keys in the tensor (masked to ~0); keep a zero contribution so grads are
    # zeros rather than `None`.
    if prefix_k.numel() == 0:
        output = output + (encoder_key_states.sum() + encoder_value_states.sum()) * 0
    return output
