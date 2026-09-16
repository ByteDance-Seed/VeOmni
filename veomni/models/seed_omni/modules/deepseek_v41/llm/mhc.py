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

"""Manifold-constrained Hyper-Connections, wired the way V4.1 pipelines them.

The residual stream is ``hc_mult`` parallel copies. Each sublayer sits between a
*collapse* (the copies weighted down into one input) and an *expand* (the output
placed back across the copies, with the incoming stream mixed in through a
doubly-stochastic ``comb``). The three coefficient sets -- ``pre``, ``post``,
``comb`` -- all come out of one projection of the flattened stream, with ``comb``
Sinkhorn-projected onto the doubly-stochastic manifold.

The core arithmetic is identical to ``DeepseekV4HyperConnection``'s. The wiring
is not, and that is what this file exists for: **V4.1 uses each site's ``pre`` at
the *next* site**, not its own. Attention collapses with the previous block's
FFN ``pre``, the FFN collapses with this block's attention ``pre``, and the final
stream collapse uses the last block's FFN ``pre``.

That one-site pipeline is why neither V4 vehicle can be reused:

* ``DeepseekV4HyperConnection.forward`` returns ``(post, comb, collapsed)``. Its
  ``pre`` is consumed internally to build ``collapsed`` and never leaves, so
  there is nothing to hand to the next site.
* ``veomni.ops.kernels.mhc``'s ``mhc.pre`` variant makes the same assumption one
  level down -- ``mhc_pre_tile_kernels`` ends in
  ``mhc_pre_apply_mix(hidden_streams, pre)`` with the ``pre`` it just computed.

So the mixes and the collapse are split into two entry points here. The *expand*
has no such problem and goes through the existing ``mhc.post`` kernel, which is
also the heaviest of the three: it is the only one writing a full
``[B, S, hc_mult, D]`` tensor.

Splitting ``mhc.pre`` into registry variants of its own -- a ``mixes`` and a
``collapse`` over the same TileKernels sub-ops the fused path already calls --
would recover the rest of the kernel speedup. It is left out deliberately: it
touches shared ops, so it belongs to a change that can measure the difference
rather than to the one that first makes the model run.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ......ops.dispatch import OpSlot
from .configuration import DeepseekV41LlmConfig
from .reuse import DeepseekV4UnweightedRMSNorm


veomni_mhc_post = OpSlot("mhc", "post")

#: ``post`` is a placement weight in ``[0, 2]``, so the sigmoid is doubled. Named
#: because the value is shared with the kernel path, which takes it as an argument.
POST_MULTIPLIER = 2.0


class DeepseekV41HyperConnection(nn.Module):
    """One site's mHC coefficients: ``(pre, post, comb)`` from the stream itself.

    A site is an attention or an FFN within one block, and owns its own ``fn`` /
    ``base`` / ``scale``. Unlike V4's equivalent this returns ``pre`` rather than
    applying it, because in V4.1 the collapse it belongs to happens one site
    later -- see the module docstring.
    """

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One learned scale per output: pre, post, comb.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, S, hc, D]`` -> ``pre[B, S, hc]``, ``post[B, S, hc]``, ``comb[B, S, hc, hc]``."""
        hc = self.hc_mult
        # Normalised over the whole flattened hc*D vector, one statistic per
        # token -- not per copy. Normalising per copy would couple the copies mHC
        # deliberately keeps separate.
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = POST_MULTIPLIER * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


def mhc_collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Weight the ``hc_mult`` copies down into one sublayer input.

    ``[B, S, hc, D]`` and ``[B, S, hc]`` -> ``[B, S, D]``. ``pre`` comes from the
    *previous* site, which is the whole point of this being a free function
    rather than the tail of :meth:`DeepseekV41HyperConnection.forward`.
    """
    return (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(hidden_streams.dtype)


def mhc_expand(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Place a sublayer output back across the copies and mix the residual in.

    ``output[B, S, D]``, ``residual[B, S, hc, D]`` -> ``[B, S, hc, D]``.

    The eager form is written as a matmul against ``comb.transpose(-1, -2)``
    rather than as the reference's ``sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)``.
    They are the same contraction -- ``sum_i comb[i, j] * residual[i, d]`` -- but
    the reference's spelling materialises a ``[B, S, hc, hc, D]`` intermediate,
    which at hc_mult 4 is four times the stream it is mixing.
    """
    if veomni_mhc_post.use_non_eager_impl:
        return veomni_mhc_post(output, residual, post, comb)
    dtype = output.dtype
    return post.to(dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )


def identity_pre_mix(hidden_streams: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """The ``pre`` the first block collapses with: all weight on copy 0.

    Every copy holds the same embedding at that point, so this is one-hot rather
    than uniform only to keep the scale of the collapsed vector equal to the
    embedding's -- averaging would divide it by ``hc_mult``.
    """
    pre_mix = hidden_streams.new_zeros(hidden_streams.shape[0], hidden_streams.shape[1], hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def expand_to_streams(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``[B, S, D]`` -> ``[B, S, hc_mult, D]``, one copy per stream."""
    return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


__all__ = [
    "POST_MULTIPLIER",
    "DeepseekV41HyperConnection",
    "expand_to_streams",
    "identity_pre_mix",
    "mhc_collapse",
    "mhc_expand",
]
