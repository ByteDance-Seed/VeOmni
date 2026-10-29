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

"""Coalesced GDN Ulysses exchange with explicit process-group ownership."""

import torch
import torch.distributed as dist


def pack(q, k, v, beta, decay):
    """Concatenate same-dtype projections while retaining independent head layouts."""
    if not (q.dtype == k.dtype == v.dtype == beta.dtype == decay.dtype):
        raise ValueError("Coalescing requires identical projection dtypes")
    if q.shape != k.shape or q.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected matching [B,S,Hk,Dk] Q/K and [B,S,Hv,Dv] V")
    batch, sequence, key_heads, key_dim = q.shape
    if v.shape[:2] != q.shape[:2] or v.shape[2] % key_heads:
        raise ValueError("Value heads must be a multiple of key heads")
    ratio = v.shape[2] // key_heads
    value_dim = v.shape[-1]
    if beta.shape != (*v.shape[:3],) or decay.shape != beta.shape:
        raise ValueError("Expected per-value-head beta and decay projections")
    fused = torch.cat(
        (
            q,
            k,
            v.reshape(batch, sequence, key_heads, ratio * value_dim),
            beta.reshape(batch, sequence, key_heads, ratio),
            decay.reshape(batch, sequence, key_heads, ratio),
        ),
        dim=-1,
    )
    return fused, (key_dim, value_dim, ratio)


def unpack(fused, layout):
    """Restore the five projections without mixing heads or samples."""
    key_dim, value_dim, ratio = layout
    q, k, v, beta, decay = fused.split((key_dim, key_dim, ratio * value_dim, ratio, ratio), dim=-1)
    batch, sequence, key_heads, _ = q.shape
    return (
        q,
        k,
        v.reshape(batch, sequence, key_heads * ratio, value_dim),
        beta.reshape(batch, sequence, key_heads * ratio),
        decay.reshape(batch, sequence, key_heads * ratio),
    )


def prepare_exchange(x, world, heads_to_sequence):
    """Place destination rank first; do not perform arithmetic on payloads."""
    if x.ndim != 4 or world < 1:
        raise ValueError("Expected [B,S,H,D] and a positive group size")
    batch, sequence, heads, width = x.shape
    if heads_to_sequence:
        if heads % world:
            raise ValueError("Head count must divide evenly across ranks")
        local_heads = heads // world
        send = x.reshape(batch, sequence, world, local_heads, width).permute(2, 0, 1, 3, 4)
        shape = (batch, sequence, local_heads, width)
    else:
        if sequence % world:
            raise ValueError("Sequence length must divide evenly across ranks")
        local_sequence = sequence // world
        send = x.reshape(batch, world, local_sequence, heads, width).permute(1, 0, 2, 3, 4)
        shape = (batch, local_sequence, heads, width)
    return send.contiguous(), shape


def finish_exchange(received, shape, heads_to_sequence):
    """Restore source-rank order to the corresponding sequence or head axis."""
    world = received.shape[0]
    batch, local_sequence, local_heads, width = shape
    received = received.reshape(world, batch, local_sequence, local_heads, width)
    if heads_to_sequence:
        return received.permute(1, 0, 2, 3, 4).reshape(batch, world * local_sequence, local_heads, width)
    return received.permute(1, 2, 0, 3, 4).reshape(batch, local_sequence, world * local_heads, width)


def _exchange_aligned(x, group, heads_to_sequence):
    """Exchange 512-byte-aligned peer packets and remove per-peer padding."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    send, shape = prepare_exchange(x, world, heads_to_sequence)
    peer = send.flatten(1)
    elements = peer.shape[1]
    quantum = 512 // peer.element_size()
    padding = (-elements) % quantum
    if padding:
        peer = torch.nn.functional.pad(peer, (0, padding))
    received = torch.empty_like(peer)
    dist.all_to_all_single(received, peer, group=group)
    # Padding belongs to each peer's byte packet, never to the model sequence.
    received = received[:, :elements].contiguous().reshape(send.shape)
    return finish_exchange(received, shape, heads_to_sequence)


class _AlignedPackedAllToAll(torch.autograd.Function):
    """Invert the explicit-group permutation in backward without rescaling."""

    @staticmethod
    def forward(ctx, x, group, heads_to_sequence):
        """Apply the forward rank permutation."""
        ctx.group = group
        ctx.heads_to_sequence = heads_to_sequence
        return _exchange_aligned(x, group, heads_to_sequence)

    @staticmethod
    def backward(ctx, gradient):
        """Apply its inverse using the same communicator."""
        return _AlignedPackedAllToAll.apply(gradient, ctx.group, not ctx.heads_to_sequence), None, None


def coalesced_gather_aligned(q, k, v, beta, decay, group):
    """Gather GDN sequence shards using one same-dtype projection collective."""
    if group is None:
        raise ValueError("An explicit sequence-parallel group is required")
    fused, layout = pack(q, k, v, beta, decay)
    q, k, v, beta, decay = unpack(_AlignedPackedAllToAll.apply(fused, group, True), layout)
    # Projection slices feed the caller's concatenation; do not copy them first.
    return q, k, v, beta.contiguous(), decay.contiguous()
