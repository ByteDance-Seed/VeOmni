"""Vocab-parallel (``emb`` extra-parallel) embedding lookup + tied projection.

Shared by any module whose embedding table is ``Shard(0)``-split on dim-0 (vocab)
over the ``emb`` extra-parallel group AND additionally FSDP-sharded on dim-1
(hidden) over the ``emb_fsdp`` sub-mesh -- the base ``TextEncoder``'s
``embed_tokens`` and Seedream 5.0's 258 GB over-encoding table both take this
shape. The lookup / projection kernels (``AllToAllEmbedding`` /
``VocabParallelLinear``) need this rank's *whole* emb chunk (all hidden), so the
FSDP hidden shards are gathered back first (see :meth:`emb_local_weight`).

All methods are ``@staticmethod`` so a top-level model (``TextEncoder``) or an
inner ``nn.Module`` (Seedream's ``SHMOverEncodingEmbedding``) can call them
regardless of ``self``; mix the class in for method-style access.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from veomni.distributed.parallel_state import get_parallel_state
from veomni.ops.kernels.embed import AllToAllEmbedding, VocabParallelLinear


class EmbParallelMixin:
    @staticmethod
    def emb_parallel_active() -> bool:
        """True when the ``emb`` extra-parallel group is present and size > 1."""
        ps = get_parallel_state()
        return "emb" in ps.extra_parallel_sizes and ps.extra_parallel_enabled("emb")

    @staticmethod
    def emb_local_weight(weight: torch.Tensor) -> torch.Tensor:
        """Reconstruct this emb-rank's full ``[vocab/emb, hidden]`` slice.

        The weight is ``Shard(0)`` (vocab) over ``emb`` and FSDP-sharded on the
        hidden dim over ``emb_fsdp``; the kernels need this rank's whole emb chunk
        across all hidden -- gathered WITHOUT mixing other emb ranks:

        * ``emb_fsdp == 1`` (world == emb): the local shard already spans all
          hidden, so ``to_local()`` returns it as a view -- no (tens-of-GB) copy.
        * ``emb_fsdp  > 1``: ``full_tensor()`` all-gathers the hidden shards.
          Detach under inference (``full_tensor``'s redistribute trips on an
          in-place ``detach_`` of a grad-requiring DTensor); keep grad in training
          so the kernel's backward reaches the sharded param.

        With ``emb`` OFF the weight is still a DTensor -- a plain FSDP2 ``Shard(0)``
        over ``dp_shard`` -- because a tied head reads ``embed_tokens.weight``
        directly, bypassing the embedding's own FSDP2 unshard hook. There is no
        ``emb`` group to consult, so ``full_tensor()`` all-gathers the whole
        ``[vocab, hidden]`` table for the plain ``F.linear`` projection (grad kept
        in training, detached in inference, as above).

        Non-DTensor weights (single replica / eager) pass through unchanged.
        """
        if not isinstance(weight, DTensor):
            return weight
        if not EmbParallelMixin.emb_parallel_active():
            return weight.full_tensor() if torch.is_grad_enabled() else weight.detach().full_tensor()
        ps = get_parallel_state()
        if ps.extra_parallel_fsdp_size("emb") == 1:
            return weight.to_local()
        return weight.full_tensor() if torch.is_grad_enabled() else weight.detach().full_tensor()

    @staticmethod
    def emb_parallel_lookup(embedding: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup: vocab-parallel when ``emb`` is on, else a plain call.

        ``embedding`` is the row-owning module (an ``nn.Embedding`` or a lazy
        table). Under ``emb`` its ``.weight`` is the ``Shard(0)`` DTensor and
        ``AllToAllEmbedding`` all-to-all dispatches each global id to its owning
        rank, so the result is bit-identical to a full-table lookup.
        """
        if not EmbParallelMixin.emb_parallel_active():
            return embedding(ids)
        ps = get_parallel_state()
        weight = EmbParallelMixin.emb_local_weight(embedding.weight)
        return AllToAllEmbedding.apply(ps.extra_parallel_group("emb"), ids, weight)

    @staticmethod
    def emb_parallel_project(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project ``hidden -> vocab`` logits with a (possibly ``emb``-sharded) weight.

        Dual of :meth:`emb_parallel_lookup` for a tied head: ``VocabParallelLinear``
        all-gathers the vocab shards over ``emb`` to full-vocab logits; off ``emb``
        it is a plain ``F.linear``. (``full_tensor()`` returns the fp32 master
        param, so cast to the activation dtype before the matmul.)
        """
        weight = EmbParallelMixin.emb_local_weight(weight).to(hidden_states.dtype)
        if EmbParallelMixin.emb_parallel_active():
            ps = get_parallel_state()
            return VocabParallelLinear.apply(ps.extra_parallel_group("emb"), hidden_states, weight)
        return F.linear(hidden_states, weight)


__all__ = ["EmbParallelMixin"]
