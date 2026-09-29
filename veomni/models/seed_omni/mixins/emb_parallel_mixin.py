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

"""Vocab-parallel (``emb`` extra-parallel) embedding lookup + tied projection.

For a module whose embedding table is ``Shard(0)``-split on dim-0 (vocab) over
the ``emb`` extra-parallel group, and FSDP-sharded on dim-1 (hidden) over the
``emb_fsdp`` sub-mesh -- a text encoder's ``embed_tokens`` and a very large
over-encoding table (hundreds of GB) both take this shape.

The table lives in a :class:`VocabParallelEmbedding`, which the parallel plan
wraps as its own FSDP2 unit. Every read of its weight goes through that unit's
unshard hooks, so the kernels always see the plain ``[vocab/emb, hidden]`` rows
in the mixed-precision param dtype, and FSDP2 reduce-scatters and scales the
weight gradient like any other parameter's. Reading ``embedding.weight`` from
outside those hooks would instead see the sharded DTensor and bypass the
gradient reduction.

All :class:`EmbParallelMixin` methods are ``@staticmethod`` so a top-level model
(a text encoder) or an inner ``nn.Module`` (an over-encoding embedding) can call
them regardless of ``self``; mix the class in for method-style access.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
from torch.distributed.tensor import DTensor

from ....distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from ....ops.kernels.embed import AllToAllEmbedding, VocabParallelLinear


class EmbParallelMixin:
    @staticmethod
    def emb_parallel_active() -> bool:
        """True when the ``emb`` extra-parallel group is present and size > 1."""
        if not is_parallel_state_initialized():
            return False
        ps = get_parallel_state()
        return "emb" in ps.extra_parallel_sizes and ps.extra_parallel_enabled("emb")

    @staticmethod
    def emb_parallel_lookup(embedding: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup of global ``ids``; vocab-parallel when ``emb`` is on."""
        if EmbParallelMixin.emb_parallel_active() and not isinstance(embedding, VocabParallelEmbedding):
            raise TypeError(
                f"emb parallel is on but {type(embedding).__name__} is not a VocabParallelEmbedding; "
                "a plain lookup would index this rank's vocab shard with global ids."
            )
        return embedding(ids)

    @staticmethod
    def emb_parallel_project(hidden_states: torch.Tensor, embedding: nn.Module) -> torch.Tensor:
        """Tied head: project ``hidden -> vocab`` logits with ``embedding``'s weight.

        A :class:`VocabParallelEmbedding` that is its own FSDP2 unit gets
        ``project`` registered as an FSDP forward method on first use, so the
        projection unshards the weight and hooks its gradient exactly like the
        lookup does.
        """
        if isinstance(embedding, VocabParallelEmbedding):
            if isinstance(embedding, FSDPModule) and not getattr(embedding, "_project_is_fsdp_method", False):
                register_fsdp_forward_method(embedding, "project")
                embedding._project_is_fsdp_method = True
            return embedding.project(hidden_states)
        if EmbParallelMixin.emb_parallel_active() or isinstance(embedding.weight, DTensor):
            raise TypeError(
                f"{type(embedding).__name__}.weight is sharded outside its unshard hooks; "
                "use VocabParallelEmbedding so the tied projection runs under FSDP2."
            )
        return F.linear(hidden_states, embedding.weight.to(hidden_states.dtype))


class VocabParallelEmbedding(nn.Embedding):
    """``nn.Embedding`` that takes global ids when its rows are split over ``emb``.

    ``num_embeddings`` stays the global vocab size; under ``emb`` the weight
    holds only this rank's contiguous ``[vocab/emb, hidden]`` rows. With ``emb``
    off it is exactly ``nn.Embedding``.

    ``padding_idx`` reads back as an index into the rows this rank holds, and as
    ``None`` on ranks that do not hold the padding row, so initializers that
    index the weight with it (``nn.Embedding.reset_parameters``, HF
    ``_init_weights``) zero the right row after the table is split. The global
    id is kept for masking the padding gradient.
    """

    @property
    def padding_idx(self) -> Optional[int]:
        idx = self._global_padding_idx
        rows = self.weight.shape[0]
        if idx is None or rows == self.num_embeddings:
            return idx
        local = idx - get_parallel_state().extra_parallel_rank("emb") * rows
        return local if 0 <= local < rows else None

    @padding_idx.setter
    def padding_idx(self, idx: Optional[int]) -> None:
        self._global_padding_idx = idx

    def _check_emb_layout(self) -> bool:
        """Whether ``emb`` is on, after checking the weight holds the rows that implies."""
        name = type(self).__name__
        if isinstance(self.weight, DTensor):
            raise RuntimeError(
                f"{name}.weight is a DTensor here, so it is being read outside its FSDP2 unshard hooks; "
                "call it as a module or through EmbParallelMixin."
            )
        active = EmbParallelMixin.emb_parallel_active()
        emb_size = get_parallel_state().extra_parallel_sizes["emb"] if active else 1
        rows = self.weight.shape[0]
        if rows * emb_size != self.num_embeddings:
            raise RuntimeError(
                f"{name} holds {rows} of {self.num_embeddings} vocab rows, but the emb group here has "
                f"{emb_size} rank(s): the table was not split for the parallel state it runs under."
            )
        return active

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._check_emb_layout():
            return super().forward(input)
        if self.max_norm is not None:
            raise NotImplementedError("VocabParallelEmbedding does not support max_norm under emb parallel.")
        output = AllToAllEmbedding.apply(get_parallel_state().extra_parallel_group("emb"), input, self.weight)
        if self._global_padding_idx is not None:
            # Same as F.embedding: the padding row receives no gradient.
            output = torch.where((input == self._global_padding_idx).unsqueeze(-1), output.detach(), output)
        return output

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tied head logits over the full vocab; call through :meth:`EmbParallelMixin.emb_parallel_project`."""
        active = self._check_emb_layout()
        weight = self.weight.to(hidden_states.dtype)
        if not active:
            return F.linear(hidden_states, weight)
        return VocabParallelLinear.apply(get_parallel_state().extra_parallel_group("emb"), hidden_states, weight)


__all__ = ["EmbParallelMixin", "VocabParallelEmbedding"]
