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

"""``ShardedEmbedding``: an ``nn.Embedding`` whose rows are split over the ``emb`` group.

The parallel plan shards the table on dim-0 (vocab) over the ``emb``
extra-parallel group, and FSDP2 shards each rank's rows on dim-1 (hidden) over
the ``emb_fsdp`` sub-mesh. The module is its own FSDP2 unit, so its forward
reads the plain ``[vocab/emb, hidden]`` rows inside that unit's unshard hooks,
and FSDP2 reduce-scatters and scales the weight gradient like any other
parameter's. Reading ``weight`` from outside the module's forward sees the
sharded DTensor instead, which is why there is no tied output projection.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ..parallel_state import get_parallel_state, is_parallel_state_initialized
from .all_to_all import AllToAllEmbedding


def _emb_enabled() -> bool:
    if not is_parallel_state_initialized():
        return False
    ps = get_parallel_state()
    return "emb" in ps.extra_parallel_sizes and ps.extra_parallel_enabled("emb")


class ShardedEmbedding(nn.Embedding):
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
        if not _emb_enabled():
            raise self._layout_error(rows, emb_size=1)
        local = idx - get_parallel_state().extra_parallel_rank("emb") * rows
        return local if 0 <= local < rows else None

    @padding_idx.setter
    def padding_idx(self, idx: Optional[int]) -> None:
        self._global_padding_idx = idx

    def extra_repr(self) -> str:
        # nn.Embedding formats from ``__dict__``, which has no ``padding_idx`` now that it is a property.
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self._global_padding_idx is not None:
            s += f", padding_idx={self._global_padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.scale_grad_by_freq:
            s += ", scale_grad_by_freq=True"
        if self.sparse:
            s += ", sparse=True"
        return s

    def _layout_error(self, rows: int, emb_size: int) -> RuntimeError:
        return RuntimeError(
            f"{type(self).__name__} holds {rows} of {self.num_embeddings} vocab rows, but the emb group here has "
            f"{emb_size} rank(s): the table was not split for the parallel state it runs under."
        )

    def _check_emb_layout(self) -> bool:
        """Whether ``emb`` is on, after checking the weight holds the rows that implies."""
        if isinstance(self.weight, DTensor):
            raise RuntimeError(
                f"{type(self).__name__}.weight is a DTensor here, so it is being read outside its FSDP2 unshard "
                "hooks; call the module instead of reading its weight."
            )
        enabled = _emb_enabled()
        emb_size = get_parallel_state().extra_parallel_sizes["emb"] if enabled else 1
        rows = self.weight.shape[0]
        if rows * emb_size != self.num_embeddings:
            raise self._layout_error(rows, emb_size)
        return enabled

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._check_emb_layout():
            return super().forward(input)
        if self.max_norm is not None:
            raise NotImplementedError("ShardedEmbedding does not support max_norm under emb parallel.")
        output = AllToAllEmbedding.apply(get_parallel_state().extra_parallel_group("emb"), input, self.weight)
        if self._global_padding_idx is not None:
            # Same as F.embedding: the padding row receives no gradient.
            output = torch.where((input == self._global_padding_idx).unsqueeze(-1), output.detach(), output)
        return output


__all__ = ["ShardedEmbedding"]
