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

"""What a caller carries between n-gram lookups, and nothing more.

The hashing is already incremental -- :func:`gather_lookback` takes ``history``
and ``start_pos``, and :meth:`DeepseekV41Ngram.forward` hands back the
``ngram_compressed`` ids that feed the next call. What was missing is the
bookkeeping, and the part of it worth writing down is how little has to be kept.

An n-gram ending at a position reaches back ``max_ngram_size - 1`` tokens and no
further, so that is the whole cache: a fixed-width tail, not a transcript. The
two arguments look like they must agree -- the docstring on ``gather_lookback``
describes ``history`` as ``[B, start_pos]`` -- but they are independent in the
arithmetic. ``history`` decides how far back the walk can read; ``start_pos``
only tells the blocking rule whether the walk has reached the true start of the
sequence. Keeping a short tail and an honest ``start_pos`` is therefore exact,
not an approximation, and it is what keeps a long conversation's n-gram state
constant-sized instead of growing with the transcript.

The one combination that must not happen is ``history=None`` with a non-zero
``start_pos``: the lookback would read forward from index 0 and the blocking
rule, seeing a position past the start, would not stop it. :class:`
DeepseekV41NgramCache` cannot express that state, which is the reason it exists
rather than callers passing the two arguments themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch


if TYPE_CHECKING:
    from .modeling import DeepseekV41Ngram


@dataclass
class DeepseekV41NgramCache:
    """The trailing compressed ids an n-gram can still reach, plus where we are.

    Attributes:
        lookback: ``max_ngram_size - 1``, the deepest a hash reads.
        history: ``[B, min(start_pos, lookback)]`` compressed ids, oldest first,
            or None before anything has been consumed.
        start_pos: absolute position of the next token, so the blocking rule can
            tell a genuine sequence start from a window boundary.
    """

    lookback: int
    history: Optional[torch.Tensor] = None
    start_pos: int = 0

    @classmethod
    def from_config(cls, config) -> "DeepseekV41NgramCache":
        return cls(lookback=config.max_ngram_size - 1)

    def snapshot(self) -> tuple:
        """Clone the tail so a rejected speculative token can rewind the hash."""
        history = None if self.history is None else self.history.clone()
        return history, self.start_pos

    def restore(self, snap: tuple) -> None:
        history, start_pos = snap
        self.history = None if history is None else history.clone()
        self.start_pos = start_pos

    def advance(self, compressed: torch.Tensor) -> None:
        """Absorb a step's compressed ids, keeping only the reachable tail."""
        self.history = compressed if self.history is None else torch.cat([self.history, compressed], dim=1)
        # Sliced by explicit bound rather than ``[-lookback:]``: a lookback of 0
        # would make that negative index the identity and carry the whole tensor.
        keep = self.history.shape[1] - self.lookback
        if keep > 0:
            self.history = self.history[:, keep:]
        self.start_pos += int(compressed.shape[1])


def ngram_lookup_step(
    module: "DeepseekV41Ngram",
    input_ids: torch.Tensor,
    cache: DeepseekV41NgramCache,
    token_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """One n-gram lookup against the carried tail, advancing the cache.

    Takes any number of tokens, so the prompt goes through the same call as the
    tokens after it -- a prefill is just the first step, and there is no second
    code path to keep in agreement with this one.
    """
    outputs = module(input_ids, token_mask=token_mask, history=cache.history, start_pos=cache.start_pos)
    cache.advance(outputs["ngram_compressed"])
    return outputs


__all__ = ["DeepseekV41NgramCache", "ngram_lookup_step"]
