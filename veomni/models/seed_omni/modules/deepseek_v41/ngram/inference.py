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

"""``deepseek_v41_ngram``'s FSM ``generate`` -- hash the prompt, then each token.

The rows this module emits are a per-position property, not a per-segment one,
so they travel to the backbone as ``ngram_embeds`` in the returned dict rather
than being written onto carrier items -- the same choice the training hooks
make, and for the same reason: ~24 KB/token would otherwise survive into nodes
that have no use for it.

Prefill reuses :func:`pack_ngram_inputs`, which walks the carrier and covers
image spans with ``pad_token_id`` under a ``False`` mask. A decode step cannot:
its carrier tail is an ``output`` item, which that packer would treat as a span
and pad away. The step reads the sampled id from ``meta['input_ids']`` instead
-- the same key training reads, written there by the text encoder when it
samples, which keeps the token ids on the carrier like everything else.
"""

from __future__ import annotations

from typing import Any

import torch

from ....utils.conversation import ConversationItem
from .decode import DeepseekV41NgramCache
from .packing import pack_ngram_inputs


class DeepseekV41NgramInferenceMixin:
    """FSM ``generate`` for the two hash tables.

    Listed *before* ``PretrainedOmniModule`` in the bases so the base's no-op
    ``reset_*_inference_state`` defaults do not shadow these by MRO. A surviving
    cache would not raise -- it would hash the next conversation's first tokens
    against the previous one's tail.
    """

    _ngram_cache: DeepseekV41NgramCache | None

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._ngram_cache = None

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if getattr(self, "_ngram_cache", None) is None:
            self._ngram_cache = DeepseekV41NgramCache.from_config(self.config)
            input_ids, token_mask, _ = pack_ngram_inputs([conversation_list], self.device, self.config.pad_token_id)
        else:
            input_ids, token_mask = self._sampled_token(conversation_list)

        snapshot = self._ngram_cache.snapshot()
        compressed = self.compress(input_ids, token_mask)
        n_tokens = int(input_ids.shape[1])

        def dspark_ngram_commit(n_accepted: int) -> None:
            """Rewind hashes of rejected draft tokens. Prefill never calls this."""
            if n_accepted >= n_tokens:
                return
            self._ngram_cache.restore(snapshot)
            if n_accepted > 0:
                self._ngram_cache.advance(compressed[:, :n_accepted])

        hash_ids = self.hash_ids(
            input_ids,
            token_mask,
            history=snapshot[0],
            start_pos=snapshot[1],
        )
        self._ngram_cache.advance(compressed)
        return {
            "conversation_list": conversation_list,
            "ngram_embeds": self.lookup(hash_ids),
            "dspark_ngram_commit": dspark_ngram_commit,
        }

    def _sampled_token(self, conversation_list: list[ConversationItem]) -> tuple:
        """The ids the text encoder just sampled, and their always-true mask."""
        tail = conversation_list[-1]
        token_ids = tail.meta.get("input_ids")
        if token_ids is None:
            raise ValueError(
                "deepseek_v41_ngram needs meta['input_ids'] on the carrier tail to hash a decode step: it "
                "hashes token ids, and the text encoder overwrites `value` with the sampled token's "
                "embedding. DeepseekV41TextEncoder.generate stashes the copy -- a tail without it means the "
                "FSM body ran this node before the sampler."
            )
        input_ids = token_ids.reshape(1, -1).to(self.device)
        return input_ids, torch.ones_like(input_ids, dtype=torch.bool)


__all__ = ["DeepseekV41NgramInferenceMixin"]
