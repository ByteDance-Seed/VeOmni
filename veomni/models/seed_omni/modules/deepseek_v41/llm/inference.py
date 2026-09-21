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

"""``deepseek_v41_llm``'s FSM ``generate`` -- prefill the prompt, then step.

The shape is the one every SeedOmni backbone uses: state on the module, a
branch on whether the cache exists, and a ``ConversationItem`` of type
``output`` carrying the tail hidden states for the text encoder to sample from.
What differs is underneath. There is no HF ``past_key_values`` here, because
V4.1's attention is not a KV cache -- a sliding window, an append-only
compressed history and a partially filled group per layer, which
:class:`DeepseekV41DecodeCache` holds and ``decode_backbone_step`` reads.

Prefill goes through :func:`pack_llm_conversations_for_forward` like the dense
backbones do, and for the same reason: the carrier is the only data path, and
that helper is its reader. It yields a flat ``arange`` of positions, which is
exactly what ``build_compression_plans`` wants -- the compression groups and the
window are derived from ``position_ids`` and from nothing else.

Prefill is one batched ``forward`` that captures the window, compressed KV,
index keys and pending tail each layer's attention already computed. The
per-position loop remains as :meth:`DeepseekV41LlmInferenceMixin._prefill_per_position`
so the capture can be checked against it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ....utils.conversation import ConversationItem, is_dummy
from ...base.llm_packing import pack_llm_conversations_for_forward
from .decode import DeepseekV41DecodeCache, decode_backbone_step


def build_image_mask(conversation_list: List[ConversationItem], device: torch.device) -> Optional[torch.Tensor]:
    """``[1, S]`` true inside an image span, or None when the prompt has no image.

    None rather than an all-false mask, matching what ``forward`` documents: the
    MoE's VL routing bias and the Engram gate both skip their select entirely
    when there is no image, and an all-false mask would make them do the work to
    learn the same thing.

    The spans are measured off the same items, in the same order and with the
    same dummy filter as the packer, so the mask lines up with the embeddings by
    construction rather than by a shared length assumption.
    """
    flags = []
    for item in conversation_list:
        if is_dummy(item):
            continue
        flags.append(torch.full((item.value.size(0),), item.type == "image", dtype=torch.bool, device=device))
    if not flags or not any(bool(flag.any()) for flag in flags):
        return None
    return torch.cat(flags).unsqueeze(0)


class DeepseekV41LlmInferenceMixin:
    """FSM ``generate`` for the CED backbone.

    Listed *before* ``OmniPreTrainedModel`` in the bases: the base ships no-op
    ``reset_*_inference_state`` defaults, and MRO resolves left to right, so put
    second these would shadow the real ones below and the caches would survive
    across requests -- a second conversation decoding against the first one's
    sliding window, silently.
    """

    _decode_cache: Optional[DeepseekV41DecodeCache]
    _next_position: int

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._decode_cache = None
        self._next_position = 0

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if getattr(self, "_decode_cache", None) is None:
            return self._prefill(conversation_list, ngram_embeds)
        return self._step(conversation_list, ngram_embeds)

    def _prefill(
        self,
        conversation_list: List[ConversationItem],
        ngram_embeds: Optional[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        """One batched backbone forward, capturing the decode caches from it.

        The training forward already computes every window row, compressed group
        and index key decode needs, then throws them away. Capture writes those
        tensors onto :class:`DeepseekV41DecodeCache` instead of walking the prompt
        token by token. Grouping is a function of ``position_ids`` alone, so the
        batched groups and the per-position groups are the same groups.

        The per-position loop is :meth:`_prefill_per_position`, kept as the
        reference the capture is tested against.
        """
        inputs_embeds, _, position_ids, _ = pack_llm_conversations_for_forward([conversation_list], self.device)
        if inputs_embeds.shape[1] == 0:
            raise ValueError("deepseek_v41_llm was asked to prefill an empty conversation.")
        image_mask = build_image_mask(conversation_list, self.device)

        orig_len = int(inputs_embeds.shape[1])
        self._decode_cache = DeepseekV41DecodeCache.empty(self.config)
        outputs = self.model(
            inputs_embeds,
            position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            capture=self._decode_cache.layers,
            capture_seq_len=orig_len,
        )
        hidden_states = outputs["hidden_states"][:, :orig_len]
        if hidden_states.numel() and not torch.isfinite(hidden_states).all():
            window = self._decode_cache.layers[0].window
            row_ok = torch.isfinite(hidden_states[0]).all(dim=-1)
            nan_rows = (~row_ok).nonzero(as_tuple=True)[0]
            window_nan = 0 if window is None else int(torch.isnan(window).sum())
            raise RuntimeError(
                "batched prefill produced non-finite hidden states: "
                f"shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype} "
                f"nan={int(torch.isnan(hidden_states).sum())} inf={int(torch.isinf(hidden_states).sum())} "
                f"nan_rows={int(nan_rows.numel())}/{orig_len} "
                f"last_row_finite={bool(row_ok[-1])} layer0_window_nan={window_nan}"
            )
        self._next_position = orig_len
        return self._emit(conversation_list, hidden_states)

    def _prefill_per_position(
        self,
        conversation_list: List[ConversationItem],
        ngram_embeds: Optional[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        """Step the prompt one token at a time.

        The reference implementation for the batched-capture path. Each step is
        :func:`decode_backbone_step`, which is also how decode continues after
        either prefill. Not used by :meth:`generate`.
        """
        inputs_embeds, _, _, _ = pack_llm_conversations_for_forward([conversation_list], self.device)
        image_mask = build_image_mask(conversation_list, self.device)

        self._decode_cache = DeepseekV41DecodeCache.empty(self.config)
        hidden_states = None
        for position in range(inputs_embeds.shape[1]):
            hidden_states = decode_backbone_step(
                self.model,
                inputs_embeds[:, position : position + 1],
                position,
                self._decode_cache,
                ngram_embeds=None if ngram_embeds is None else [e[:, position : position + 1] for e in ngram_embeds],
                image_mask=None if image_mask is None else image_mask[:, position : position + 1],
            )
        if hidden_states is None:
            raise ValueError("deepseek_v41_llm was asked to prefill an empty conversation.")
        self._next_position = int(inputs_embeds.shape[1])

        return self._emit(conversation_list, hidden_states)

    def _step(
        self,
        conversation_list: List[ConversationItem],
        ngram_embeds: Optional[List[torch.Tensor]],
    ) -> Dict[str, Any]:
        tail = conversation_list[-1]
        if tail.type != "output":
            raise ValueError(
                f"deepseek_v41_llm expected the carrier to end on an 'output' item, found '{tail.type}'. "
                "The text encoder writes the sampled token's embedding onto that item; a different tail "
                "means the FSM body ran the two in the wrong order."
            )

        inputs_embeds = tail.value[-1:].to(self.device).unsqueeze(0)
        hidden_states = decode_backbone_step(
            self.model,
            inputs_embeds,
            self._next_position,
            self._decode_cache,
            ngram_embeds=ngram_embeds,
            # A sampled token is text by construction: an image reaches the
            # carrier from the vision module, never from the sampler.
            image_mask=None,
        )
        self._next_position += 1
        return self._emit(conversation_list, hidden_states)

    def _emit(self, conversation_list: List[ConversationItem], hidden_states: torch.Tensor) -> Dict[str, Any]:
        conversation_list.append(
            ConversationItem(type="output", value=self._tail_hidden(hidden_states), role="assistant")
        )
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            return hidden_states.squeeze(0)[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()


__all__ = ["DeepseekV41LlmInferenceMixin", "build_image_mask"]
