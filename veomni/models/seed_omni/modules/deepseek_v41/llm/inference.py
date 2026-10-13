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

Prefill captures the window, compressed KV, index keys and pending tail.
Training's full-sequence ``forward`` is used when the prompt fits in the
sliding window (or there is no encoder/decoder split). Longer prompts use
:meth:`DeepseekV41Backbone.forward_ced_prefill`: encoder on the full prompt,
decoder on the last window. The per-position loop remains as
:meth:`DeepseekV41LlmInferenceMixin._prefill_per_position` so the capture can
be checked against it.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ....utils.conversation import ConversationItem
from ...base.llm_packing import pack_llm_conversations_for_forward
from .decode import DeepseekV41DecodeCache, decode_backbone_step


def build_image_mask(conversation_list: list[ConversationItem], device: torch.device) -> torch.Tensor | None:
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
        if item.is_dummy:
            continue
        flags.append(torch.full((item.value.size(0),), item.type == "image", dtype=torch.bool, device=device))
    if not flags or not any(bool(flag.any()) for flag in flags):
        return None
    return torch.cat(flags).unsqueeze(0)


class DeepseekV41LlmInferenceMixin:
    """FSM ``generate`` for the CED backbone.

    Listed *before* ``PretrainedOmniModule`` in the bases: the base ships no-op
    ``reset_*_inference_state`` defaults, and MRO resolves left to right, so put
    second these would shadow the real ones below and the caches would survive
    across requests -- a second conversation decoding against the first one's
    sliding window, silently.
    """

    _decode_cache: DeepseekV41DecodeCache | None
    _next_position: int

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._decode_cache = None
        self._next_position = 0

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        ngram_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if getattr(self, "_decode_cache", None) is None:
            return self._prefill(conversation_list, ngram_embeds)
        return self._step(conversation_list, ngram_embeds, kwargs)

    def _prefill(
        self,
        conversation_list: list[ConversationItem],
        ngram_embeds: list[torch.Tensor] | None,
    ) -> dict[str, Any]:
        """One batched backbone forward, capturing the decode caches from it.

        Short prompts (and configs with no encoder/decoder split) go through
        the training ``forward``. Longer CED prompts go through
        :meth:`DeepseekV41Backbone.forward_ced_prefill`. Grouping is a function
        of ``position_ids`` alone, so the batched groups and the per-position
        groups are the same groups on the encoder.

        The per-position loop is :meth:`_prefill_per_position`, kept as the
        reference the capture is tested against.
        """
        inputs_embeds, _, position_ids, _ = pack_llm_conversations_for_forward([conversation_list], self.device)
        if inputs_embeds.shape[1] == 0:
            raise ValueError("deepseek_v41_llm was asked to prefill an empty conversation.")
        image_mask = build_image_mask(conversation_list, self.device)

        orig_len = int(inputs_embeds.shape[1])
        self._decode_cache = DeepseekV41DecodeCache.empty(self.config)
        split = int(self.config.candidate_source_layer_id)
        window = int(self.config.sliding_window)
        prefill = (
            self.model.forward_ced_prefill if 0 < split < len(self.model.layers) and orig_len > window else self.model
        )
        outputs = prefill(
            inputs_embeds,
            position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            capture=self._decode_cache.layers,
            capture_seq_len=orig_len,
        )
        hidden_states = outputs["hidden_states"][:, :orig_len]
        if hidden_states.numel() and not torch.isfinite(hidden_states).all():
            raise RuntimeError(
                "batched prefill produced non-finite hidden states: "
                f"shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype}"
            )
        draft_hiddens = outputs.get("draft_hiddens")
        if draft_hiddens and self._decode_cache.dspark:
            seed_width = int(draft_hiddens[0].shape[1])
            self.model.seed_dspark_windows(
                draft_hiddens,
                position_ids[:, -seed_width:],
                self._decode_cache.dspark,
            )
            self._decode_cache.dspark_main = (
                self.model.mtp[0].project_main([hidden[:, -1:] for hidden in draft_hiddens]).detach()
            )
        self._decode_cache.dspark_ready = False
        self._next_position = orig_len
        return self._emit(conversation_list, hidden_states)

    def _prefill_per_position(
        self,
        conversation_list: list[ConversationItem],
        ngram_embeds: list[torch.Tensor] | None,
    ) -> dict[str, Any]:
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
        conversation_list: list[ConversationItem],
        ngram_embeds: list[torch.Tensor] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        tail = conversation_list[-1]
        if tail.type != "output":
            raise ValueError(
                f"deepseek_v41_llm expected the carrier to end on an 'output' item, found '{tail.type}'. "
                "The text encoder writes the sampled token's embedding onto that item; a different tail "
                "means the FSM body ran the two in the wrong order."
            )

        inputs_embeds, ngram_embeds = self._this_step_inputs(tail, ngram_embeds)
        n_tokens = int(inputs_embeds.shape[1])
        spec_ids = tail.meta.get("dspark_spec_ids")
        ngram_commit: Callable[[int], None] | None = kwargs.get("dspark_ngram_commit")

        if n_tokens == 1 or spec_ids is None:
            spec_hiddens = self._run_spec(tail)
            hidden_states = decode_backbone_step(
                self.model,
                inputs_embeds,
                self._next_position,
                self._decode_cache,
                ngram_embeds=ngram_embeds,
                image_mask=None,
            )
            accepted = 1
            self._next_position += 1
        else:
            spec_hiddens = None
            accepted, hidden_states = self._verify_block(
                inputs_embeds,
                ngram_embeds,
                spec_ids,
                kwargs.get("dspark_lm_head_weight"),
                kwargs.get("dspark_sampling") or {},
            )
            self._fill_spec_embeds(tail, inputs_embeds, accepted)

        self._decode_cache.dspark_ready = True
        if n_tokens > 1:
            spec_hiddens = self._run_spec(tail)
        if ngram_commit is not None:
            ngram_commit(accepted)

        outputs = self._emit(conversation_list, hidden_states)
        markov = None if spec_hiddens is None else self.model.mtp[-1].markov_head
        last_id = tail.meta.get("input_ids")
        if last_id is not None:
            last_id = last_id.reshape(-1)[accepted - 1 : accepted]
        extras = None
        if spec_ids is not None and accepted > 1:
            extras = spec_ids.reshape(-1)[: accepted - 1]
        outputs["dspark_hiddens"] = spec_hiddens
        outputs["dspark_start_id"] = None if spec_hiddens is None else last_id
        outputs["dspark_markov_embed_weight"] = None if markov is None else markov.embed.weight
        outputs["dspark_markov_head_weight"] = None if markov is None else markov.head.weight
        outputs["dspark_accepted_ids"] = extras
        return outputs

    def _verify_block(
        self,
        inputs_embeds: torch.Tensor,
        ngram_embeds: list[torch.Tensor] | None,
        spec_ids: torch.Tensor | None,
        lm_head_weight: torch.Tensor | None,
        sampling: dict[str, Any],
    ) -> tuple:
        """Run K draft tokens, keep the AR prefix, roll the KV back on the first miss.

        Verify incrementally: decode token i, then sample from token i-1's hidden
        to decide whether token i was a real hit. Decoding the whole block first
        would pay K backbone steps on a miss -- slower than not speculating.
        """
        n_tokens = int(inputs_embeds.shape[1])
        position = self._next_position
        spec = [] if spec_ids is None else [int(token) for token in spec_ids.reshape(-1).tolist()]
        hiddens = []
        accepted = 1
        for index in range(n_tokens):
            snap = self._decode_cache.snapshot()
            hidden = decode_backbone_step(
                self.model,
                inputs_embeds[:, index : index + 1],
                position + index,
                self._decode_cache,
                ngram_embeds=self._token_ngram(ngram_embeds, index),
                image_mask=None,
            )
            if index > 0:
                if not spec or lm_head_weight is None or index - 1 >= len(spec):
                    self._decode_cache.restore(snap)
                    break
                sampled = self._sample_hidden(hiddens[-1], lm_head_weight, sampling)
                if sampled != spec[index - 1]:
                    self._decode_cache.restore(snap)
                    break
                accepted = index + 1
            hiddens.append(hidden)

        self._next_position = position + accepted
        return accepted, hiddens[accepted - 1]

    def _fill_spec_embeds(self, tail: ConversationItem, inputs_embeds: torch.Tensor, accepted: int) -> None:
        """Point DSpark's next draft at the last verified token, not the first of the block."""
        block = tail.meta.get("dspark_embeds")
        if block is None or accepted <= 1:
            return
        last = inputs_embeds[:, accepted - 1]
        noise = tail.meta.get("dspark_noise_embed")
        if not torch.is_tensor(block):
            return
        if block.dim() == 2:
            block = block.unsqueeze(0)
        filled = last.unsqueeze(1).expand(-1, block.shape[1], -1).clone()
        if torch.is_tensor(noise) and block.shape[1] > 1:
            filled[:, 1:] = noise.reshape(1, 1, -1).expand(filled.shape[0], filled.shape[1] - 1, -1)
        tail.meta["dspark_embeds"] = filled.squeeze(0) if filled.shape[0] == 1 else filled

    @staticmethod
    def _token_ngram(ngram_embeds: list[torch.Tensor] | None, index: int) -> list[torch.Tensor] | None:
        if ngram_embeds is None:
            return None
        return [row[:, index : index + 1] for row in ngram_embeds]

    def _sample_hidden(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        sampling: dict[str, Any],
    ) -> int:
        last = hidden_states[:, -1, :] if hidden_states.dim() == 3 else hidden_states[-1]
        logits = F.linear(last, lm_head_weight)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        do_sample = bool(sampling.get("do_sample", True))
        if not do_sample:
            return self._agree_on_token(int(logits.argmax(dim=-1).item()), last.device)
        temperature = float(sampling.get("temperature", 1.0))
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        top_p = float(sampling.get("top_p", 1.0))
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return self._agree_on_token(int(token.item()), last.device)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative - sorted_probs > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        return logits.scatter(1, sorted_indices, sorted_logits)

    @staticmethod
    def _agree_on_token(token_id: int, device: torch.device) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return token_id
        agreed = torch.tensor([token_id], dtype=torch.long, device=device)
        dist.broadcast(agreed, src=0)
        return int(agreed.item())

    def _run_spec(self, tail: ConversationItem) -> torch.Tensor | None:
        """Draft a block from the previous verified token, matching official decode ``forward_spec``.

        The first generated token only fills ``dspark_main``; official
        ``forward_spec(..., start_pos=0)`` is seed-only and we already did that
        at prefill. Spec starts on the second decode step.
        """
        cache = self._decode_cache
        if (
            not self.model.mtp
            or cache is None
            or not cache.dspark_ready
            or cache.dspark_main is None
            or tail.meta.get("dspark_embeds") is None
        ):
            return None
        embeds = tail.meta["dspark_embeds"]
        if not torch.is_tensor(embeds):
            return None
        return self.model.forward_spec(
            embeds.to(self.device),
            self._next_position - 1,
            cache.dspark,
            cache.dspark_main,
        )

    def _this_step_inputs(
        self,
        tail: ConversationItem,
        ngram_embeds: list[torch.Tensor] | None,
    ) -> tuple:
        """Embeddings and n-gram rows for *this* decode step, same width.

        ``maybe_merge_outputs`` concatenates consecutive output items. Prefill's
        hidden and a later sampled embedding must not be verified as a DSpark
        block: the n-gram node only hashed the new ids, and an empty n-gram
        slice broadcasts Engram's stream to sequence length 0.
        """
        inputs_embeds = tail.value.to(self.device)
        if inputs_embeds.dim() == 1:
            inputs_embeds = inputs_embeds.view(1, 1, -1)
        elif inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        n_tokens = int(inputs_embeds.shape[1])
        stored_ids = tail.meta.get("input_ids")
        if stored_ids is not None:
            n_tokens = min(n_tokens, int(stored_ids.reshape(-1).numel()))
        ngram_len = None if not ngram_embeds else int(ngram_embeds[0].shape[1])
        if ngram_len is not None:
            n_tokens = min(n_tokens, ngram_len)
        if n_tokens < 1:
            raise ValueError(
                "decode step has no tokens: "
                f"embeds={tuple(tail.value.shape)} ngram_len={ngram_len} "
                f"input_ids={None if stored_ids is None else int(stored_ids.numel())}"
            )
        if inputs_embeds.shape[1] != n_tokens:
            inputs_embeds = inputs_embeds[:, -n_tokens:]
        if ngram_embeds is not None and ngram_len != n_tokens:
            ngram_embeds = [row[:, -n_tokens:] for row in ngram_embeds]
        if tail.meta.get("dspark_spec_ids") is None and n_tokens > 1:
            inputs_embeds = inputs_embeds[:, -1:]
            ngram_embeds = None if ngram_embeds is None else [row[:, -1:] for row in ngram_embeds]
        return inputs_embeds, ngram_embeds

    def _emit(self, conversation_list: list[ConversationItem], hidden_states: torch.Tensor) -> dict[str, Any]:
        hidden = self._tail_hidden(hidden_states)
        if conversation_list and conversation_list[-1].type == "output":
            # In-place: appending a second output lets maybe_merge_outputs cat the
            # previous hidden (or token embed) onto the next sampled embedding.
            conversation_list[-1].value = hidden
        else:
            conversation_list.append(ConversationItem(type="output", value=hidden, role="assistant"))
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            return hidden_states.squeeze(0)[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()


__all__ = ["DeepseekV41LlmInferenceMixin", "build_image_mask"]
