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

"""DeepSeek-V4.1 CED backbone (``deepseek_v41_llm``).

The 40 decoder layers, the Engram connections that write n-gram lookups into the
mHC stream, and the final collapse + norm. **No embedding and no LM head**: by
the G1 ruling those live in ``deepseek_v41_text_encoder``, which also serves the
DSpark draft stages, so this module takes ``inputs_embeds`` and returns hidden
states.

What arrives already done:

* the token embedding, from ``deepseek_v41_text_encoder``;
* the image spans, spliced into those embeddings by ``deepseek_v41_vision`` --
  the reference does this inline as ``merge_image_embeddings``, and here it has
  happened one node earlier in the graph;
* the n-gram row gathers, from ``deepseek_v41_ngram``, one tensor per Engram
  layer in ``engram_layer_ids`` order.

The layer loop is the released ``Transformer.forward``'s, minus the KV caches.
Three orderings in it are load-bearing and each is noted where it happens: the
Engram write lands *before* its layer, the DSpark hook reads the stream *before*
the layer rather than after, and the final collapse uses the last layer's FFN
``pre`` rather than one computed from the final stream.

This module is training-only. Every cache in the reference exists to carry state
between decode steps, and the attention stack dropped them deliberately (see
``attention.py``); wiring a generation mixin that expects ``past_key_values``
would produce a model that runs and answers wrongly. Generation is the
``deepseek_v41`` generation graph's problem, and it is not wired yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ....omni_pretrained_model import OmniPreTrainedModel
from .attention import (
    DeepseekV41RotaryEmbedding,
    IndexerLossTerms,
    SharedAttentionState,
    build_compression_plans,
    sequence_slices_from_positions,
)
from .configuration import DeepseekV41LlmConfig
from .decoder_layer import DeepseekV41DecoderLayer
from .engram_connection import build_engram_connections
from .inference import DeepseekV41LlmInferenceMixin
from .mhc import expand_to_streams, identity_pre_mix, mhc_collapse
from .reuse import DeepseekV4RMSNorm


class DeepseekV41Backbone(nn.Module):
    """The layer stack itself, separated from the OmniModule wrapper around it.

    Split out so the stack can be built and exercised without the
    ``PreTrainedModel`` machinery -- config validation, weight init hooks,
    ``from_pretrained`` -- which is most of what a test of the layer loop would
    otherwise have to stand up.
    """

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.engram = build_engram_connections(config)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # One instance for the whole stack: the two frequency schedules depend on
        # the config alone, and a per-layer copy would be 40 identical buffers.
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        image_mask: Optional[torch.Tensor] = None,
        return_draft_hiddens: bool = False,
    ) -> Dict[str, Any]:
        if ngram_embeds is None and self.engram:
            raise ValueError(
                f"this backbone has Engram connections at layers {list(self.config.engram_layer_ids)} but no "
                "ngram_embeds were passed. The deepseek_v41_ngram node must run before this one in the graph; "
                "its absence would train the connections' wkv on nothing while the loss still fell."
            )
        if ngram_embeds is not None and len(ngram_embeds) != len(self.config.engram_layer_ids):
            raise ValueError(
                f"ngram_embeds has {len(ngram_embeds)} entries but engram_layer_ids names "
                f"{len(self.config.engram_layer_ids)} layers. They are positional -- entry i belongs to "
                f"layer {list(self.config.engram_layer_ids)[:1]}... -- so a length mismatch silently "
                "pairs a layer with another layer's table."
            )

        # Image tokens take no part in an n-gram, so the Engram gate is zeroed
        # there. The reference derives the same mask as `~image_mask`; a text-only
        # batch has no mask at all rather than an all-true one, which lets both
        # consumers skip their select entirely.
        token_mask = None if image_mask is None else ~image_mask

        plans = build_compression_plans(self.config, position_ids, sequence_slices_from_positions(position_ids))
        shared = SharedAttentionState()

        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)

        draft_hiddens: List[torch.Tensor] = []
        router_logits: List[torch.Tensor] = []
        indexer_terms: List[IndexerLossTerms] = []
        draft_targets = set(self.config.dspark_target_layer_ids)

        for layer_idx, layer in enumerate(self.layers):
            if str(layer_idx) in self.engram:
                # Before the layer, not after: the reference adds the lookup to the
                # stream at the top of the loop body, so this layer's attention is
                # the first thing that sees it. Writing it after would hand the
                # n-gram to the *next* layer and leave this one reading a stream
                # the trained weights never produced.
                connection = self.engram[str(layer_idx)]
                hidden_streams = connection(hidden_streams, ngram_embeds[connection.layer_hash_index], token_mask)
            if return_draft_hiddens and layer_idx in draft_targets:
                # The MTP head reads its target layer's *input*, not its output --
                # `main_hiddens.append(h.mean(dim=2))` sits above `layer(...)` in
                # the reference. Collected here so the DSpark work has the hook at
                # the right place; off by default because three [B, S, D] tensors
                # are not free and nothing consumes them yet.
                draft_hiddens.append(hidden_streams.mean(dim=2))

            hidden_streams, pre_mix, layer_logits, indexer_loss = layer(
                hidden_streams, pre_mix, position_ids, self.rotary_emb, shared, plans, image_mask
            )
            router_logits.append(layer_logits)
            if indexer_loss is not None:
                indexer_terms.append(indexer_loss)

        if self.config.dsa_indexer_loss and not indexer_terms:
            # A config that enables the objective but names no index source would
            # otherwise accept the flag and train nothing: there would be no
            # student, the KL would never be built, and the fold-in downstream
            # would be skipped in silence -- a plausible loss curve training the
            # one set of weights this flag exists to reach.
            raise RuntimeError(
                "dsa_indexer_loss is enabled but no layer built an indexer KL: index_source_layer_ids="
                f"{list(self.config.index_source_layer_ids)} selects no layer that owns an indexer, or "
                "dsa_indexer_loss_coef is non-positive. The flag would be accepted and train nothing."
            )

        # The last layer's FFN `pre`, not one derived from the final stream. See
        # the module docstring; `DeepseekV4HyperHead` implements the latter and is
        # why it cannot be reused here.
        hidden_states = self.norm(mhc_collapse(hidden_streams, pre_mix))

        outputs: Dict[str, Any] = {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
        }
        if indexer_terms:
            outputs["indexer_kl_total"] = torch.stack([term.kl for term in indexer_terms]).sum()
            outputs["indexer_uniform_total"] = torch.stack([term.uniform for term in indexer_terms]).sum()
            # The count travels so the *metric* can be a per-layer mean while the
            # objective stays a sum: dividing the sum by the layer count here would
            # make the gradient each indexer receives depend on how many other
            # layers happened to carry one.
            outputs["indexer_kl_layers"] = len(indexer_terms)
            outputs["indexer_query_tokens"] = hidden_states.shape[0] * hidden_states.shape[1]
        if return_draft_hiddens:
            outputs["draft_hiddens"] = draft_hiddens
        return outputs


class DeepseekV41Llm(DeepseekV41LlmInferenceMixin, OmniPreTrainedModel):
    """``deepseek_v41_llm`` -- the CED backbone as an OmniModule.

    Takes ``inputs_embeds`` because the embedding belongs to the sibling text
    encoder, and returns ``hidden_states`` for it to project back to logits.
    """

    config_class = DeepseekV41LlmConfig
    base_model_prefix = "deepseek_v41_llm"
    main_input_name = "inputs_embeds"
    # The Engram connections hang off the model rather than off the layers that
    # use them, so every decoder layer has the same parameter set and one wrap
    # policy covers the stack. The connections are wrapped separately: at layers
    # 1 and 14 only, `wkv` is 5120 x (hidden * (hc_mult + 1)), which is large
    # enough to be worth its own shard and too irregular to fold into the layer.
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41EngramConnection"]
    supports_gradient_checkpointing = True

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41Backbone(config)
        self._decode_cache = None
        self._next_position = 0
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """There is none -- the text encoder owns it.

        Declared rather than inherited so the failure is a named refusal. HF's
        default walks ``base_model`` looking for an embedding and raises an
        ``AttributeError`` from somewhere inside ``PreTrainedModel``, which reads
        like a bug in this class rather than the module boundary it is.
        """
        raise NotImplementedError(
            "deepseek_v41_llm holds no token embedding: it lives in deepseek_v41_text_encoder, which also "
            "serves the DSpark draft stages. This module takes inputs_embeds."
        )

    def set_input_embeddings(self, value: nn.Module) -> None:
        raise NotImplementedError(
            "deepseek_v41_llm holds no token embedding; set it on deepseek_v41_text_encoder instead."
        )

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        image_mask: Optional[torch.Tensor] = None,
        return_draft_hiddens: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """``[B, S, hidden]`` embeddings -> ``[B, S, hidden]`` hidden states.

        ``position_ids`` restarts at zero per packed sample and is the only place
        the sample boundaries are declared: the sliding window, the compression
        groups and the compressed-history visibility are all derived from it.

        ``image_mask`` is ``[B, S]``, true inside an image span. It drives two
        unrelated things -- the MoE's VL routing bias and (negated) the Engram
        gate -- and ``None`` means a text-only batch rather than an all-false
        mask, which lets both skip their select.

        The auxiliary KL leaves in the returned dict rather than on ``self``.
        Gradient checkpointing wraps the layers, and a tensor that reaches the
        loss by any route other than the flattened output carries no graph: the
        indexer would receive no gradient at all while the logged KL fell.
        """
        del kwargs
        if inputs_embeds is None:
            raise ValueError("deepseek_v41_llm requires inputs_embeds; it holds no token embedding.")
        if position_ids is None:
            raise ValueError(
                "deepseek_v41_llm requires position_ids: they carry the packed sample boundaries, and "
                "assuming a single arange here would compress across them."
            )
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            return_draft_hiddens=return_draft_hiddens,
        )


__all__ = ["DeepseekV41Backbone", "DeepseekV41Llm"]
