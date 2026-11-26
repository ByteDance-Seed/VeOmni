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
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.nn as nn

from ...module_modeling_base import PretrainedOmniModule
from .attention import (
    DeepseekV41RotaryEmbedding,
    IndexerLossTerms,
    SharedAttentionState,
    build_compression_plans,
    sequence_slices_from_positions,
)
from .configuration import DeepseekV41LlmConfig
from .decoder_layer import DeepseekV41DecoderLayer
from .mhc import DeepseekV41HyperConnection, expand_to_streams, identity_pre_mix, mhc_collapse
from .moe import DeepseekV41TopKRouter
from .reuse import DeepseekV4Experts, DeepseekV4RMSNorm
from .sequence_shard import SequenceShard


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
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        # One instance for the whole stack: the two frequency schedules depend on
        # the config alone, and a per-layer copy would be 40 identical buffers.
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    @property
    def has_engram(self) -> bool:
        return any(layer.engram is not None for layer in self.layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_embeds: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        return_draft_hiddens: bool = False,
        shard: SequenceShard | None = None,
    ) -> dict[str, Any]:
        """``shard`` runs this rank's rows of a sequence-parallel row.

        Every per-token input -- embeddings, positions, n-gram rows, image mask
        -- is then this rank's rows, and the outputs are too.
        """
        if ngram_embeds is None and self.has_engram:
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

        row_position_ids = position_ids if shard is None else shard.position_ids
        plans = build_compression_plans(
            self.config, row_position_ids, sequence_slices_from_positions(row_position_ids)
        )
        shared = SharedAttentionState()
        if len(self.layers) > 0:
            first_attn = self.layers[0].self_attn
            if shard is None:
                shared.window_indices = first_attn._window_indices(position_ids, position_ids.shape[1])
            else:
                shared.window_indices = first_attn._local_window_indices(shard)
            shared.main_rope_cos_sin = self.rotary_emb(position_ids, compressed=False)
            shared.compress_rope_cos_sin = self.rotary_emb(position_ids, compressed=True)

        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)

        router_logits: list[torch.Tensor] = []
        indexer_terms: list[IndexerLossTerms] = []
        draft_targets = set(self.config.dspark_target_layer_ids)
        draft_inputs: dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.layers):
            layer.mlp._recompute_routing_cache = None
            layer.mlp._enable_recompute_cache = bool(getattr(layer, "gradient_checkpointing", False) and self.training)
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer._recompute_indexer_cache = None
                layer.self_attn.indexer._enable_recompute_cache = layer.mlp._enable_recompute_cache

            layer_ngram = None if layer.engram is None else ngram_embeds[layer.engram.layer_hash_index]
            capture_draft = return_draft_hiddens and layer_idx in draft_targets

            # A copy per layer: gradient checkpointing replays each layer during
            # backward, after every later source layer has republished. On one
            # shared object the replay would read a later layer's KV and top-k.
            shared = dataclasses.replace(shared)
            hidden_streams, pre_mix, layer_logits, indexer_loss, draft_input = layer(
                hidden_streams,
                pre_mix,
                position_ids,
                self.rotary_emb,
                shared,
                plans,
                image_mask,
                shard=shard,
                ngram_embed=layer_ngram,
                token_mask=token_mask if layer_ngram is not None else None,
                capture_draft_input=capture_draft,
            )
            if capture_draft:
                draft_inputs[layer_idx] = draft_input
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

        outputs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
        }
        if indexer_terms:
            kl_total = torch.stack([term.kl for term in indexer_terms]).sum()
            uniform_total = torch.stack([term.uniform for term in indexer_terms]).sum()
            query_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            if shard is not None:
                # Each rank sums only its own rows. Scaled by the group size,
                # the ranks' gradients add up to size times the full-row one --
                # what FSDP's division by dp * sp expects, and what the output
                # gather already gives every other parameter -- and the
                # fsdp-group mean of the logged value is the full-row KL.
                kl_total = kl_total * shard.size
                uniform_total = uniform_total * shard.size
                query_tokens = hidden_states.shape[0] * shard.valid_len
            outputs["indexer_kl_total"] = kl_total
            outputs["indexer_uniform_total"] = uniform_total
            # The count travels so the *metric* can be a per-layer mean while the
            # objective stays a sum: dividing the sum by the layer count here would
            # make the gradient each indexer receives depend on how many other
            # layers happened to carry one.
            outputs["indexer_kl_layers"] = len(indexer_terms)
            outputs["indexer_query_tokens"] = query_tokens
        if return_draft_hiddens:
            outputs["draft_hiddens"] = [draft_inputs[idx] for idx in self.config.dspark_target_layer_ids]
        return outputs


class DeepseekV41Llm(PretrainedOmniModule):
    """``deepseek_v41_llm`` -- the CED backbone as an OmniModule.

    Takes ``inputs_embeds`` because the embedding belongs to the sibling text
    encoder, and returns ``hidden_states`` for it to project back to logits.
    """

    config_class = DeepseekV41LlmConfig
    # ``SequenceShard`` already gives each CP rank its own row slice; ``forward_post``
    # writes that slice back item by item instead of gathering the full row.
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_llm"
    main_input_name = "inputs_embeds"
    # Each Engram connection lives inside its ``DeepseekV41DecoderLayer`` and
    # DSpark / MTP lives in its own ``deepseek_v41_mtp`` OmniModule, so the
    # FSDP2 module tree here is a strictly uniform 40-layer decoder stack.
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    supports_gradient_checkpointing = True

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # DeepSeek-V4.1 backbone dispatches DSA/MQA attention via VeomniOp rather than HF ALL_ATTENTION_FUNCTIONS.
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: DeepseekV41Llm):
        """Dequant fp8/fp4 into bf16 when this module was built for training."""
        from ..checkpoint_tensor_converter import create_deepseek_v41_llm_checkpoint_tensor_converter

        return create_deepseek_v41_llm_checkpoint_tensor_converter(model)

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41Backbone(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Fill the raw ``Parameter``s HF's Linear/Embedding init does not see.

        mHC ``fn``/``base``/``scale``, the router, and stacked expert slabs are
        ``torch.empty`` until this runs. V4 does the same in its generated
        ``_init_weights``; without it a random-init toy Sinkhorn/MoE is a lottery.
        Checkpoint load overwrites these.
        """
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41HyperConnection):
            nn.init.normal_(module.fn, mean=0.0, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Experts) and not getattr(module, "experts_are_quantized", False):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)

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

    def freeze_model(self) -> None:
        """Freeze backbone layers that are not selected for training.

        ``n_trainable_layers=N`` keeps ``layers[:N]`` (the 8xH20 default).
        ``trainable_layer_ids`` names an explicit set — use it to keep specific
        layers (e.g., DSpark target layers) instead of the prefix. ``None`` on
        both trains the whole backbone stack.

        Engram connections live on their decoder layer, so freezing ``layers.14``
        freezes ``layers.14.engram`` automatically. The final RMSNorm rides the
        frozen tail. DSpark / MTP lives in ``deepseek_v41_mtp`` with its own
        ``freeze`` policy.
        """
        trainable = self._trainable_backbone_layer_ids()
        if trainable is None:
            return
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx not in trainable:
                layer.requires_grad_(False)
        last = self.config.num_hidden_layers - 1
        if last not in trainable:
            self.model.norm.requires_grad_(False)

    def _trainable_backbone_layer_ids(self) -> set | None:
        if self.config.trainable_layer_ids is not None:
            return set(self.config.trainable_layer_ids)
        n = self.config.n_trainable_layers
        if n is None:
            return None
        return set(range(n))

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        ngram_embeds: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        return_draft_hiddens: bool | None = None,
        sequence_shard: SequenceShard | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

        ``sequence_shard`` is set by the training hook under sequence
        parallelism; every per-token input and output is then this rank's rows.
        """
        del kwargs
        if inputs_embeds is None:
            raise ValueError("deepseek_v41_llm requires inputs_embeds; it holds no token embedding.")
        if position_ids is None:
            raise ValueError(
                "deepseek_v41_llm requires position_ids: they carry the packed sample boundaries, and "
                "assuming a single arange here would compress across them."
            )
        if return_draft_hiddens is None:
            return_draft_hiddens = bool(
                self.config.emit_draft_hiddens
                and self.config.dspark_target_layer_ids
                and self.config.dspark_block_size > 0
            )
        return self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            return_draft_hiddens=return_draft_hiddens,
            shard=sequence_shard,
        )


__all__ = [
    "DeepseekV41Backbone",
    "DeepseekV41Llm",
]
