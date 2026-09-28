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

Generation is ``DeepseekV41LlmInferenceMixin``. Training still uses this
module's full-sequence ``forward``. Prefill capture is
:meth:`DeepseekV41Backbone.forward_ced_prefill`: encoder on the full prompt,
decoder only the last sliding window. Each AR step is ``decode_backbone_step``;
a DSpark hit verifies the rest of the block and rolls the KV back on a miss.
There is no HF ``past_key_values``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ....omni_pretrained_model import OmniPreTrainedModel
from .attention import (
    DeepseekV41AttentionCache,
    DeepseekV41RotaryEmbedding,
    IndexerLossTerms,
    SharedAttentionState,
    build_compression_plans,
    sequence_slices_from_positions,
)
from .configuration import DeepseekV41LlmConfig
from .decoder_layer import DeepseekV41DecoderLayer
from .dspark import build_dspark_blocks
from .engram_connection import build_engram_connections
from .inference import DeepseekV41LlmInferenceMixin
from .mhc import DeepseekV41HyperConnection, expand_to_streams, identity_pre_mix, mhc_collapse, veomni_mhc_post
from .moe import DeepseekV41TopKRouter
from .quantization import veomni_qat_implementation
from .reuse import DeepseekV4Experts, DeepseekV4RMSNorm, veomni_moe_experts_forward
from .reuse import veomni_qat_implementation as veomni_v4_expert_qat_implementation


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
        self.mtp = build_dspark_blocks(config)
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
        capture: Optional[List[DeepseekV41AttentionCache]] = None,
        capture_seq_len: Optional[int] = None,
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
        if capture is not None and len(capture) != len(self.layers):
            raise ValueError(
                f"capture has {len(capture)} entries but the backbone has {len(self.layers)} layers. "
                "Prefill writes one cache slot per layer, matching DeepseekV41DecodeCache.empty."
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

        router_logits: List[torch.Tensor] = []
        indexer_terms: List[IndexerLossTerms] = []
        draft_targets = set(self.config.dspark_target_layer_ids)
        draft_inputs: Dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.layers):
            if str(layer_idx) in self.engram:
                # Before the layer, not after: the reference adds the lookup to the
                # stream at the top of the loop body, so this layer's attention is
                # the first thing that sees it. Writing it after would hand the
                # n-gram to the *next* layer and leave this one reading a stream
                # the trained weights never produced.
                connection = self.engram[str(layer_idx)]
                hidden_streams = connection(hidden_streams, ngram_embeds[connection.layer_hash_index], token_mask)
            if (return_draft_hiddens or self.mtp) and layer_idx in draft_targets:
                # The MTP head reads its target layer's *input*, not its output --
                # `main_hiddens.append(h.mean(dim=2))` sits above `layer(...)` in
                # the reference. Stored by layer id so a padded 3-wide ``main_proj``
                # can reuse the last decoder stream twice.
                draft_inputs[layer_idx] = hidden_streams.mean(dim=2)

            hidden_streams, pre_mix, layer_logits, indexer_loss = layer(
                hidden_streams,
                pre_mix,
                position_ids,
                self.rotary_emb,
                shared,
                plans,
                image_mask,
                capture=None if capture is None else capture[layer_idx],
                capture_seq_len=capture_seq_len,
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
        if return_draft_hiddens or self.mtp:
            outputs["draft_hiddens"] = [draft_inputs[idx] for idx in self.config.dspark_target_layer_ids]
        return outputs

    def forward_ced_prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        image_mask: Optional[torch.Tensor] = None,
        capture: Optional[List[DeepseekV41AttentionCache]] = None,
        capture_seq_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Prefill capture: encoder on the full prompt, decoder on the last window.

        Training uses :meth:`forward`. Official decode still runs every layer
        (16B); this only cuts decoder *prefill* to the sliding window.
        """
        if capture is None:
            raise ValueError("forward_ced_prefill writes decode caches; capture is required.")
        if len(capture) != len(self.layers):
            raise ValueError(f"capture has {len(capture)} entries but the backbone has {len(self.layers)} layers.")
        if ngram_embeds is None and self.engram:
            raise ValueError(
                f"this backbone has Engram connections at layers {list(self.config.engram_layer_ids)} but no "
                "ngram_embeds were passed."
            )
        if ngram_embeds is not None and len(ngram_embeds) != len(self.config.engram_layer_ids):
            raise ValueError(
                f"ngram_embeds has {len(ngram_embeds)} entries but engram_layer_ids names "
                f"{len(self.config.engram_layer_ids)} layers."
            )
        token_mask = None if image_mask is None else ~image_mask
        full_position_ids = position_ids
        valid_len = int(inputs_embeds.shape[1]) if capture_seq_len is None else int(capture_seq_len)
        split = int(self.config.candidate_source_layer_id)
        window = int(self.config.sliding_window)
        if not (0 < split < len(self.layers)):
            raise ValueError(f"forward_ced_prefill needs 0 < candidate_source_layer_id < n_layers, got {split}.")
        if valid_len <= window:
            raise ValueError(
                f"forward_ced_prefill needs a prompt longer than sliding_window={window}, got {valid_len}."
            )
        plans = build_compression_plans(self.config, position_ids, sequence_slices_from_positions(position_ids))
        shared = SharedAttentionState()
        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)
        draft_inputs: Dict[int, torch.Tensor] = {}
        router_logits: List[torch.Tensor] = []
        indexer_terms: List[IndexerLossTerms] = []
        draft_targets = set(self.config.dspark_target_layer_ids)

        def run_layer(
            layer_idx: int,
            *,
            layer_plans: Dict,
            layer_capture_seq_len: Optional[int],
            layer_ngram: Optional[List[torch.Tensor]],
            layer_image_mask: Optional[torch.Tensor],
            layer_token_mask: Optional[torch.Tensor],
            layer_position_ids: torch.Tensor,
            compress_hidden: Optional[torch.Tensor] = None,
            compress_position_ids: Optional[torch.Tensor] = None,
        ) -> None:
            nonlocal hidden_streams, pre_mix
            layer = self.layers[layer_idx]
            if str(layer_idx) in self.engram:
                connection = self.engram[str(layer_idx)]
                hidden_streams = connection(hidden_streams, layer_ngram[connection.layer_hash_index], layer_token_mask)
            if self.mtp and layer_idx in draft_targets:
                draft_inputs[layer_idx] = hidden_streams.mean(dim=2)
            hidden_streams, pre_mix, layer_logits, indexer_loss = layer(
                hidden_streams,
                pre_mix,
                layer_position_ids,
                self.rotary_emb,
                shared,
                layer_plans,
                layer_image_mask,
                capture=capture[layer_idx],
                capture_seq_len=layer_capture_seq_len,
                compress_hidden=compress_hidden,
                compress_position_ids=compress_position_ids,
            )
            router_logits.append(layer_logits)
            if indexer_loss is not None:
                indexer_terms.append(indexer_loss)

        for layer_idx in range(split):
            run_layer(
                layer_idx,
                layer_plans=plans,
                layer_capture_seq_len=capture_seq_len,
                layer_ngram=ngram_embeds,
                layer_image_mask=image_mask,
                layer_token_mask=token_mask,
                layer_position_ids=position_ids,
            )

        boundary = self.layers[split]
        compress_hidden = boundary.input_layernorm(mhc_collapse(hidden_streams, pre_mix))
        start = valid_len - window
        hidden_streams = hidden_streams[:, -window:]
        pre_mix = pre_mix[:, -window:]
        position_ids = full_position_ids[:, -window:]
        image_mask = None if image_mask is None else image_mask[:, -window:]
        token_mask = None if token_mask is None else token_mask[:, -window:]
        ngram_embeds = None if ngram_embeds is None else [row[:, -window:] for row in ngram_embeds]
        decoder_plans = {ratio: plan.narrow_queries(start, valid_len) for ratio, plan in plans.items()}
        for layer_idx in range(split, len(self.layers)):
            run_layer(
                layer_idx,
                layer_plans=decoder_plans,
                layer_capture_seq_len=window,
                layer_ngram=ngram_embeds,
                layer_image_mask=image_mask,
                layer_token_mask=token_mask,
                layer_position_ids=position_ids,
                compress_hidden=compress_hidden if layer_idx == split else None,
                compress_position_ids=full_position_ids if layer_idx == split else None,
            )
        draft_hiddens = [draft_inputs[idx] for idx in self.config.dspark_target_layer_ids] if self.mtp else []
        if draft_hiddens:
            width = hidden_streams.shape[1]
            draft_hiddens = [hidden[:, -width:] for hidden in draft_hiddens]

        hidden_states = self.norm(mhc_collapse(hidden_streams, pre_mix))
        outputs: Dict[str, Any] = {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
        }
        if indexer_terms:
            outputs["indexer_kl_total"] = torch.stack([term.kl for term in indexer_terms]).sum()
            outputs["indexer_uniform_total"] = torch.stack([term.uniform for term in indexer_terms]).sum()
            outputs["indexer_kl_layers"] = len(indexer_terms)
            outputs["indexer_query_tokens"] = hidden_states.shape[0] * hidden_states.shape[1]
        if draft_hiddens:
            outputs["draft_hiddens"] = draft_hiddens
        return outputs

    def seed_dspark_windows(
        self,
        draft_hiddens: List[torch.Tensor],
        position_ids: torch.Tensor,
        caches: List[DeepseekV41AttentionCache],
    ) -> Optional[torch.Tensor]:
        """Write each DSpark stage's window KV from the backbone's target-layer inputs."""
        if not self.mtp:
            return None
        if len(draft_hiddens) != len(self.config.dspark_target_layer_ids):
            raise ValueError(
                f"DSpark expected {len(self.config.dspark_target_layer_ids)} target-layer hiddens, "
                f"got {len(draft_hiddens)}"
            )
        if len(caches) != len(self.mtp):
            raise ValueError(f"DSpark cache has {len(caches)} slots but there are {len(self.mtp)} stages")
        main_x = self.mtp[0].project_main(draft_hiddens)
        for stage, cache in zip(self.mtp, caches):
            stage.seed_window(main_x, position_ids, self.rotary_emb, cache)
        return main_x

    def forward_dspark_train(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        draft_hiddens: List[torch.Tensor],
    ) -> torch.Tensor:
        """Teacher-forced DSpark on the packed row. Returns ``[B, S, hidden]``.

        Decode runs one 5-token block after a verified token. Training cannot
        afford that loop, so the three stages run once over the packed sequence
        with causal sliding-window attention. Same modules as ``forward_spec``;
        different position arithmetic (see ``forward_draft_packed``).
        """
        if not self.mtp:
            raise RuntimeError("forward_dspark_train was called but this backbone has no DSpark stages")
        if len(draft_hiddens) != len(self.config.dspark_target_layer_ids):
            raise ValueError(
                f"DSpark expected {len(self.config.dspark_target_layer_ids)} target-layer hiddens, "
                f"got {len(draft_hiddens)}"
            )
        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)
        main: Any = draft_hiddens
        draft = None
        for stage in self.mtp:
            hidden_streams, pre_mix, main, draft = stage(hidden_streams, pre_mix, position_ids, self.rotary_emb, main)
        return draft

    def forward_spec(
        self,
        draft_embeds: torch.Tensor,
        position: int,
        caches: List[DeepseekV41AttentionCache],
        main_x: torch.Tensor,
    ) -> torch.Tensor:
        """One DSpark block after a verified token, matching official ``forward_spec``.

        ``draft_embeds`` is ``[B, block_size, D]`` -- the sampled token in slot 0
        and noise tokens after. ``position`` is the verified token's index.
        ``main_x`` is ``project_main`` of that token's target-layer inputs.
        """
        if not self.mtp:
            raise RuntimeError("forward_spec was called but this backbone has no DSpark stages")
        if draft_embeds.dim() == 2:
            draft_embeds = draft_embeds.unsqueeze(0)
        if main_x.dim() == 2:
            main_x = main_x.unsqueeze(0)
        batch = draft_embeds.shape[0]
        position_ids = torch.full((batch, 1), position, dtype=torch.long, device=draft_embeds.device)
        hidden_streams = expand_to_streams(draft_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)
        for stage, cache in zip(self.mtp, caches):
            hidden_streams, pre_mix = stage.forward_draft(
                hidden_streams, pre_mix, position_ids, self.rotary_emb, cache, main_x
            )
        return self.mtp[-1].collapse_draft(hidden_streams, pre_mix)


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
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41EngramConnection", "DeepseekV41DSparkBlock"]
    supports_gradient_checkpointing = True

    @staticmethod
    def _create_checkpoint_tensor_converter(model: "DeepseekV41Llm"):
        """Dequant fp8/fp4 into bf16 when this module was built for training."""
        from ..checkpoint_tensor_converter import create_deepseek_v41_llm_checkpoint_tensor_converter

        return create_deepseek_v41_llm_checkpoint_tensor_converter(model)

    def __init__(self, config: DeepseekV41LlmConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41Backbone(config)
        self._decode_cache = None
        self._next_position = 0
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Fill the raw ``Parameter``s HF's Linear/Embedding init does not see.

        mHC ``fn``/``base``/``scale``, the router, and stacked expert slabs are
        ``torch.empty`` until this runs. V4 does the same in its generated
        ``_init_weights``; without it a random-init toy Sinkhorn/MoE is a lottery.
        Checkpoint load overwrites these.
        """
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
        ``trainable_layer_ids`` names an explicit set — use it to keep the
        DSpark target layers instead of the prefix. ``None`` on both trains
        the whole stack, DSpark included.

        Engram connections follow their layer index, not the decoder-layer
        wrap, so a connection at layer 14 is frozen when that id is not
        selected even though it is not a child of ``layers.14``. The final
        RMSNorm rides the frozen tail. Partial backbone freeze freezes
        DSpark unless ``freeze_mtp=False``: three draft stages are ~14 B,
        and AdamW on them does not fit 8×H20 next to a 2-layer backbone.
        """
        trainable = self._trainable_backbone_layer_ids()
        if trainable is None:
            return
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx not in trainable:
                layer.requires_grad_(False)
        for key, connection in self.model.engram.items():
            if int(key) not in trainable:
                connection.requires_grad_(False)
        freeze_mtp = True if self.config.freeze_mtp is None else self.config.freeze_mtp
        if freeze_mtp:
            self.model.mtp.requires_grad_(False)
        last = self.config.num_hidden_layers - 1
        if last not in trainable:
            self.model.norm.requires_grad_(False)

    def _trainable_backbone_layer_ids(self) -> Optional[set]:
        if self.config.trainable_layer_ids is not None:
            return set(self.config.trainable_layer_ids)
        n = self.config.n_trainable_layers
        if n is None:
            return None
        return set(range(n))

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        image_mask: Optional[torch.Tensor] = None,
        capture: Optional[List[DeepseekV41AttentionCache]] = None,
        capture_seq_len: Optional[int] = None,
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
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            capture=capture,
            capture_seq_len=capture_seq_len,
            return_draft_hiddens=return_draft_hiddens,
        )
        mtp_trainable = bool(self.model.mtp) and any(p.requires_grad for p in self.model.mtp.parameters())
        if self.training and mtp_trainable:
            draft_hiddens = outputs.get("draft_hiddens")
            if not draft_hiddens:
                raise RuntimeError(
                    "DSpark is trainable but the backbone did not collect draft_hiddens. "
                    "dspark_target_layer_ids must name live backbone layers."
                )
            outputs["dspark_hidden_states"] = self.model.forward_dspark_train(
                inputs_embeds, position_ids, draft_hiddens
            )
        return outputs


__all__ = [
    "DeepseekV41Backbone",
    "DeepseekV41Llm",
    "veomni_mhc_post",
    "veomni_moe_experts_forward",
    "veomni_qat_implementation",
    "veomni_v4_expert_qat_implementation",
]
