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

"""Graph hooks, parallel plan and metering for the DeepSeek-V4.1 backbone.

The packing follows the shared AR path (:func:`pack_llm_conversations_for_forward`
/ :func:`scatter_llm_hidden_states`), because V4.1's backbone consumes exactly
what that produces: pre-embedded segments in one varlen row with per-sample
``position_ids`` restarting at zero. Two things are added on top, and both exist
because they must be derived from the *same* segmentation the embeddings were:

* the **image mask**, built from the conversation items' types alongside the
  pack rather than from a separate pass. It drives the MoE's VL routing bias and
  (negated) the Engram gate, and a mask that disagreed with the packing would
  route the wrong tokens without changing any shape.
* the **n-gram rows**, which arrive from the ``deepseek_v41_ngram`` node and are
  length-checked against the packed row here. They are positional along the
  sequence, so a stale or differently-packed tensor would pair every token with
  another token's memory.

**Sequence parallelism** does not use the shared path's attention all-to-all.
The compression groups, the candidate blocks and the compressed-history
visibility are built from the whole row in Python before any kernel runs, so
``forward_pre`` hands the backbone a :class:`SequenceShard` carrying the full
positions alongside this rank's rows, and the attention gathers its one KV
head instead of scattering query heads (see :mod:`..sequence_shard`).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed._tensor import Shard

from .......distributed.parallel_plan import ParallelPlan
from .......distributed.parallel_state import get_parallel_state
from .......utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens
from .......utils.tensor_utils import unflatten
from .....mixins.base_mixin import BaseMixin
from .....mixins.metric_meter_mixin import MetricMeterMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, is_dummy
from ....base.llm_packing import pack_llm_conversations_for_forward, scatter_llm_hidden_states
from ...ngram.packing import pack_ngram_inputs
from ...quant_scheme import quantized_storage_dtype
from ..configuration import DeepseekV41LlmConfig
from ..mhc import veomni_mhc_post
from ..modeling import DeepseekV41Llm
from ..quantization import veomni_qat_implementation
from ..reuse import veomni_moe_experts_forward
from ..reuse import veomni_qat_implementation as veomni_v4_expert_qat_implementation
from ..sequence_shard import SequenceShard


def build_packed_image_mask(
    conversations: List[List[ConversationItem]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """``[1, total]`` bool over the packed row, true inside an image span.

    Walks the conversation in the same order and with the same dummy-skipping as
    :func:`pack_llm_conversations_for_forward`, so the mask lines up with the
    embeddings by construction rather than by a length that happens to agree.

    Returns ``None`` for a text-only batch. That is not the same as an all-false
    mask to the consumers: both the router's bias select and the Engram gate's
    mask skip entirely on ``None``, and there is no point paying for either when
    no token is an image.
    """
    flags: List[torch.Tensor] = []
    saw_image = False
    for sample in conversations:
        for item in sample:
            if is_dummy(item):
                continue
            length = item.value.size(0)
            is_image = item.type == "image"
            saw_image |= is_image
            flags.append(torch.full((length,), is_image, dtype=torch.bool, device=device))
    if not saw_image:
        return None
    return torch.cat(flags).unsqueeze(0)


def fold_dummy_anchors(
    inputs_embeds: torch.Tensor,
    conversations: List[List[ConversationItem]],
) -> torch.Tensor:
    """Add ``0 * mean`` of every ``dummy`` item so its encoder stays in the backward.

    The packer skips dummy items, so without this a text-only micro-batch drops
    the ViT's output before the loss. The ViT's FSDP unit then never runs its
    backward on that rank while ranks holding a real image do, and the two sides
    wait on each other's collectives forever.
    """
    for sample in conversations:
        for item in sample:
            if is_dummy(item) and isinstance(item.value, torch.Tensor):
                inputs_embeds = inputs_embeds + item.value.mean().to(inputs_embeds.device, inputs_embeds.dtype) * 0
    return inputs_embeds


def fold_indexer_kl_loss(outputs: Dict[str, Any], coef: float) -> Optional[torch.Tensor]:
    """Mean indexer KL, scaled by ``dsa_indexer_loss_coef``, or None when the objective is off.

    The backbone sums KL over query rows so each source layer's gradient does not
    depend on how many other layers built a term. The language-model loss is a
    mean, so the fold-in uses the same mean -- matching
    ``DeepseekV4ForCausalLM`` -- rather than adding a sum onto a mean.
    """
    total = outputs.get("indexer_kl_total")
    if total is None or coef <= 0:
        return None
    tokens = outputs.get("indexer_query_tokens")
    denom = 1 if not tokens else max(int(tokens), 1)
    return coef * (total / denom)


def shard_packed_row(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    ngram_embeds: Optional[List[torch.Tensor]],
    image_mask: Optional[torch.Tensor],
    group: Any,
    rank: int,
    size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor], SequenceShard]:
    """Pad the packed row to a multiple of ``size`` and keep this rank's rows.

    Every SP rank holds the same packed row. The pad is appended as one more
    packed sample, positions restarting at zero: sample boundaries are read off
    the positions, so no real token can attend to it. Zeros at every pad
    position would instead make each pad token its own sample.
    """

    def pad_rows(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x.new_zeros((x.shape[0], pad, *x.shape[2:]))], dim=1)

    valid_len = inputs_embeds.shape[1]
    pad = -valid_len % size
    if pad:
        tail = torch.arange(pad, device=position_ids.device, dtype=position_ids.dtype)
        position_ids = torch.cat([position_ids, tail.expand(position_ids.shape[0], -1)], dim=1)
        inputs_embeds = pad_rows(inputs_embeds)
        ngram_embeds = None if ngram_embeds is None else [pad_rows(rows) for rows in ngram_embeds]
        image_mask = None if image_mask is None else pad_rows(image_mask)
    shard = SequenceShard(group=group, rank=rank, size=size, position_ids=position_ids, valid_len=valid_len)
    return (
        shard.local_rows(inputs_embeds),
        shard.local_rows(position_ids),
        None if ngram_embeds is None else [shard.local_rows(rows) for rows in ngram_embeds],
        None if image_mask is None else shard.local_rows(image_mask),
        shard,
    )


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks: pack in, scatter out, thread the n-gram rows."""

    config: DeepseekV41LlmConfig
    device: torch.device
    training: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Optional[List[List[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        self._sequence_shard: Optional[SequenceShard] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        ngram_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        inputs_embeds, _, position_ids, inputs_embeds_shape = pack_llm_conversations_for_forward(
            conversation_list, self.device
        )
        if self.training:
            inputs_embeds = fold_dummy_anchors(inputs_embeds, conversation_list)
        image_mask = build_packed_image_mask(conversation_list, self.device)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = inputs_embeds_shape

        ngram_embeds = self._align_ngram_embeds(ngram_embeds, inputs_embeds.shape[1])
        if self.dspark_trains:
            # Image rows never carry a DSpark label, so the id they pack to is arbitrary.
            kwargs["dspark_input_ids"], _ = pack_ngram_inputs(conversation_list, self.device, pad_token_id=0)

        # This rank's own per-sample lengths, read off the packed position ids --
        # the same tensor the backbone derives its sample boundaries from, so the
        # metered lengths cannot drift from the ones actually computed.
        (cu_seq_lens_q, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
        self.metric_meter_set_seqlens(
            "forward", [int(s) for s in valid_seqlens_from_cu_seqlens(cu_seq_lens_q).tolist()]
        )

        ps = get_parallel_state()
        if ps.sp_size > 1:
            inputs_embeds, position_ids, ngram_embeds, image_mask, self._sequence_shard = shard_packed_row(
                inputs_embeds, position_ids, ngram_embeds, image_mask, ps.sp_group, ps.sp_rank, ps.sp_size
            )
            kwargs["sequence_shard"] = self._sequence_shard

        return dict(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            ngram_embeds=ngram_embeds,
            image_mask=image_mask,
            **kwargs,
        )

    def _align_ngram_embeds(
        self,
        ngram_embeds: Optional[List[torch.Tensor]],
        packed_length: int,
    ) -> Optional[List[torch.Tensor]]:
        """Check the n-gram rows against the row they will be indexed alongside.

        The rows are positional: entry ``t`` is the memory for token ``t`` of the
        packed sequence. A length mismatch means the ngram node saw a different
        packing than this one did, which pairs every token past the divergence
        with another token's memory -- a wrong answer with the right shape at
        every layer that reads it.
        """
        if ngram_embeds is None:
            return None
        aligned = []
        for index, rows in enumerate(ngram_embeds):
            if rows.shape[1] != packed_length:
                raise ValueError(
                    f"ngram_embeds[{index}] covers {rows.shape[1]} positions but the packed row has "
                    f"{packed_length}. The deepseek_v41_ngram node must be given the same conversation "
                    "packing as this module; the rows are positional and a mismatch silently pairs each "
                    "token with another token's n-gram memory."
                )
            aligned.append(rows.to(self.device))
        return aligned

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs["hidden_states"]
        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        shard = self._sequence_shard
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None
        self._sequence_shard = None

        dspark_rows = {
            meta_key: outputs[key]
            for key, meta_key in (
                ("dspark_hidden_states", "dspark_hidden"),
                ("dspark_markov_bias", "dspark_markov_bias"),
            )
            if outputs.get(key) is not None
        }
        if shard is not None:
            # Every SP rank scores the full row downstream, so each one needs
            # all of it back. The gather's summed backward is the size-times
            # convention FSDP's division by dp * sp undoes.
            hidden_states = shard.gather(hidden_states).narrow(1, 0, shard.valid_len)
            dspark_rows = {key: shard.gather(rows).narrow(1, 0, shard.valid_len) for key, rows in dspark_rows.items()}

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        scatter_llm_hidden_states(conversation, unflatten(hidden_states, pack_shape))

        for meta_key, rows in dspark_rows.items():
            if rows.dim() == 3 and rows.size(0) == 1:
                rows = rows.squeeze(0)
            row_iter = iter(unflatten(rows, pack_shape))
            for sample in conversation:
                for part in sample:
                    if is_dummy(part):
                        continue
                    part.meta[meta_key] = next(row_iter)
            if next(row_iter, None) is not None:
                raise RuntimeError(f"{meta_key} segment count exceeds non-dummy conversation items.")

        forwarded: Dict[str, Any] = {"conversation_list": conversation}
        # OmniModel only collects ``_loss``. Leaving the KL tensors on the batch
        # would hand them to ``text_encoder.decode`` as unexpected kwargs; folding
        # here is how the indexer gets a gradient at all -- top-k does not
        # backprop into the scores it ranked.
        kl_loss = fold_indexer_kl_loss(outputs, self.config.dsa_indexer_loss_coef)
        if kl_loss is not None:
            forwarded["_loss"] = kl_loss
        return forwarded


class MeterMixin(MetricMeterMixin):
    """Per-module training meter.

    V4.1's attention is **linear** in sequence length, not quadratic: every query
    attends to ``sliding_window`` raw entries plus at most ``index_topk``
    compressed ones, regardless of how long the sequence is. Reusing the dense
    backbones' ``12 * seqlen^2 * ...`` term would report an MFU that grows with
    context on a model whose cost does not -- at 32k tokens and the released
    settings it overstates the attention term by more than an order of magnitude.

    The indexer is the one term that does grow with the square, and only on the
    ``index_source_layer_ids``: scoring every query against every reachable
    compressed slot is ``S * (S / ratio)`` before the top-k. That is why it is
    counted separately rather than folded into the attention.
    """

    config: DeepseekV41LlmConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        config = self.config
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = config.head_dim
        layers = config.num_hidden_layers

        # Per-token linear work, per layer.
        attn_linear = (
            hidden * config.q_lora_rank  # wq_a
            + config.q_lora_rank * heads * head_dim  # wq_b
            + hidden * head_dim  # wkv
            + heads * head_dim * config.o_lora_rank  # wo_a, block-diagonal
            + config.o_groups * config.o_lora_rank * hidden  # wo_b
        )
        # Routed experts cost only the activated ones; the shared expert runs for
        # every token; the router projects to every expert.
        moe = (
            config.num_experts_per_tok * 3 * hidden * config.moe_intermediate_size
            + config.n_shared_experts * 3 * hidden * config.moe_intermediate_size
            + hidden * config.n_routed_experts
        )
        # Both mHC sites project the flattened hc_mult stream once per token.
        mhc = 2 * (2 + config.hc_mult) * config.hc_mult * config.hc_mult * hidden
        per_token = (attn_linear + moe + mhc) * layers

        # The Engram connections, on their layers only.
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        per_token += (
            len(config.engram_layer_ids) * n_hash_cols * config.engram_head_dim * hidden * (config.hc_mult + 1)
        )

        tokens = sum(seqlens)
        dense_flops = 6 * per_token * tokens

        # Sparse attention: a bounded number of KV entries per query, so linear.
        attended = config.sliding_window + config.index_topk
        attn_flops = 12 * tokens * attended * head_dim * heads * layers

        # The eager indexer, on its own layers: S queries x S/ratio slots.
        ratios = [r for r in config.compress_ratios[:layers] if r > 0]
        mean_ratio = (sum(ratios) / len(ratios)) if ratios else 1.0
        index_pairs = sum(s * s / mean_ratio for s in seqlens)
        indexer_flops = (
            12 * index_pairs * config.index_head_dim * config.index_n_heads * len(config.index_source_layer_ids)
        )

        return (dense_flops + attn_flops + indexer_flops) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """Family assembly plus the expert-parallel plan.

    No ``InferenceMixin``: the attention stack carries no KV cache, so there is
    no decode path to accelerate. Generation belongs to the ``deepseek_v41``
    generation graph and is not wired yet.
    """

    def get_parallel_plan(self) -> ParallelPlan:
        # fqns are module-local: the stack lives under ``self.model``, which is the
        # bare backbone rather than another ``ForCausalLM`` wrapper.
        ep_plan = {
            "model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "model.layers.*.mlp.experts.down_proj": Shard(0),
            "model.mtp.*.mlp.experts.gate_up_proj": Shard(0),
            "model.mtp.*.mlp.experts.down_proj": Shard(0),
        }
        # A quantized expert weight is two tensors, and both are indexed by the
        # expert on dim 0. Sharding the slab without its scales would leave
        # ~15.6 GiB of scales replicated on every rank -- enough, together with
        # the n-gram tables, to put eight H20s over their 97 GiB.
        #
        # Added only when they exist. A bf16 training module has no scale
        # Parameters -- the load converter spent them -- and a plan entry that
        # resolves to nothing is exactly what a stale prefix looks like -- the
        # thing `test_plan_fqns_resolve_against_the_real_module_tree` exists to
        # catch.
        if quantized_storage_dtype(self.config) is not None:
            ep_plan["model.layers.*.mlp.experts.gate_up_proj_scale"] = Shard(0)
            ep_plan["model.layers.*.mlp.experts.down_proj_scale"] = Shard(0)
            ep_plan["model.mtp.*.mlp.experts.gate_up_proj_scale"] = Shard(0)
            ep_plan["model.mtp.*.mlp.experts.down_proj_scale"] = Shard(0)
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class DeepseekV41LlmAccelerated(VeOmniMixin, DeepseekV41Llm):
    pass


__all__ = [
    "DeepseekV41LlmAccelerated",
    "build_packed_image_mask",
    "fold_indexer_kl_loss",
    "veomni_mhc_post",
    "veomni_moe_experts_forward",
    "veomni_qat_implementation",
    "veomni_v4_expert_qat_implementation",
]
