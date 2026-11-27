"""VeOmni-accelerated Qwen3OmniLlm — training-graph hooks only.

``generate()`` and inference-state management live on the native
:class:`Qwen3OmniLlm` in ``modeling.py``; FSDP/DDP/SP need no different
inference path today.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.distributed.tensor.placement_types import Shard

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad
from veomni.utils.tensor_utils import unflatten

from .......distributed.parallel_plan import ParallelPlan
from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.llm_packing import item_token_len
from ....base.sp_shard import (
    gather_sp_sharded_groups,
    modality_groups,
    pad_sp_shard,
    single_modality_shard_batch,
    split_sp_sharded_tensor,
)
from ...carrier import is_thinker_input, scatter_hidden_states
from ..configuration import Qwen3OmniLlmConfig
from ..modeling import (
    THINKER_EMBED,
    THINKER_HIDDEN,
    Qwen3OmniLlm,
    pack_qwen3omni_conversations_for_forward,
)


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the Qwen3-Omni thinker backbone (training path).

    Packs every backbone-input ``conversation_list`` item's embedding segment into
    one bs=1 varlen sequence, rebuilds 3-row TMRoPE position ids (text / audio runs +
    per-image grid positions), marks visual positions for DeepStack, and threads
    the per-layer DeepStack features into the thinker text model.
    """

    config: Qwen3OmniLlmConfig
    device: torch.device
    training: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None
        # Active sample's pre-pad packed length. Under SP the output-gather hook
        # narrows the all-gathered (padded) output back to it.
        self._sp_own_len: Optional[int] = None
        # Pre-slice row count for the router-logits gather (see
        # ``_full_sequence_router_logits``).
        self._sp_router_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ps = get_parallel_state()
        sharded = ps.sp_size > 1 and ps.skip_sp_gather_between_modules
        if sharded and not single_modality_shard_batch(conversation_list):
            # Mixed producers: each one sliced only the subsequence *it* wrote
            # (text in carrier order, then images, then videos), so concatenating
            # their windows does not reconstruct this rank's window over the packed
            # row — the producers' window widths differ from the packed one. Gather
            # each producer back to full length and let the ordinary packer slice
            # the reassembled row below. `single_modality_shard_batch` is the case
            # where the windows already concatenate exactly, and is untouched.
            gather_sp_sharded_groups(
                modality_groups(conversation_list),
                ps.sp_group,
                meta_list_keys=("deepstack",),
            )
            sharded = False
        # ``sharded`` here means only "the values on the carrier are windows, so
        # lengths must come from ``meta``". It is deliberately *not* what decides
        # the output shape: the switch is run-wide, so this module owes the next
        # one a window whatever path its own input took — see ``forward_post``.
        # ``sharded`` means the producers handed this rank their windows, so the
        # packed embeds are already this rank's row and the lengths/positions come
        # from ``meta``. Everything else is the same packer.
        packed = pack_qwen3omni_conversations_for_forward(
            conversation_list,
            self.device,
            self._spatial_merge_size,
            self.config.position_id_per_seconds,
            sharded=sharded,
        )
        inputs_embeds = packed["inputs_embeds"]

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = packed["inputs_embeds_shape"]

        # The packed row's length, independent of whether the values arrived
        # already sliced: the mask and the DeepStack schedule below are always
        # expressed over the full row and narrowed once by the SP slice.
        full_len = sum(item_token_len(item) for sample in conversation_list for item in sample if not item.is_dummy)

        # Normalize the visual side to always-real tensors: the SP input-slice branch
        # below slices ``visual_pos_masks`` and indexes the per-layer DeepStack embeds,
        # so they must be real tensors (not ``None`` / lists) on every rank.
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._normalize_visual_inputs(
            inputs_embeds,
            packed["visual_pos_masks"],
            packed["deepstack_visual_embeds"],
            full_len=full_len,
        )

        if ps.sp_size > 1:
            # SP: hand this rank its Ulysses shard of the packed sample, including
            # the TMRoPE positions and DeepStack rows. Pad the sample to a multiple
            # of ``sp_size``, extend the full-sample varlen ``cu_seqlens`` with the
            # pad tail (the attention all-to-all reconstructs the full sequence
            # before the kernel), then hand this rank only its ``1/sp_size`` chunk
            # of embeds (dim 1), 3-row positions (dim 2) and ``visual_pos_masks``
            # (dim 1). The DeepStack embeds are indexed by visual position, so slice
            # out exactly the rows whose visual tokens fall in this rank's chunk
            # (``n_before`` skipped + ``k_local`` kept).
            #
            # ``sharded`` changes what "this rank's chunk" means for the embeds
            # only: the producers already wrote their windows onto the carrier, so
            # ``inputs_embeds`` *is* the shard and must not be sliced a second time.
            # Padding restores the window width the attention gathers to. The
            # metadata is built over the full row either way, so it is padded and
            # sliced in both modes.
            group, sp_size, sp_rank = ps.sp_group, ps.sp_size, ps.sp_rank

            full_len = sum(
                item_token_len(item) for sample in conversation_list for item in sample if not item.is_dummy
            )
            self._sp_own_len = full_len
            # Same length, but tracked apart from ``_sp_own_len``: the router loss
            # gathers its own tensor set and is consumed before the hidden-state
            # gather, so sharing one field would have them clobber each other.
            self._sp_router_own_len = full_len
            cu = packed["cu_seq_lens"]
            max_length = int((cu[1:] - cu[:-1]).max().item()) if cu.numel() > 1 else 0

            if sharded:
                embeds = pad_sp_shard(inputs_embeds, full_len, sp_size, dim=1)
            else:
                embeds = sp_pad(inputs_embeds, dim=1, pad_value=0)
            pids = sp_pad(packed["position_ids"], dim=2, pad_value=0)
            mask = sp_pad(visual_pos_masks, dim=1, pad_value=0)
            padded_len = pids.size(2)
            pad_len = padded_len - self._sp_own_len
            if pad_len > 0:
                tail = torch.tensor([padded_len], dtype=cu.dtype, device=cu.device)
                cu = torch.cat([cu, tail], dim=0)
                max_length = max(max_length, pad_len)

            unit = padded_len // sp_size
            n_before = int(mask[:, : unit * sp_rank].sum().item())
            if not sharded:
                embeds = slice_input_tensor(embeds, dim=1, padding=False, group=group)
            pids = slice_input_tensor(pids, dim=2, padding=False, group=group)
            mask = slice_input_tensor(mask, dim=1, padding=False, group=group)
            if deepstack_visual_embeds is not None:
                k_local = int(mask.sum().item())
                deepstack_visual_embeds = deepstack_visual_embeds[:, n_before : n_before + k_local, :]

            return dict(
                inputs_embeds=embeds,
                position_ids=pids,
                visual_pos_masks=mask,
                deepstack_visual_embeds=deepstack_visual_embeds,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max_length,
                max_length_k=max_length,
                output_hidden_states=self.config.collect_thinker_states,
            )

        return dict(
            inputs_embeds=inputs_embeds,
            # 3-axis TMRoPE only. Every training return below also carries
            # ``cu_seq_lens_q``, so the causal mask is built from the packed
            # boundaries and never from these positions — the extra contiguous
            # text row the non-packed path needs (see ``pack_*_for_forward``)
            # would just be a fourth row nobody reads.
            position_ids=packed["position_ids"],
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cu_seq_lens_q=packed["cu_seq_lens"],
            cu_seq_lens_k=packed["cu_seq_lens"],
            max_length_q=packed["max_length"],
            max_length_k=packed["max_length"],
            output_hidden_states=self.config.collect_thinker_states,
            **kwargs,
        )

    def _normalize_visual_inputs(
        self,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        *,
        full_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Turn the packed visual side into always-real tensors (see ``forward_pre``).

        Real visuals: stack the per-layer DeepStack embeds into ``(num_layers, N, D)``.
        No visuals (text and/or audio only): return an all-False mask + an EMPTY
        ``(num_layers, 0, D)`` deepstack, and fold the dummy DeepStack anchor into
        ``inputs_embeds`` so the vision mergers that produced it still get (zero)
        gradient. This replaces the modeling's ``visual_pos_masks=None`` anchor path,
        which the SP input-slice cannot use (it needs a real mask on every rank).

        The mask is built over ``full_len``, the length of the **packed row**, not over
        ``inputs_embeds.size(1)``: under the shard handoff the embeds already hold this
        rank's window, and the caller pads and narrows the mask by that same full-row
        schedule. Building it at the window width made the second narrowing halve it
        again, so the mask described half the rows it was paired with.
        """
        if visual_pos_masks is not None:
            return inputs_embeds, visual_pos_masks, torch.stack(deepstack_visual_embeds, dim=0)

        mask = torch.zeros(1, full_len, dtype=torch.bool, device=self.device)
        if not deepstack_visual_embeds:
            # No DeepStack layers to anchor (e.g. text/audio-only run without vision).
            return inputs_embeds, mask, None
        num_layers = len(deepstack_visual_embeds)
        hidden_dim = deepstack_visual_embeds[0].size(-1)
        # ``sum`` for the same reason as ``_fold_fsdp_dummy_anchors``: a zero-row
        # slice is NaN under ``mean``, and ``NaN * 0`` still poisons the row.
        anchor = sum(layer.float().sum() for layer in deepstack_visual_embeds) * 0.0
        inputs_embeds = inputs_embeds + anchor.to(inputs_embeds.dtype)
        empty_deepstack = inputs_embeds.new_zeros(num_layers, 0, hidden_dim)
        return inputs_embeds, mask, empty_deepstack

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        hidden_states = outputs.pop("hidden_states")
        router_logits = outputs.pop("router_logits", None)
        # Dropped rather than used: the balancing loss is computed over the full
        # gathered sequence, where every row is real, so no mask applies. Keeping
        # the key would only invite passing a local-width mask to a full-width
        # tensor — the shape mismatch this replaced.
        outputs.pop("router_attention_mask", None)
        traces = self._thinker_traces(outputs)

        ps = get_parallel_state()
        # The switch is run-wide, so the output is a window whenever it is on —
        # regardless of whether *this* module's inputs needed regathering. The
        # consumer (`decode_pre`, or the next module's packer) follows the same
        # switch, and a full-length output here would leave, for instance, full
        # hidden states scored against a half-length label slice.
        if ps.sp_size > 1 and ps.skip_sp_gather_between_modules:
            # Shard handoff: this rank keeps its own window and the consumer (the
            # next module's packer, or ``decode``'s label slice) concatenates from
            # the carrier. The traces follow the same split so they stay aligned
            # with the hidden states the talker pairs them against.
            conversation = self._conversation_carrier
            self._conversation_carrier = None
            self._pack_inputs_embeds_shape = None
            self._sp_own_len = None
            window_lengths = [item_token_len(item) for sample in conversation for item in sample if not item.is_dummy]
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
                hidden_states = hidden_states.squeeze(0)
            scatter_hidden_states(
                conversation,
                split_sp_sharded_tensor(hidden_states, window_lengths, ps.sp_rank, ps.sp_size),
            )
            if traces:
                embed_trace, hidden_trace = (_unbatch(t).squeeze(0) for t in traces[:2])
                _scatter_thinker_trace_slices(
                    conversation,
                    split_sp_sharded_tensor(embed_trace, window_lengths, ps.sp_rank, ps.sp_size),
                    split_sp_sharded_tensor(hidden_trace, window_lengths, ps.sp_rank, ps.sp_size),
                )
            result: Dict[str, Any] = {"conversation_list": conversation}
            # Cleared *after* the router loss has consumed the length, the same
            # ordering the gathering path uses below.
            result = self._with_router_loss(result, router_logits, conversation)
            self._sp_router_own_len = None
            return result

        if ps.sp_size > 1:
            # SP output-gather: all-gather Ulysses shards back to the full sequence
            # (autograd-aware; backward sums grads across the SP group), then strip
            # the SP pad tail so the packed length matches its carrier below.
            hidden_states = gather_outputs(hidden_states, gather_dim=1, group=get_parallel_state().sp_group)
            hidden_states = hidden_states.narrow(1, 0, self._sp_own_len)
            traces = tuple(
                gather_outputs(t, gather_dim=1, group=get_parallel_state().sp_group).narrow(1, 0, self._sp_own_len)
                for t in traces
            )

        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        scatter_hidden_states(conversation, unflatten(hidden_states, pack_shape))
        if traces:
            _scatter_thinker_traces(conversation, *(unflatten(_unbatch(t), pack_shape) for t in traces))

        result: Dict[str, Any] = {"conversation_list": conversation}
        # Cleared only after every consumer has run: ``_full_sequence_router_logits``
        # needs it, and it is read *after* the block above cleared the others.
        # Clearing it up there left `None` reaching `.narrow(..., None)`.
        result = self._with_router_loss(result, router_logits, conversation)
        self._sp_router_own_len = None
        return result

    def _with_router_loss(
        self,
        result: Dict[str, Any],
        router_logits: Any,
        conversation: list[list[ConversationItem]],
    ) -> Dict[str, Any]:
        """Attach the router-balancing term, if this run collects router logits.

        Shared by the gathering and the shard-handoff exits, so the two cannot
        drift on the coefficient or on which tensor the objective sees.
        """
        if router_logits is not None:
            # ``OmniModel.forward`` sums every node's ``_loss``, so a mid-graph
            # module contributes by returning one. Without this the objective was
            # configured and the logits were collected, then dropped: the run
            # silently trained on the decode head's CE alone.
            cfg = self.config.text_config
            aux_loss = self.router_balancing_loss(self._full_sequence_router_logits(router_logits))
            if isinstance(aux_loss, torch.Tensor):
                # ``router_aux_loss_coef`` is applied here rather than inside the
                # loss so a zero coefficient still reports the raw term.
                coefficient = getattr(cfg, "router_aux_loss_coef", 0.0)
                device = conversation[0][0].value.device
                result["_loss"] = aux_loss.to(device=device) * coefficient
        return result

    def _full_sequence_router_logits(self, router_logits: Any) -> Any:
        """Router logits over the whole packed sequence, not this rank's window.

        The balancing term is a *product* of two batch-level means (per-expert
        token share times mean router probability), so it is not linear in the
        rows: averaging per-rank values does not reproduce the value over the
        combined rows. Under Ulysses each rank therefore has to see every row
        before the product forms, so the per-layer logits are gathered back and
        the SP pad tail stripped — the same concatenation the hidden-state
        gather below produces.

        With ``sp_size == 1`` this is the identity, and there is no pad, so the
        single-device result is unchanged.
        """
        if router_logits is None or get_parallel_state().sp_size == 1:
            return router_logits
        group = get_parallel_state().sp_group
        own_len = self._sp_router_own_len
        gathered = tuple(
            gather_outputs(layer, gather_dim=0, group=group).narrow(0, 0, own_len) for layer in router_logits
        )
        return gathered

    def _thinker_traces(self, outputs: Dict[str, Any]) -> tuple:
        """The talker's two inputs, pulled out of the layer stack this forward kept.

        Index 0 of ``all_hidden_states`` is the input embedding and index i>0 is
        decoder layer i's output, so both traces come from the one forward the
        thinker already ran — no second pass and no separate embedding lookup.

        Detached on purpose. The talker's loss must not reach back through the
        thinker: this round trains the talker against a frozen understanding
        stack, and letting the gradient in would keep every thinker activation
        alive to produce parameter gradients that are all masked off anyway.
        Joint training would be a deliberate change here, not an accident.
        """
        if not self.config.collect_thinker_states:
            return ()
        all_hidden_states = outputs.get("all_hidden_states")
        if all_hidden_states is None:
            raise RuntimeError(
                "qwen3omni_llm: collect_thinker_states is on but the backbone returned no all_hidden_states. "
                "forward_pre must pass output_hidden_states=True."
            )
        accept = self.config.accept_hidden_layer
        if accept >= len(all_hidden_states):
            # Bare indexing would raise IndexError from inside a post hook,
            # which says nothing about the checkpoint split that set the layer.
            raise ValueError(
                f"qwen3omni_llm: accept_hidden_layer={accept} but this backbone has "
                f"{len(all_hidden_states) - 1} layers. The talker reads an interior layer of the thinker; "
                f"the checkpoint split copies the index from talker_config.accept_hidden_layer."
            )
        return (all_hidden_states[0].detach(), all_hidden_states[accept].detach())


def _scatter_thinker_trace_slices(
    conversation: list[list[ConversationItem]],
    embed_slices: list[torch.Tensor],
    hidden_slices: list[torch.Tensor],
) -> None:
    """Leave each item's window slice of both traces on the item, in carrier order.

    The shard-handoff counterpart of :func:`_scatter_thinker_traces`: the walk and
    the order are identical (``is_thinker_input`` items only, paired by position),
    but each item receives the intersection of its span with this rank's window.
    """
    embed_iterator = iter(embed_slices)
    hidden_iterator = iter(hidden_slices)
    for sample in conversation:
        for part in sample:
            if not is_thinker_input(part):
                continue
            part.meta[THINKER_EMBED] = next(embed_iterator)
            part.meta[THINKER_HIDDEN] = next(hidden_iterator)


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``_spatial_merge_size`` is inherited from the native :class:`Qwen3OmniLlm`.

    ``generate()`` and inference-state reset already live on that native class
    (via its own :class:`~.modeling.InferenceMixin`), so no ``InferenceMixin``
    is needed here.
    """

    def get_parallel_plan(self) -> ParallelPlan:
        # The thinker backbone is MoE (30B-A3B), so it needs the same Expert
        # Parallel plan as qwen3_moe — the dense qwen3vl sibling this module is
        # otherwise modelled on has none. fqn is module-local: ``self.language_model``
        # is the bare thinker text model with layers directly under it, so there is
        # no ``model.`` prefix (unlike the transformers-level parallel plan).
        ep_plan = {
            "language_model.layers.*.mlp.experts.gate_up_proj": Shard(0),
            "language_model.layers.*.mlp.experts.down_proj": Shard(0),
        }
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


def _unbatch(trace: torch.Tensor) -> torch.Tensor:
    return trace.squeeze(0) if trace.dim() == 3 and trace.size(0) == 1 else trace


def _scatter_thinker_traces(
    conversation: list[list[ConversationItem]],
    embeds: list[torch.Tensor],
    hiddens: list[torch.Tensor],
) -> None:
    """Leave each item's slice of the two traces on the item itself.

    ``item.value`` is already spoken for — ``scatter_hidden_states`` puts
    the final hidden state there for the text head — so these go to ``meta``.
    Same walk, same order, so an item's traces line up with the value beside
    them, and the talker can read a prompt straight off the carrier instead of
    the graph having to route tensors between two modules.
    """
    embeds_iter, hiddens_iter = iter(embeds), iter(hiddens)
    for sample in conversation:
        for part in sample:
            # Same skip as the packer: these are paired by position.
            if not is_thinker_input(part):
                continue
            part.meta[THINKER_EMBED] = next(embeds_iter)
            part.meta[THINKER_HIDDEN] = next(hiddens_iter)
    if next(embeds_iter, None) is not None or next(hiddens_iter, None) is not None:
        raise RuntimeError("_scatter_thinker_traces: trace segment count exceeds packed conversation items.")


def _fold_fsdp_dummy_anchors(
    inputs_embeds: torch.Tensor,
    conversations: list[list[ConversationItem]],
) -> torch.Tensor:
    for sample in conversations:
        for part in sample:
            if not part.is_dummy:
                continue
            if isinstance(part.value, torch.Tensor):
                # ``sum``, not ``mean``: under the shard handoff a dummy can hold a
                # zero-row window (it lies outside this rank's slice), and ``mean``
                # of an empty tensor is NaN — which ``NaN * 0`` propagates into the
                # real embeddings. ``sum`` of an empty tensor is 0.0, and the
                # multiplication still keeps the anchor differentiable.
                inputs_embeds = (
                    inputs_embeds + part.value.sum().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) * 0.0
                )
    return inputs_embeds


class Qwen3OmniLlmAccelerated(VeOmniMixin, Qwen3OmniLlm):
    pass


__all__ = ["Qwen3OmniLlmAccelerated", "TrainingMixin", "VeOmniMixin"]
