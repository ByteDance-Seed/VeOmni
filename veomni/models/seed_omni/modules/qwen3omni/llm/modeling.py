"""Qwen3-Omni thinker AR backbone (no wte / lm_head, with DeepStack injection)."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.loss_utils import load_balancing_loss
from veomni.models.transformers.qwen3_omni_moe.checkpoint_tensor_converter import (
    Qwen3OmniMoeCheckpointTensorConverter,
    convert_qwen3_omni_moe_fqn_to_index_mapping,
)
from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
)
from veomni.ops import VeomniOp
from veomni.utils.tensor_utils import naflatten

from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from ..carrier import is_thinker_input
from .configuration import Qwen3OmniLlmConfig


# Where the training path leaves the two traces the talker consumes, one entry
# per non-dummy carrier item. Inference cannot use the carrier for this (see
# ``flush_thinker_states``) but training can: it is a single forward over the
# whole packed sequence, so the same per-item split that returns hidden states
# to ``item.value`` returns these to ``item.meta``.
THINKER_EMBED = "thinker_embed"
THINKER_HIDDEN = "thinker_hidden"


def _cumsum(values: list[int]) -> list[int]:
    out: list[int] = []
    total = 0
    for v in values:
        total += v
        out.append(total)
    return out


def qwen3omni_vision_position_ids(
    start: int,
    grid_thw: torch.Tensor,
    merge: int,
    *,
    position_id_per_seconds: int = 13,
    second_per_grid: float = 1.0,
) -> torch.Tensor:
    """3-row (t/h/w) TMRoPE positions for one image or video clip.

    Mirrors ``Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision``.
    The temporal row is scaled to *seconds* (``second_per_grid *
    position_id_per_seconds``) rather than frame index, which is what lets an
    audio track share a timeline with video. For a still image the caller passes
    ``second_per_grid=1.0``, matching upstream's ``t_index = arange(grid_t) * 1 *
    position_id_per_seconds``.

    Returns a **float** tensor: ``second_per_grid`` is ``temporal_patch_size /
    fps`` and is rarely an integral number of position units, so rounding the
    temporal row here would shift every position after the clip.
    """
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    gh, gw = h // merge, w // merge
    t_index = torch.arange(t, dtype=torch.float) * float(second_per_grid) * position_id_per_seconds
    temporal = t_index.repeat_interleave(gh * gw)
    height = torch.arange(gh, dtype=torch.float).repeat_interleave(gw).repeat(t)
    width = torch.arange(gw, dtype=torch.float).repeat(gh * t)
    return torch.stack([temporal, height, width], dim=0) + start


def _second_per_grid(item: ConversationItem) -> float:
    """Seconds covered by one temporal grid step of a vision item.

    A still image is one instant, so upstream uses 1.0 for it. A video clip's
    value depends on the sampling rate (``temporal_patch_size / fps``) and only
    the vision preprocessor knows it, so a video that arrives without it is a
    bug rather than something to default: guessing 1.0 would place the clip on a
    wrong timeline and, worse, shift every position after it.
    """
    if item.type == "image":
        return float(item.meta.get("second_per_grid", 1.0))
    second_per_grid = item.meta.get("second_per_grid")
    if second_per_grid is None:
        raise ValueError(
            "qwen3omni_llm: video item is missing meta['second_per_grid'] (temporal_patch_size / fps). "
            "The vision preprocessor must set it — TMRoPE's temporal row is in seconds, not frame index."
        )
    return float(second_per_grid)


def collect_qwen3omni_dummy_deepstack(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    for sample in conversations:
        for item in sample:
            if item.is_dummy and item.type in ("image", "video") and "deepstack" in item.meta:
                return [d.to(device) for d in item.meta["deepstack"]]
    return None


def pack_qwen3omni_conversations_for_forward(
    conversations: list[list[ConversationItem]],
    device: torch.device,
    spatial_merge_size: int,
    position_id_per_seconds: int = 13,
    *,
    sharded: bool = False,
) -> Dict[str, Any]:
    """Pack Qwen3-Omni conversation embeds for backbone forward (training + inference).

    Position ids follow upstream ``get_rope_index``: every segment starts at
    ``previous_segment.max() + 1``. Text and standalone-audio segments get plain
    ``arange`` on all three rows (TMRoPE degenerates to 1-D there); only vision
    segments spread across t/h/w. They are float, like upstream's — see
    :func:`qwen3omni_vision_position_ids`.

    ``sharded`` reads the batch after a ``skip_sp_gather_between_modules``
    handoff: each ``item.value`` then holds only this rank's window, so lengths
    and positions must come from ``meta`` (:func:`item_token_len`) rather than
    from the value's row count. The returned ``position_ids`` /
    ``visual_pos_masks`` / ``deepstack`` are still over the **full** row — the
    caller pads and slices them — and ``inputs_embeds_shape`` holds the window
    row counts the hidden-state scatter needs.
    """
    from ...base.llm_packing import item_token_len

    inputs_embeds_list: list[torch.Tensor] = []
    position_ids_list: list[torch.Tensor] = []
    visual_pos_masks_list: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    deepstack_chunks: list[list[torch.Tensor]] = []
    for sample in conversations:
        sample_len = 0
        current_pos = 0.0
        for item in sample:
            # Assistant audio is the talker's label; the thinker never reads it.
            if not is_thinker_input(item):
                continue
            embeds = item.value.to(device)
            length = item_token_len(item) if sharded else embeds.size(0)
            if length == 0:
                # An empty segment carries no position and would break the
                # ``seg_pos.max()`` cursor below. It is kept in the packed
                # sequence as a zero-row entry rather than dropped, because the
                # hidden-state scatter pairs rows to items by position: dropping
                # it here while ``scatter_llm_hidden_states`` visits every
                # backbone-input item would hand this row's slice to the empty
                # item and raise ``StopIteration`` on the next — after the
                # carrier had already been partly overwritten.
                #
                # ``embeds`` is already ``(0, H)``: every backbone-input producer
                # emits a 2-D ``(rows, hidden)`` row, and an empty one keeps its
                # hidden width (a text row is embedded to `(0, hidden)`, a clip to
                # `(0, num_mel_bins)` patches). So there is no width to recover
                # here — only the row to keep, so packing and the hidden-state
                # scatter visit the same segments.
                inputs_embeds_list.append(embeds)
                position_ids_list.append(torch.zeros(3, 0, dtype=torch.float, device=device))
                visual_pos_masks_list.append(torch.zeros(0, dtype=torch.bool, device=device))
                if item.type in ("image", "video") and "deepstack" in item.meta:
                    deepstack_chunks.append([d.to(device)[:0] for d in item.meta["deepstack"]])
                continue
            inputs_embeds_list.append(embeds)
            is_visual = item.type in ("image", "video")
            if is_visual:
                seg_pos = qwen3omni_vision_position_ids(
                    current_pos,
                    item.meta["grid_thw"],
                    spatial_merge_size,
                    position_id_per_seconds=position_id_per_seconds,
                    second_per_grid=_second_per_grid(item),
                ).to(device)
                visual_pos_masks_list.append(torch.ones(length, dtype=torch.bool, device=device))
                deepstack_chunks.append([d.to(device) for d in item.meta["deepstack"]])
            else:
                seg_pos = (
                    torch.arange(length, dtype=torch.float, device=device).view(1, -1).expand(3, -1) + current_pos
                )
                visual_pos_masks_list.append(torch.zeros(length, dtype=torch.bool, device=device))
            current_pos = float(seg_pos.max()) + 1.0
            position_ids_list.append(seg_pos)
            sample_len += length
        sample_lengths.append(sample_len)

    inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
    if inputs_embeds.dim() == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if sharded:
        # ``naflatten`` reported the *window* row counts, which is what the
        # hidden-state scatter needs. The sequence metadata below is over the full
        # row, so the batch shape is restated in full lengths.
        inputs_embeds_shape = torch.tensor(
            [[int(item_token_len(item))] for sample in conversations for item in sample if is_thinker_input(item)],
            dtype=torch.long,
            device="cpu",
        )
    position_ids = torch.cat(position_ids_list, dim=1).unsqueeze(1)
    visual_pos_masks = torch.cat(visual_pos_masks_list, dim=0).unsqueeze(0)

    if deepstack_chunks:
        num_layers = len(deepstack_chunks[0])
        deepstack_visual_embeds = [
            torch.cat([chunk[layer] for chunk in deepstack_chunks], dim=0) for layer in range(num_layers)
        ]
    else:
        deepstack_visual_embeds = collect_qwen3omni_dummy_deepstack(conversations, device)
        visual_pos_masks = None

    cu_seq_lens = torch.tensor([0, *_cumsum(sample_lengths)], dtype=torch.int32, device=device)
    max_length = max(sample_lengths) if sample_lengths else 0

    return {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
        "cu_seq_lens": cu_seq_lens,
        "max_length": max_length,
        "inputs_embeds_shape": inputs_embeds_shape,
    }


class InferenceMixin:
    """FSM ``generate`` (with TMRoPE + DeepStack) — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3OmniLlm`'s bases: ``PretrainedOmniModule`` ships no-op
    ``reset_local_inference_state`` / ``reset_global_inference_state`` defaults,
    and MRO resolves left-to-right — put second, those no-ops would shadow the
    real implementations below.

    Each step appends the tail hidden state as an ``output`` item. The talker
    consumes those same hidden states later in the graph, so they must stay on
    the carrier rather than be reduced to token ids here.
    """

    def reset_local_inference_state(self) -> None:
        return

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None
        self._next_position = 0.0
        self._thinker_embed_cache = []
        self._accept_hidden_cache = []

    def flush_thinker_states(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Hand the talker the two thinker traces it consumes, then clear them.

        Reached as its own generation-graph node (the FSM splats ``ctx`` into
        node methods, hence ``**kwargs``), so the returned keys land in ``ctx``
        and the talker node picks them up by parameter name. A node rather than
        a sibling-module call because only the graph knows when the thinking
        phase has ended — the backbone itself never sees the ``text_done``
        signal that the text encoder raises.

        Upstream builds these by concatenating ``hidden_states[0]`` and
        ``hidden_states[accept_hidden_layer]`` across every thinker generation
        step (``Qwen3OmniMoeForConditionalGeneration.generate``). The FSM sees
        the same steps one at a time, so the backbone accumulates them here
        instead — the carrier cannot, because ``maybe_merge_outputs`` merges only
        item values and would drop per-step metadata.

        Returns ``(1, T, D)`` tensors covering the prompt plus every generated
        token except the last (the final token's embedding is never fed back, so
        it has no step of its own — matching upstream, where ``thinker_embed`` is
        one shorter than ``sequences``).
        """
        states = {
            "thinker_embed": torch.cat(self._thinker_embed_cache, dim=1),
            "thinker_hidden": torch.cat(self._accept_hidden_cache, dim=1),
        }
        self._thinker_embed_cache = []
        self._accept_hidden_cache = []
        return states

    def _cache_thinker_states(self, inputs_embeds: torch.Tensor, all_hidden_states: Any) -> None:
        accept = all_hidden_states[self.config.accept_hidden_layer]
        self._thinker_embed_cache.append(inputs_embeds.detach())
        self._accept_hidden_cache.append(accept.detach())

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        # ``forward_pre`` slices the packed sample to this rank's Ulysses shard;
        # ``generate`` has no such hook, so under sequence parallelism the
        # Ulysses exchange would rebuild the full sequence from a prompt every
        # rank already holds (4 rows in, 8 out) while ``cu_seq_lens`` still
        # described the original — a wrong answer rather than a crash. The shared
        # ``SimpleArGenerationMixin`` refuses this the same way.
        if get_parallel_state().sp_size > 1:
            raise ValueError(
                "qwen3omni_llm.generate feeds the full prompt to the backbone, which conflicts with Ulysses "
                "sequence parallelism (attention expects this rank's shard). Run inference with sp_size=1."
            )
        if self._past_key_values is None:
            packed = pack_qwen3omni_conversations_for_forward(
                [conversation_list],
                self.device,
                self._spatial_merge_size,
                self.config.position_id_per_seconds,
            )
            position_ids = packed["position_ids"]
            collect = self.config.collect_thinker_states
            outputs = self.forward(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=None,
                position_ids=position_ids,
                visual_pos_masks=packed["visual_pos_masks"],
                deepstack_visual_embeds=packed["deepstack_visual_embeds"],
                past_key_values=self._past_key_values,
                cu_seq_lens_q=packed["cu_seq_lens"],
                cu_seq_lens_k=packed["cu_seq_lens"],
                max_length_q=packed["max_length"],
                max_length_k=packed["max_length"],
                use_cache=True,
                output_hidden_states=collect,
            )
            if collect:
                self._cache_thinker_states(packed["inputs_embeds"], outputs["all_hidden_states"])
            self._past_key_values = outputs["past_key_values"]
            self._next_position = float(position_ids.max()) + 1.0
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=self._tail_hidden_from_forward(outputs["hidden_states"]),
                    role="assistant",
                )
            )
            return {"conversation_list": conversation_list}

        tail_part = conversation_list[-1]
        assert tail_part.type == "output"
        inputs_embeds = tail_part.value[-1:].to(self.device).unsqueeze(0)
        position_ids = torch.full((3, 1, 1), self._next_position, dtype=torch.float, device=self.device)
        collect = self.config.collect_thinker_states
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self._past_key_values,
            use_cache=True,
            output_hidden_states=collect,
        )
        if collect:
            self._cache_thinker_states(inputs_embeds, outputs["all_hidden_states"])
        self._past_key_values = outputs["past_key_values"]
        self._next_position += 1.0
        conversation_list.append(
            ConversationItem(
                type="output",
                value=self._tail_hidden_from_forward(outputs["hidden_states"]),
                role="assistant",
            )
        )
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()


class Qwen3OmniLlm(InferenceMixin, PretrainedOmniModule):
    """Qwen3-Omni thinker backbone (no wte, no lm_head).

    Token / image / audio embeds are produced by the sibling
    ``qwen3omni_text_encoder`` / ``qwen3omni_vision`` / ``qwen3omni_audio``
    modules and live on ``conversation_list`` items. :meth:`pre_forward`
    concatenates them per sample, rebuilds TMRoPE position ids, and threads the
    per-layer DeepStack features into the text model.
    """

    config_class = Qwen3OmniLlmConfig
    base_model_prefix = "qwen3omni_llm"
    main_input_name = "inputs_embeds"
    # Hidden states and both thinker traces are plain per-item concatenations of
    # the carrier, so ``forward_post`` can hand this rank's window on item by item
    # and the consumer concatenates it back — the same contract the qwen3 backbone
    # implements.
    supports_sp_sharded_handoff = True
    _no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    supports_gradient_checkpointing = True
    # This wrapper owns no attention — ``self.language_model`` does, and the
    # patched thinker accepts both eager and SDPA. Without this the base
    # ``_supports_sdpa = False`` rejects ``sdpa`` / ``veomni_sdpa`` at
    # construction, and Ulysses refuses a plain ``eager`` (it performs no
    # exchange of its own) — so sequence-parallel training had no backend it
    # could be configured with.
    _supports_sdpa = True

    def __init__(self, config: Qwen3OmniLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3OmniMoeThinkerTextModel._from_config(self.config.text_config)
        self.language_model.set_input_embeddings(nn.Identity())
        # Instance-local handle for the router-balancing objective (see
        # ``router_balancing_loss``). Built from ``ops_config``, so changing the
        # config after construction does not replace it — the same rule every
        # other op handle follows.
        self.veomni_load_balancing_loss = VeomniOp(
            "load_balancing_loss",
            "standard",
            self.config.ops_config.load_balancing_loss_implementation,
        )
        self._past_key_values: Any = None
        self._next_position: float = 0.0
        self._thinker_embed_cache: List[torch.Tensor] = []
        self._accept_hidden_cache: List[torch.Tensor] = []
        self.post_init()

    @property
    def _spatial_merge_size(self) -> int:
        return self.config.spatial_merge_size

    def _router_aux_loss_enabled(self) -> bool:
        """Whether the upstream router-balancing objective is configured.

        Mirrors ``Qwen3OmniMoeThinkerForConditionalGeneration.forward``:
        ``output_router_logits`` defaults to the config field. Off by default, so
        the common path collects nothing.
        """
        return bool(getattr(self.config.text_config, "output_router_logits", False))

    def router_balancing_loss(self, gate_logits: Any, attention_mask: Any = None) -> Any:
        """The router-balancing term, through this module's bound ``VeomniOp``.

        Bind it here rather than calling the raw function: the row is what
        ``ops_config.load_balancing_loss_implementation`` selects, so a fused
        backend configured for the run actually reaches this objective — the same
        handle the standalone thinker builds in its own ``__init__``.
        """
        return load_balancing_loss(
            gate_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
            op=self.veomni_load_balancing_loss,
        )

    def freeze_model(self) -> None:
        """Freeze the whole backbone when ``config.freeze`` (talker training)."""
        if self.config.freeze:
            self.language_model.requires_grad_(False)

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # The backbone's causal mask must come from an explicit one whenever it
        # cannot be derived from ``cu_seq_lens``.
        #
        # ``mask_utils._should_use_packed_mask`` declines a single-segment
        # ``cu_seq_lens`` ("one segment per row is not packing"), which would
        # leave the mask to Transformers' ``find_packed_sequence_indices``. That
        # helper treats a position *gap* as a sequence boundary, and TMRoPE
        # guarantees gaps — its temporal row repeats across a vision span and
        # jumps at the segment after it. A single-sample image+text batch then
        # reads as several packed sequences, and the resulting mask indexes past
        # the row count and raises.
        if attention_mask is None and position_ids is not None and "cu_seq_lens_q" in kwargs:
            cu = kwargs["cu_seq_lens_q"]
            if cu is not None and int(cu.numel()) == 2:
                # One sequence: plain causal, over the *full* sequence. Taken from
                # ``cu_seqlens`` and not from ``inputs_embeds``, because under
                # Ulysses that tensor already holds this rank's shard — and the
                # backends that consume a mask (SDPA in particular) describe the
                # full sequence that ``cu_seqlens`` names.
                rows = cu[-1].item()
                attention_mask = torch.ones(1, rows, dtype=torch.long, device=position_ids.device)

        # Router logits are only collected when the router-balancing objective is
        # configured; passing the flag unconditionally would make every layer
        # build the list for nothing. Same gate as the qwen3_moe sibling.
        output_router_logits = self._router_aux_loss_enabled()
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        result = {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
            # ``None`` unless the objective is on, and popped by the graph hook
            # rather than carried through the DAG.
            "router_logits": outputs.router_logits if output_router_logits else None,
            # Deliberately *not* ``attention_mask``. That one describes the rows
            # this call's backend consumed, which under Ulysses is this rank's
            # shard — while the router logits the hook passes to the balancing
            # loss are laid out the same way. The hook builds its own mask from
            # the logits' own row count, so the two can never disagree.
            "router_attention_mask": None,
        }
        if kwargs.get("output_hidden_states"):
            # Named apart from ``hidden_states`` (the last layer, what the lm_head
            # reads) so the graph cannot confuse the two. Index 0 is the input
            # embedding; index i>0 is decoder layer i's output.
            result["all_hidden_states"] = outputs.hidden_states
        return result


def _create_qwen3omni_llm_checkpoint_tensor_converter(model: "Qwen3OmniLlm"):
    """Per-expert -> fused experts converter for the omni backbone.

    An upstream checkpoint stores experts per expert
    (``thinker.model.layers.N.mlp.experts.{j}.{gate,up,down}_proj.weight``); the
    v5 modeling holds one stacked ``experts.gate_up_proj`` / ``experts.down_proj``.
    HF ``from_pretrained`` fuses natively, but the VeOmni meta-init / FSDP loader
    needs the converter explicitly — without it those keys match no parameter and
    the experts keep their fused-shape *initialisation*, so a direct upstream-root
    load silently trains on random experts while the offline split works fine.
    """
    return Qwen3OmniMoeCheckpointTensorConverter(num_experts=model.config.text_config.num_experts)


# Wire the conversion onto this wrapper, the way the qwen3_moe sibling does: the
# loader resolves the class through ``OMNI_MODEL_REGISTRY``, so the hook has to
# be registered here rather than on the transformers-side model.
Qwen3OmniLlm._create_checkpoint_tensor_converter = staticmethod(_create_qwen3omni_llm_checkpoint_tensor_converter)
Qwen3OmniLlm._convert_fqn_to_index_mapping = staticmethod(convert_qwen3_omni_moe_fqn_to_index_mapping)


__all__ = [
    "THINKER_EMBED",
    "THINKER_HIDDEN",
    "Qwen3OmniLlm",
    "collect_qwen3omni_dummy_deepstack",
    "pack_qwen3omni_conversations_for_forward",
    "qwen3omni_vision_position_ids",
]
