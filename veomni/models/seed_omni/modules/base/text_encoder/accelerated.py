"""VeOmni-accelerated TextEncoder — training graph hooks.

FSM ``generate`` + its sampling helpers now live natively on
:class:`~.modeling.TextEncoder` (see its docstring) so pure HF usage of a
family's native class works without this accelerated wrapper. Only the
SP-aware training pre/forward/post hooks stay here.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_pad,
)
from veomni.ops.kernels.cross_entropy import ForCausalLMLoss
from veomni.ops.kernels.cross_entropy.eager import eager_cross_entropy
from veomni.utils.tensor_utils import naflatten, unflatten

from ....mixins.base_mixin import BaseMixin
from ....mixins.emb_parallel_mixin import EmbParallelMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy
from .chat_template import TextEncoderChatTemplate
from .configuration import TextEncoderConfig
from .modeling import TextEncoder, scatter_text_encoder_embeds


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks shared by every text encoder."""

    config: TextEncoderConfig
    device: torch.device
    _chat_template: TextEncoderChatTemplate
    _encode_batch_shape: torch.LongTensor | None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        input_ids = self._prepare_encode_inputs(self._conversation_carrier)

        # Metering: this rank's OWN packed token count, stashed BEFORE the SP slice
        # below. The meter sums over the DP group only, so each rank reports just its
        # own tokens (not the SP peers that hold the same replicated sample) —
        # identical to the non-SP run for both SP-disabled and uniform SP.
        self.metric_meter_set_seqlens("encode", [int(input_ids.numel())])

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its ``1/sp_size`` slice of the packed
            # token sequence (wte is a per-token lookup, so slicing is exact —
            # mirrors VeOmni's whole-model SP where the embedding is
            # sequence-sharded). Every SP rank already holds the same ``input_ids``
            # (the dataloader replicates each shard); pad it to a multiple of
            # ``sp_size`` and take this rank's contiguous chunk. ``encode_post``
            # all-gathers the embeds back.
            self._sp_own_len = input_ids.size(0)
            input_ids = sp_pad(input_ids, dim=0, pad_value=0)
            input_ids = slice_input_tensor(input_ids, dim=0, padding=False, group=get_parallel_state().sp_group)
        return {"input_ids": input_ids}

    @post_forward("encode")
    def encode_post(self, inputs_embeds: torch.Tensor) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather token shards back to the full sequence
            # (autograd-aware; backward sums grads across the SP group), then drop
            # the SP pad tail so the packed length matches its carrier below.
            inputs_embeds = gather_outputs(inputs_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            inputs_embeds = inputs_embeds.narrow(0, 0, self._sp_own_len)
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        batch_shape = self._encode_batch_shape
        self._encode_batch_shape = None
        scatter_text_encoder_embeds(conversation, unflatten(inputs_embeds, batch_shape))
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
        # No SP sharding here: the lm-head loss is a token-weighted mean over the whole
        # ``dp_sp`` (``fsdp_group``) mesh (``modeling.decode`` →
        # ``reduce_sequence_parallel_loss``), so each rank scoring only its OWN span
        # yields the identical global loss + gradient regardless of the DP/SP split.
        # Skipping the gather-concat keeps peak logits at this rank's own span.
        return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    @post_forward("decode")
    def decode_post(self, loss: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        # V2 single-loss protocol: drop logits, rename ``loss`` → ``_loss``.
        if loss is not None:
            return {"_loss": loss, "conversation_list": conversation}
        # TODO: scatter logits for rl training
        return {"conversation_list": conversation}

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel lookup when the ``emb`` extra-parallel group is active.

        Overrides the native ``self.embed_tokens(input_ids)`` shortcut so every
        family picks up ``AllToAllEmbedding`` under ``emb`` (mixed in via
        :class:`~....mixins.emb_parallel_mixin.EmbParallelMixin`); off ``emb`` it
        is exactly the native call.
        """
        return self.emb_parallel_lookup(self.embed_tokens, input_ids)

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> torch.Tensor:
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            input_ids.extend(self._chat_template.pack_input_ids(sample))
        # ``naflatten`` keeps the shape on CPU (avoids the post-forward D2H sync);
        # the flat ids may be CPU (worker path) or device (fallback) — move once.
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        input_ids = input_ids.to(self.device, non_blocking=True)
        return input_ids

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif part.type in ("image", "video"):
                    # Vision segment carries projected patch embeds; keep one row
                    # (no label) so the sequence stays aligned, like the backbone.
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)  # CPU

        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        # Single H2D move (labels are tiny host-side bookkeeping); the loss forward
        # orders after the backbone kernels on the GPU stream without a CPU stall.
        shift_labels = shift_labels.to(device=hidden_states.device, non_blocking=True)
        return hidden_states, shift_labels

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tied-head projection that stays FSDP2-safe.

        The native :meth:`~.modeling.TextEncoder._project` reads
        ``embed_tokens.weight`` directly, which is only safe off-FSDP (pure HF
        inference): under FSDP2 that read bypasses the embedding's own
        unshard/reshard hook (``encode`` and ``decode`` are separate graph-node
        forward calls, so there's no guarantee ``embed_tokens`` was unsharded by
        this call), leaving a sharded ``DTensor`` that a plain ``F.linear``
        can't multiply against the (unsharded) activation. ``emb_parallel_project``
        (mixed in via :class:`~....mixins.emb_parallel_mixin.EmbParallelMixin`)
        explicitly gathers the weight first — vocab-parallel when the ``emb``
        extra-parallel group is active, otherwise a plain ``full_tensor()``
        all-gather — so this override, not the native one, must run at train time.
        """
        if not self.config.tie_word_embeddings:
            return self.lm_head(hidden_states)
        return self.emb_parallel_project(hidden_states, self.embed_tokens.weight)

    def decode(
        self,
        hidden_states: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict:
        """FSDP2/SP-safe counterpart of :meth:`~.modeling.TextEncoder.decode`.

        Two departures from the native version, both runtime concerns the pure-HF
        layer deliberately stays out of:

        (a) The loss goes through the ops-selected cross-entropy backend rather
        than a bare ``F.cross_entropy``, so an untied bias-free head gets the
        fused linear+CE kernel (Liger / chunk_loss) and never materializes the
        ``[T, V]`` logits. Passing ``logits=None`` + ``hidden_states`` + ``weights``
        is what opts into that contract; a tied or biased head cannot (the fused
        kernels take a weight tensor only, no bias, and the tied weight needs the
        gather in :meth:`_project`), so it keeps an explicit projection and eager CE.

        (b) ``loss_reduction_group=fsdp_group`` token-weights the loss mean across
        the whole ``dp_sp`` mesh — required once SP is enabled, since each rank only
        scores its own (unsliced, per :meth:`decode_pre`) span.
        ``ForCausalLMLoss`` applies exactly the
        :func:`~veomni.distributed.sequence_parallel.reduce_sequence_parallel_loss`
        this method used to call inline.

        (c) A span with no supervised token scores 0.0, matching the native
        clamped denominator required by constraint 7b. Every fused backend
        normalizes by its own supervised-token count with no way to clamp it
        (Liger divides by ``n_non_ignore``, chunk_loss recounts the labels), so
        such a span is routed to the eager branch and given an explicit
        denominator instead of dividing 0 by 0.
        """
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        target_labels = labels if labels is not None else shift_labels
        fsdp_group = get_parallel_state().fsdp_group
        num_supervised_tokens = None if target_labels is None else (target_labels != -100).sum()

        if target_labels is None:
            # Inference path: materialize logits because no loss is requested.
            logits = self._project(hidden_states)
        elif self.lm_head is not None and self.lm_head.bias is None and num_supervised_tokens > 0:
            loss, logits, _ = self.loss_function(
                logits=None,
                labels=target_labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                loss_reduction_group=fsdp_group,
            )
        else:
            logits = self._project(hidden_states)
            loss, _, _ = ForCausalLMLoss(
                logits=logits,
                labels=target_labels,
                shift_labels=shift_labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_supervised_tokens.clamp(min=1),
                loss_reduction_group=fsdp_group,
                cross_entropy_fn=eager_cross_entropy,
            )

        return {
            "loss": loss,
            "logits": logits,
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None
        # Active sample's packed token count. Under SP the encode output-gather hook
        # (``encode_sp_post``) narrows the all-gathered (seq-padded) embeds to it.
        self._sp_own_len: Optional[int] = None


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the text encoder (wte + lm_head)."""

    config: TextEncoderConfig

    def estimate_flops(self, seqlens: List[int]) -> float:
        # This module owns wte (an embedding lookup ≈ 0 FLOPs) + the lm_head
        # projection (hidden → vocab); the transformer layers belong to the
        # backbone module. fwd+bwd ⇒ 6x; lm_head params = vocab * hidden.
        lm_head_n = self.config.vocab_size * self.config.hidden_size
        return 6 * lm_head_n * sum(seqlens) / 1e12

    # Token lengths come from ``metric_meter_set_seqlens`` in ``encode_pre`` (the
    # full pre-slice packed length); ``decode`` stashes nothing and contributes
    # no tokens (its lm_head FLOPs are covered by the ``encode`` count). So the
    # default ``metric_meter_token_lengths`` (drains the stash) is used as-is.


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin, EmbParallelMixin):
    """Shared training / inference plumbing for every text encoder.

    No ``InferenceMixin`` here: FSM ``generate`` and its sampling /
    prompt-embedding helpers already live natively on
    :class:`~.modeling.TextEncoder` via its own
    :class:`~.modeling.InferenceMixin` (and each family's ``generate``
    override) — every ``XxxTextEncoderAccelerated(VeOmniMixin, XxxTextEncoder)``
    inherits that through its native base, so nothing needs to be re-declared
    here. Concrete modules (janus / qwen3 / qwen3vl / bagel) subclass this and
    define ``XxxTextEncoderPreprocessor`` in their ``processing.py`` by
    subclassing
    :class:`~veomni.models.seed_omni.modules.base.text_encoder.processing.TextEncoderPreprocessor`
    and implementing :meth:`~veomni.models.seed_omni.modules.base.text_encoder.processing.TextEncoderPreprocessor.build_chat_template`
    with the module-local chat template.  Register via ``preprocessor_class`` on
    the family's HF-native ``modeling.py`` class (parallel to ``image_processor_class``
    on vision/codec modules) — :class:`~veomni.models.seed_omni.processing_omni.OmniProcessor`
    resolves it through ``OMNI_MODEL_REGISTRY`` (the native class), so it must live
    there, not on this accelerated mixin. Each module's ``accelerated.py`` still owns
    the ``encode_pre`` / ``encode_post`` / ``decode_pre`` / ``decode_post`` pass-through
    hooks.
    """

    _chat_template: TextEncoderChatTemplate

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = TextEncoderChatTemplate(tokenizer)


class TextEncoderAccelerated(VeOmniMixin, TextEncoder):
    """Training/runtime text encoder — vocab-parallel embed + SP-aware CE.

    ``_embed_tokens`` / ``_project`` / ``decode`` overrides (FSDP2-/``emb``-safe)
    live on :class:`TrainingMixin` above so every family's own accelerated
    composition — which mixes in *its own* ``TrainingMixin(BaseTrainingMixin)``
    rather than this class — inherits them too.
    """


__all__ = ["TextEncoderAccelerated", "scatter_text_encoder_embeds"]
