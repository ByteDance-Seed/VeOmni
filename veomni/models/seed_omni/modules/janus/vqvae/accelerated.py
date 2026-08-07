"""VeOmni-accelerated JanusVqvae — training-graph hooks only.

``generate()``, ``finalize()`` and VQ-buffer inference state live on the
native :class:`JanusVqvae` in ``modeling.py`` — this file only carries the
training encode/decode graph hooks + the FSDP dummy-encode override.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs, slice_input_tensor
from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from .configuration import JanusVqvaeConfig
from .modeling import JanusVqvae
from .processing import JanusVqvaeProcessor


_SOURCE = "janus_vqvae"


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`JanusVqvae` modeling APIs."""

    config: JanusVqvaeConfig
    device: torch.device
    dtype: torch.dtype
    _image_processor: JanusVqvaeProcessor

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        # Active sample's image count. Under SP the encode output-gather hook
        # (``encode_sp_post``) narrows the all-gathered (batch-padded) embeds + VQ ids.
        self._sp_own_len: Optional[int] = None

    # Training hooks — one pre/post pair per call-site (tagged with its method),
    # routed by :class:`BaseMixin` ``pre_forward`` / ``post_forward`` dispatchers.
    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        # Real gen images and worker-built dummies both carry source == _SOURCE
        # (normalized on CPU by the JanusVqvaePreprocessor); stack + move.
        items = list(iter_desired_items(conversation_list, types=["image"], sources=[_SOURCE]))
        pixel_values = torch.stack([it.value for it in items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        # Single batch-level dummy flag: True only when *every* fed image is a
        # worker-injected dummy (the whole batch is dummy); if any image is real
        # it is False. Passed as a scalar so ``encode`` can short-circuit to a
        # dummy output when there is no anchor to maintain.
        is_dummy_flag = all(is_dummy(it) for it in items)
        # Metering: this rank's OWN image count, stashed BEFORE the SP slice below.
        # The meter sums over the DP group only, so each rank reports just its own
        # images (not the SP peers that hold the same replicated batch) — identical
        # to the non-SP run for both SP-disabled and uniform SP.
        self._metric_meter_stash_tokens(int(pixel_values.shape[0]))

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its ``1/sp_size`` slice of the image
            # batch (the VQVAE codec shards the batch dim — its attention is not
            # Ulysses). Every SP rank already holds the same image batch (the
            # dataloader replicates each shard); pad it to a multiple of ``sp_size``
            # and take this rank's contiguous chunk. ``is_dummy`` is dropped: the SP
            # path is a training path that always runs the codec (the FSDP grad
            # anchor). ``encode_post`` all-gathers the shards back.
            self._sp_own_len = pixel_values.size(0)
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=get_parallel_state().sp_group)
            return {"pixel_values": pixel_values}
        # ``is_dummy`` stays for the plain (non-SP) encode: a non-SP inference/eval
        # short-circuit that skips the codec for an all-dummy batch with no anchor.
        return {"pixel_values": pixel_values, "is_dummy": is_dummy_flag}

    def _metric_meter_stash_tokens(self, num_images: int) -> None:
        # VQ image tokens per image; counted on encode only (decode is the
        # generation-head path and stashes nothing).
        self.metric_meter_set_seqlens("encode", [int(self._image_processor.num_image_tokens)] * num_images)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        image_embeds = outputs["image_embeds"]
        vq_token_ids = outputs["vq_token_ids"]
        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather batch shards back to the full image batch
            # (autograd-aware; backward sums grads across the SP group), then drop
            # the SP pad tail so the count matches its conversation items below.
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            vq_token_ids = gather_outputs(vq_token_ids, gather_dim=0, group=get_parallel_state().sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._sp_own_len)
            vq_token_ids = vq_token_ids.narrow(0, 0, self._sp_own_len)
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        # encode returns one (embed, VQ-id) row per fed item, in source order; scatter
        # them back onto the same source items (real or dummy alike).
        items = list(iter_desired_items(conversation, types=["image"], sources=[_SOURCE]))
        for item, emb, ids in zip(items, image_embeds, vq_token_ids, strict=True):
            item.value = emb.to(dtype=self.dtype)
            item.meta["janus_vqvae_labels"] = ids.to(dtype=torch.long)
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        hidden_states, labels, is_dummy_flag = self._prepare_decode_inputs(conversation_list)
        # No SP sharding here: the generation-head loss is a token-weighted mean over
        # the whole ``dp_sp`` (``fsdp_group``) mesh (``modeling.decode`` →
        # ``reduce_sequence_parallel_loss``), so each rank scoring only its OWN span
        # yields the identical global loss + gradient regardless of the DP/SP split.
        # Skipping the gather-concat keeps peak logits at this rank's own span.
        return {"hidden_states": hidden_states, "labels": labels, "is_dummy": is_dummy_flag}

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        loss = outputs.pop("loss", None)
        if loss is not None:
            outputs["_loss"] = loss
        outputs["conversation_list"] = conversation
        return outputs

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        hidden_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        # Real gen images and worker-built dummies are both tagged ``source ==
        # _SOURCE`` (and carry real-shaped ``janus_vqvae_labels`` after encode), so
        # they build the teacher-forcing span identically. ``all_dummy`` is the
        # batch-level flag returned to ``decode``: True only when *every* gen span
        # is a dummy. Dummy spans are additionally masked with -100 labels so that
        # a mixed real/dummy batch still scores its real rows and ignores its dummy
        # rows.
        all_dummy = True
        saw_span = False
        for sample in conversation_list:
            prev_hidden: torch.Tensor | None = None
            for part in sample:
                hidden_states = part.value
                if part.source == _SOURCE:
                    saw_span = True
                    if prev_hidden is None:
                        raise ValueError(
                            "JanusVqvae._prepare_decode_inputs: generation image has no preceding hidden state."
                        )
                    vq_labels = (
                        part.meta["janus_vqvae_labels"].to(device=hidden_states.device, dtype=torch.long).reshape(-1)
                    )
                    assert vq_labels.shape[0] == hidden_states.shape[0]
                    span_hidden = torch.cat([prev_hidden[-1:], hidden_states[:-1]], dim=0)
                    hidden_chunks.append(span_hidden)
                    if is_dummy(part):
                        vq_labels = torch.full_like(vq_labels, -100)
                    else:
                        all_dummy = False
                    label_chunks.append(vq_labels)
                prev_hidden = hidden_states

        hidden_states = torch.cat(hidden_chunks, dim=0).unsqueeze(0)
        labels = torch.cat(label_chunks, dim=0).unsqueeze(0)
        return hidden_states, labels, (all_dummy and saw_span)

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vq_config
        size = self._image_processor.size
        height = size.get("height")
        width = size.get("width")
        return {
            "pixel_values": torch.zeros(cfg.in_channels, height, width, dtype=self.dtype),
        }


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for the Janus VQVAE codec + generation head."""

    config: JanusVqvaeConfig
    _image_processor: JanusVqvaeProcessor

    def estimate_flops(self, seqlens: List[int]) -> float:
        # The inner VQ codec (``vqmodel``) is a frozen conv stack (``config.freeze``
        # defaults to True → forward-only, no backward) whose conv FLOPs are
        # architecture-specific and not modeled here; the trainable generation
        # head runs on ``decode``, which we deliberately don't count. So this
        # module injects no FLOPs into the global MFU — only its token count is
        # tracked above. (Add the codec conv estimate here if you need it.)
        del seqlens
        return 0.0


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    """``generate()`` / ``finalize()`` and the VQ-buffer inference state already
    live on the native :class:`~.modeling.JanusVqvae` (via its own
    :class:`~.modeling.InferenceMixin`), so no ``InferenceMixin`` is needed here.
    """

    config: JanusVqvaeConfig
    _image_processor: JanusVqvaeProcessor


class JanusVqvaeAccelerated(VeOmniMixin, JanusVqvae):
    """Training/runtime VQVAE — FSDP dummy encode + SP-aware VQ loss."""

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        if is_dummy and not (self.training and get_parallel_state().fsdp_enabled):
            return self._dummy_encode_outputs(pixel_values)
        return self._encode_pixels(pixel_values)

    def _vq_loss(self, hidden_states: torch.Tensor, gt_token_ids: torch.Tensor) -> torch.Tensor:
        from veomni.distributed.sequence_parallel import reduce_sequence_parallel_loss

        labels = gt_token_ids.to(hidden_states.device)
        logits = self.generation_head(hidden_states)
        flat_labels = labels.reshape(-1)
        ce_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_labels, ignore_index=-100, reduction="sum")
        ps = get_parallel_state()
        n_valid_local = (flat_labels != -100).sum()
        local_mean = ce_sum / n_valid_local.clamp(min=1)
        return reduce_sequence_parallel_loss(local_mean, n_valid_local.to(local_mean.dtype), group=ps.fsdp_group)


__all__ = ["JanusVqvaeAccelerated"]
