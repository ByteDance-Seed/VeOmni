from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import (
    gather_outputs,
    slice_input_tensor,
    sp_gather_seqs,
    sp_pad,
    sp_take_own_seq,
)
from ......utils import helper
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import (
    CPUPreprocessor,
    ModuleMixin,
    post_forward,
    pre_forward,
)
from ....utils.conversation import (
    ConversationItem,
    is_dummy,
    iter_desired_items,
    maybe_merge_outputs,
    seal_outputs,
)
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaeProcessor


logger = helper.create_logger(__name__)

_SOURCE = "janus_vqvae"


class JanusVqvaeCPUPreprocessor(CPUPreprocessor):
    """Worker-side image normalize for the VQVAE (generation) codec.

    Holds only the (picklable) VQVAE image processor + a CPU zero-pixel template
    — never the model. Runs the HF image processor on **CPU** (bf16, to halve
    IPC); writes the pixel tensor back into each ``assistant``-image item.
    When a micro-batch has **no** assistant image, appends a ``role="dummy"``
    placeholder per sample carrying the zero pixels (the codec + generation heads
    still run on it in the GPU forward for the FSDP gradient anchor).
    """

    def __init__(self, image_processor: JanusVqvaeProcessor, dtype: Any, dummy_pixel_values: torch.Tensor) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (C, H, W), model dtype

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["assistant"]))
        if image_items:
            # Real assistant images present → normalize them; no dummy needed.
            # Tag with the module source so the decode path can pick up real gen
            # images and dummies uniformly (single ``source == _SOURCE`` filter).
            pixel_values = self._image_processor(images=[it.value for it in image_items], return_tensors="pt")[
                "pixel_values"
            ]
            for it, px in zip(image_items, pixel_values, strict=True):
                it.value = px.to(dtype=self._dtype)
                it.source = _SOURCE
        elif not inference:
            # Training only: inject a dummy so the codec + generation heads still
            # run (FSDP gradient anchor). At inference the assistant image is
            # generated token-by-token, so there is nothing to pre-encode here.
            for sample in conversation_list:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values,
                        role="dummy",
                        source=_SOURCE,
                    )
                )


class JanusVqvaeModuleMixin(ModuleMixin):
    config: JanusVqvaeConfig
    _image_processor: JanusVqvaeProcessor

    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None
        # Per-rank image counts + this rank's index within the module SP group
        # (set in ``encode_pre``, consumed in ``encode_post`` to narrow the
        # gathered batch back to this rank's own images when module SP > 1).
        self._sp_rep_lengths: Optional[List[int]] = None
        self._sp_group_index: int = 0

        # Inference state
        self._vq_buffer: List[int] = []

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side image normalize (see :class:`JanusVqvaeCPUPreprocessor`)."""
        dummy = self.dummy_inputs()["pixel_values"]
        return JanusVqvaeCPUPreprocessor(self._image_processor, self.dtype, dummy)

    # Training hooks — one pre/post pair per call-site (tagged with its method),
    # routed by the ModuleMixin.pre_forward / post_forward dispatchers.
    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        # Real gen images and worker-built dummies both carry source == _SOURCE
        # (normalized on CPU by the JanusVqvaeCPUPreprocessor); stack + move.
        items = list(iter_desired_items(conversation_list, types=["image"], sources=[_SOURCE]))
        pixel_values = torch.stack([it.value for it in items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        # Single batch-level dummy flag: True only when *every* fed image is a
        # worker-injected dummy (the whole batch is dummy); if any image is real
        # it is False. Passed as a scalar so ``encode`` can short-circuit to a
        # dummy output when there is no anchor to maintain.
        is_dummy_flag = all(is_dummy(it) for it in items)
        # Metering: this rank's OWN image count, pre-gather / pre-slice. The meter
        # sums over the DP group only, so each rank reports just its own images
        # (not the module_sp ranks SP aggregates) — identical to the non-SP run
        # for both SP-disabled and per-module SP.
        self._metric_meter_stash_tokens(int(pixel_values.shape[0]))
        self._sp_rep_lengths = None
        self._sp_group_index = 0
        ps = get_parallel_state()
        if ps.sp_enabled:
            # The VQVAE codec attention does NOT honor Ulysses, so SP shards the
            # image *batch* dim. First aggregate the module SP group's distinct
            # image batches (module_sp ranks), then each rank encodes its own
            # images (zero-padded to a multiple of module sp_size); ``encode_post``
            # gathers the embeds + VQ ids back and restores the per-rank layout.
            pixel_values, rep_lengths, group_index = sp_gather_seqs(pixel_values, dim=0)
            self._sp_rep_lengths = rep_lengths
            self._sp_group_index = group_index
            if len(rep_lengths) > 1:
                # The aggregated batch is "all dummy" only if every SP rank is.
                # Reduce with MIN so all ranks in the module SP group agree.
                flag = torch.tensor([1 if is_dummy_flag else 0], device=self.device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=ps.sp_group)
                is_dummy_flag = bool(flag.item())
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=ps.sp_group)
        return {"pixel_values": pixel_values, "is_dummy": is_dummy_flag}

    def _metric_meter_stash_tokens(self, num_images: int) -> None:
        # VQ image tokens per image; counted on encode only (decode is the
        # generation-head path and stashes nothing).
        self.metric_meter_set_seqlens("encode", [int(self._image_processor.num_image_tokens)] * num_images)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        image_embeds = outputs["image_embeds"]
        vq_token_ids = outputs["vq_token_ids"]
        # encode returns one (embed, VQ-id) row per fed item, in source order;
        # scatter them back onto the same source items (real or dummy alike).
        items = list(iter_desired_items(conversation, types=["image"], sources=[_SOURCE]))
        ps = get_parallel_state()
        if ps.sp_enabled:
            # Gather per-rank image embeds + VQ ids (dim 0) over the MODULE SP
            # group, drop the zero-pad tail to recover the full module-group
            # batch, then narrow to this rank's own slice.
            full_count = sum(self._sp_rep_lengths)
            image_embeds = gather_outputs(
                image_embeds, gather_dim=0, padding_dim=0, unpad_dim_size=full_count, group=ps.sp_group
            )
            vq_token_ids = gather_outputs(
                vq_token_ids, gather_dim=0, padding_dim=0, unpad_dim_size=full_count, group=ps.sp_group
            )
            image_embeds = sp_take_own_seq(
                image_embeds, dim=0, seg_lengths=self._sp_rep_lengths, sp_rank=self._sp_group_index
            )
            vq_token_ids = sp_take_own_seq(
                vq_token_ids, dim=0, seg_lengths=self._sp_rep_lengths, sp_rank=self._sp_group_index
            )
            self._sp_rep_lengths = None
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
        ps = get_parallel_state()
        if ps.sp_enabled:
            # Shard the generation-head loss span across the MODULE SP group.
            # First aggregate the module SP group's distinct spans into the
            # module-group sequence (autograd-aware), then pad the packed (1, S, D)
            # span on its seq dim (labels pad with -100 so the tail is ignored) and
            # score only this rank's slice. ``modeling.decode`` aggregates via
            # ``reduce_sequence_parallel_loss`` over the module SP group (also
            # undoing FSDP's /sp grad averaging on the dp_shard_sp mesh). The loss
            # is a scalar, so no per-rank restore is needed in ``decode_post``.
            hidden_states, rep_lengths, _ = sp_gather_seqs(hidden_states, dim=1)
            labels, _, _ = sp_gather_seqs(labels, dim=1)
            if len(rep_lengths) > 1:
                # The span is "all dummy" only if every SP rank is. Reduce with MIN
                # so every rank in the module SP group takes the same decode branch
                # (keeping the loss collective symmetric). Mixed ranks (flag False)
                # still ignore their dummy rows via the -100 labels set in
                # ``_prepare_decode_inputs``.
                flag = torch.tensor([1 if is_dummy_flag else 0], device=hidden_states.device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=ps.sp_group)
                is_dummy_flag = bool(flag.item())
            hidden_states = sp_pad(hidden_states, dim=1, pad_value=0)
            labels = sp_pad(labels, dim=1, pad_value=-100)
            hidden_states = slice_input_tensor(hidden_states, dim=1, padding=False, group=ps.sp_group)
            labels = slice_input_tensor(labels, dim=1, padding=False, group=ps.sp_group)
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
        # a mixed real/dummy span — which happens when a module SP group of
        # module_sp ranks aggregates distinct per-rank spans — still scores its
        # real rows and ignores its dummy rows.
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

    # Inference hooks
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        tail_part = conversation_list[-1]
        hidden_states: torch.Tensor = tail_part.value
        hidden_states = hidden_states.to(self.device)
        batch_size = hidden_states.size(0)
        sampling = self._extract_sampling_kwargs(generation_kwargs)
        cfg_w = sampling.pop("guidance_scale", None)

        if batch_size == 2 and cfg_w > 1.0:
            cond_logits = self.generation_head(hidden_states[:1, -1:, :]).squeeze(1)
            uncond_logits = self.generation_head(hidden_states[1:, -1:, :]).squeeze(1)
            last_logits = uncond_logits + cfg_w * (cond_logits - uncond_logits)
        elif batch_size == 1:
            last_logits = self.generation_head(hidden_states[:, -1:, :]).squeeze(1)
        else:
            raise NotImplementedError(
                f"JanusVqvae.generate received hidden_states with B={batch_size}. "
                "Supported: B=1 (no CFG) or B=2 (CFG cond/uncond pair)."
            )

        sampled = self._sample_vq_token(last_logits, **sampling)
        token_id_int = int(sampled[0].item())
        self._vq_buffer.append(token_id_int)

        outputs: Dict[str, Any] = {}
        target = self._image_processor.num_image_tokens
        if len(self._vq_buffer) == target:
            generated = self._emit_buffered_image()

            tail_part = conversation_list.pop()
            seal_outputs(conversation_list, new_type="image")
            conversation_list.append(tail_part)

            outputs["generated"] = generated
            outputs[FSM_SIGNAL_KEY] = "image_complete"
        else:
            input_embeds = self.generation_aligner(self.generation_embeddings(sampled))
            tail_part.value = input_embeds
            maybe_merge_outputs(conversation_list)

        outputs["conversation_list"] = conversation_list
        return outputs

    def _emit_buffered_image(self) -> Optional[Dict[str, Any]]:
        token_ids = torch.tensor([self._vq_buffer], dtype=torch.long, device=self.device)
        self._vq_buffer.clear()
        with torch.inference_mode():
            decoded = self.vqmodel.decode(token_ids).permute(0, 2, 3, 1)
        if self._image_processor is None:
            raise RuntimeError(
                "JanusVqvae: cannot postprocess VQVAE output — no processor was "
                "loaded. Ensure `preprocessor_config.json` ships next to the weights."
            )
        image_pil = self._image_processor.postprocess(decoded)[0]
        return {"type": "image", "value": image_pil}

    def reset_local_inference_state(self) -> None:
        self._vq_buffer.clear()

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        del ctx
        target = self._image_processor.num_image_tokens
        n = len(self._vq_buffer)
        if n == 0:
            return {}
        if n < target:
            logger.warning_rank0(
                f"JanusVqvae.finalize: incomplete VQ grid ({n}/{target} tokens) — "
                "discarding partial sequence (no image emitted)."
            )
            self._vq_buffer.clear()
            return {}
        if n == target:
            generated = self._emit_buffered_image()
            return {"generated": generated}
        raise RuntimeError(
            "JanusVqvae.finalize: VQ buffer overflowed the grid — "
            "_emit_buffered_image should have fired inside generate() when n == target."
        )

    @staticmethod
    def _extract_sampling_kwargs(generation_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
            "guidance_scale": 1.0,
        }
        if generation_kwargs:
            for k in ("temperature", "top_p", "do_sample", "guidance_scale"):
                if k in generation_kwargs:
                    merged[k] = generation_kwargs[k]
        return merged

    @staticmethod
    def _sample_vq_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
            logits = logits.scatter(1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class JanusVqvaeMetricMeterMixin(MetricMeterMixin):
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


__all__ = ["JanusVqvaeModuleMixin", "JanusVqvaeMetricMeterMixin"]
