from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs, slice_input_tensor
from ......utils import helper
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import (
    ConversationItem,
    is_dummy,
    iter_desired_items,
    maybe_merge_outputs,
    seal_outputs,
)
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaePreprocessor, JanusVqvaeProcessor


logger = helper.create_logger(__name__)

_SOURCE = "janus_vqvae"


class JanusVqvaeModuleMixin(ModuleMixin):
    config: JanusVqvaeConfig
    _image_processor: JanusVqvaeProcessor
    preprocessor_class = JanusVqvaePreprocessor

    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None
        # Active sample's image count. Under SP the encode output-gather hook
        # (``encode_sp_post``) narrows the all-gathered (batch-padded) embeds + VQ ids.
        self._sp_own_len: Optional[int] = None

        # Inference state
        self._vq_buffer: List[int] = []

    # Training hooks — one pre/post pair per call-site (tagged with its method),
    # routed by the ModuleMixin.pre_forward / post_forward dispatchers.
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
