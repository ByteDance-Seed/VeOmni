"""Janus VQVAE codec + generation projection head.

``JanusVqvae(JanusVqvaeModuleMixin)`` — codec weights here;
encode/decode graph hooks in ``modulemixin.py``.

Call-site split
---------------
* :meth:`pre_forward` — stash ``conversation_list``; ``method="encode"`` pulls
  assistant ``image`` pixels, ``method="decode"`` assembles llama hidden rows +
  ``gen_ids`` labels.
* :meth:`encode` — ``pixel_values`` → ``image_embeds`` + ``vq_token_ids``.
* :meth:`decode` — training CE: ``hidden_states`` + ``labels`` → ``loss``.
* :meth:`post_forward` — write ``image_embeds`` / ``janus_vqvae_labels`` back onto
  ``conversation_list`` (encode path).

Graph entry points (YAML ``module: janus_vqvae.<method>``):

  ``encode``   — training encode node (pixels → image embeds + VQ ids).
  ``decode``   — training VQ cross-entropy loss head.
  ``generate`` — inference VQ AR step (lm_head → sample → embed → merge).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from veomni.utils import helper

from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, maybe_merge_outputs, seal_outputs
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaePreprocessor, JanusVqvaeProcessor


logger = helper.create_logger(__name__)


class InferenceMixin:
    """FSM ``generate`` (VQ AR sampling + CFG) — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`JanusVqvae`'s bases: ``OmniPreTrainedModel`` ships a no-op
    ``finalize`` default (kept as a safety net for modules that don't need
    real inference state), and MRO resolves left-to-right — put second, that
    no-op would shadow the real one below.
    """

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


class JanusVqvae(InferenceMixin, OmniPreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    image_processor_class = JanusVqvaeProcessor
    preprocessor_class = JanusVqvaePreprocessor
    base_model_prefix = "janus_vqvae"
    main_input_name = "pixel_values"
    _no_split_modules: list = []

    def __init__(self, config: JanusVqvaeConfig):
        super().__init__(config)
        self.config = config
        self.vqmodel = JanusVQVAE._from_config(config.vq_config)
        self.generation_embeddings = nn.Embedding(config.vq_config.num_embeddings, config.vq_config.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(config.vq_config)
        self.generation_head = JanusVQVAEHead(config.vq_config)
        self._image_processor: JanusVqvaeProcessor = None
        self._vq_buffer: List[int] = []
        self.post_init()

    def freeze_model(self) -> None:
        """Partial freeze: only the inner VQVAE codec (``vqmodel``).

        Matches the Janus recipe — the generation projection heads
        (``generation_embeddings`` / ``generation_aligner`` /
        ``generation_head``) stay trainable, so this module still gets an
        optimizer (over those heads).  Overrides the base whole-module
        default; gated on ``config.freeze`` (default ``True``).
        """
        if self.config.freeze:
            self.vqmodel.requires_grad_(False)

    def _encode_pixels(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            vq_out = self.vqmodel.encode(pixel_values)
        vq_token_ids = vq_out.image_tokens
        image_embeds = self.generation_aligner(self.generation_embeddings(vq_token_ids))
        if vq_token_ids.dim() == 1:
            b = pixel_values.size(0)
            vq_token_ids = vq_token_ids.reshape(b, -1)
            image_embeds = image_embeds.reshape(b, vq_token_ids.size(1), image_embeds.size(-1))
        return {"image_embeds": image_embeds, "vq_token_ids": vq_token_ids}

    def _dummy_encode_outputs(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Zero stand-ins shaped exactly like :meth:`_encode_pixels`' output (same
        batch + per-image token count) for the non-FSDP dummy, whose codec forward
        is skipped (no gradient anchor needed without FSDP). Emitting real-shaped
        zeros instead of ``None`` lets the pre/post hooks treat every batch
        identically — no ``None`` / dummy special-casing downstream."""
        vq = self.config.vq_config
        downsample = 2 ** (len(vq.channel_multiplier) - 1)
        b, _, h, w = pixel_values.shape
        num_tokens = (h // downsample) * (w // downsample)
        return {
            "image_embeds": pixel_values.new_zeros(b, num_tokens, vq.projection_dim),
            "vq_token_ids": pixel_values.new_zeros(b, num_tokens, dtype=torch.long),
        }

    def encode(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        del is_dummy
        return self._encode_pixels(pixel_values)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs, is_dummy
        if hidden_states is None or labels is None:
            return {}
        # Always route through ``_vq_loss`` — it reduces the loss over the whole
        # ``dp_sp`` (``fsdp_group``) mesh, so EVERY data-parallel rank must reach
        # the collective (an early ``is_dummy`` return on only some DP ranks would
        # desync it and hang). An all-dummy span is handled naturally: its rows are
        # ``-100`` so CE contributes 0 and the clamped denominator yields 0.0 (not
        # NaN), while ``generation_head`` still runs as the FSDP gradient anchor.
        return {"loss": self._vq_loss(hidden_states, labels)}

    def _vq_loss(self, hidden_states: torch.Tensor, gt_token_ids: torch.Tensor) -> torch.Tensor:
        # ``hidden_states`` is already teacher-forcing aligned in
        # ``_prepare_decode_inputs`` (row i = hidden after the previous token,
        # predicting VQ id i), so no further shift here — shifting again would
        # mis-align by one and bleed across concatenated image spans. Labels are VQ
        # codebook ids with -100 on dummy rows, so CE always uses ``ignore_index=-100``.
        # ``generation_head`` runs unconditionally (FSDP gradient anchor), and the
        # loss is a token-weighted mean over VALID rows: summing CE then dividing by
        # the clamped valid-token count makes an all-dummy span yield 0.0 instead of
        # NaN, on every path.
        labels = gt_token_ids.to(hidden_states.device)
        logits = self.generation_head(hidden_states)
        flat_labels = labels.reshape(-1)
        ce_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_labels, ignore_index=-100, reduction="sum")
        n_valid_local = (flat_labels != -100).sum().clamp(min=1)
        return ce_sum / n_valid_local


__all__ = ["InferenceMixin", "JanusVqvae"]
