"""
JanusVqvae — Janus' VQVAE + generation projection head as one OmniModule.

Mixin form: ``class JanusVqvae(OmniModule, PreTrainedModel)``.

Call-site split (V2)
--------------------
* :meth:`pre_forward` — stash ``conversation_list``; ``method="encode"`` pulls
  assistant ``image`` pixels, ``method="decode"`` assembles llama hidden rows +
  ``gen_ids`` labels.
* :meth:`encode` — pure encoder: ``pixel_values`` → ``image_embeds`` +
  ``vq_token_ids``.
* :meth:`decode` — training CE head: ``hidden_states`` + ``labels`` → ``_loss``.
* :meth:`post_forward` — write ``image_embeds`` / ``janus_vqvae_labels`` back onto
  ``conversation_list`` (encode path).

Graph entry points (YAML ``module: janus_vqvae.<method>``):

  ``encode``        — training encode node.
  ``decode``        — training loss head.
  ``decode_pixels`` — ``vq_token_ids`` → pixels.
  ``generate``      — inference VQ AR (lm_head → embed → merge).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVQVAEConfig
from transformers.models.janus.modeling_janus import (
    JanusVQVAE,
    JanusVQVAEAlignerMLP,
    JanusVQVAEHead,
)

from ......distributed.parallel_state import get_parallel_state
from ......utils import helper
from ....conversation import (
    ConversationItem,
    collect_modality_batch,
    is_dummy,
    iter_modality_items,
    maybe_merge_outputs,
    seal_outputs,
)
from ....generation_graph import FSM_SIGNAL_KEY
from ....module import OmniModule
from .configuration import JanusVqvaeConfig
from .processing import JanusVqvaeProcessor


logger = helper.create_logger(__name__)

# Default Janus-1.3B grid: 24 x 24 = 576 VQ tokens per image.
_DEFAULT_NUM_IMAGE_TOKENS = 576
_LLAMA_SOURCE = "janus_llama"


class JanusVqvae(OmniModule, PreTrainedModel):
    """VQVAE + generation head for Janus VQ image generation.

    The VQVAE encoder/decoder is frozen by default (matching the Janus
    paper).  Only the generation projection layers
    (``generation_embeddings``, ``generation_aligner``,
    ``generation_head``) are trainable.
    """

    config_class = JanusVqvaeConfig
    processor_class = JanusVqvaeProcessor
    base_model_prefix = "janus_vqvae"
    main_input_name = "pixel_values"
    _no_split_modules: list = []
    # The inner ``JanusVQVAE`` declares gradient-checkpointing support, so the
    # mixin advertises it too (keeps the wrapper's capability accurate and lets
    # the trainer's GC guard pass).  Note: the codec is frozen by default
    # (``config.freeze`` → :meth:`freeze_model`) and runs under ``no_grad`` in
    # training, so GC here is effectively inert — only the trainable
    # generation_* heads see grads.
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusVqvaeConfig):
        super().__init__(config)
        self.config = config
        vq_cfg = JanusVQVAEConfig(**config.vq_config) if config.vq_config else JanusVQVAEConfig()
        self._vq_cfg = vq_cfg
        self.vqmodel = JanusVQVAE._from_config(vq_cfg)
        self.generation_embeddings = nn.Embedding(vq_cfg.num_embeddings, vq_cfg.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(vq_cfg)
        self.generation_head = JanusVQVAEHead(vq_cfg)

        # NB: the ``config.freeze`` knob is honoured by :meth:`freeze_model`
        # (called once by the trainer after build), not here.

        # Per-image VQ-token buffer used by :meth:`generate` to accumulate
        # sampled tokens between FSM iterations.  Reset on each
        # ``image_complete`` signal; finalize keeps the decoded PIL
        # images so the caller can collect them after the run.
        self._vq_buffer: List[int] = []
        self._collected_images: List[Image.Image] = []

        # Auto-populated by :meth:`OmniModule.from_pretrained` from
        # ``<weights_path>/preprocessor_config.json``.  Used by
        # :meth:`generate` to convert the VQVAE's ``[-1, 1]`` float output
        # back into a PIL image — see :class:`JanusVqvaeProcessor`.
        self._processor: Optional[Any] = None
        self._conversation_carrier: Any = None
        self._decode_is_dummy: bool = False

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

    # ── JanusVqvae Main Function ─────────────────────────────────────────────

    def _encode_pixels(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``vqmodel.encode`` → ``image_embeds`` + ``vq_token_ids``."""
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            vq_out = self.vqmodel.encode(pixel_values)
        vq_token_ids = vq_out.image_tokens
        image_embeds = self.generation_aligner(self.generation_embeddings(vq_token_ids))
        if vq_token_ids.dim() == 1:
            b = pixel_values.size(0)
            vq_token_ids = vq_token_ids.reshape(b, -1)
            image_embeds = image_embeds.reshape(b, vq_token_ids.size(1), image_embeds.size(-1))
        return {"image_embeds": image_embeds, "vq_token_ids": vq_token_ids}

    def encode(self, pixel_values: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """VQVAE encode: ``pixel_values`` → ``image_embeds`` + ``vq_token_ids``."""
        if pixel_values is None and get_parallel_state().fsdp_enabled:
            dummy = self.dummy_inputs()
            return {**self._encode_pixels(dummy["pixel_values"]), "is_dummy": True}
        return self._encode_pixels(pixel_values)

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Training VQ CE head — ``hidden_states`` + ``labels`` → ``loss``."""
        del kwargs
        if hidden_states is None or labels is None:
            return {}
        if is_dummy:
            return {"loss": hidden_states.sum() * 0.0}
        return {"loss": self._vq_loss(hidden_states, labels)}

    def decode_pixels(self, vq_token_ids: torch.Tensor) -> torch.Tensor:
        """Decode a sequence of VQ token IDs to pixel values ``(B, H, W, 3)``."""
        pixel_values = self.vqmodel.decode(vq_token_ids)
        return pixel_values.permute(0, 2, 3, 1)

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        """Stash ``conversation_list``; route by graph call-site ``method``."""
        self._conversation_carrier = conversation_list
        if method == "encode":
            pixel_values = self._pixels_from_raw_images(
                collect_modality_batch(conversation_list, ["image"], roles=["assistant"])
            )
            return {"pixel_values": pixel_values}

        if method == "decode":
            hidden_states, labels, dummy_data = self._prepare_decode_inputs(conversation_list)
            return {"hidden_states": hidden_states, "labels": labels, "is_dummy": dummy_data}

        raise ValueError(f"JanusVqvae.pre_forward: unsupported method {method!r}")

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Flat concat of VQ hidden rows + ``janus_vqvae_labels`` for CE.

        Assistant generation ``image`` rows carry LLaMA hidden states that are
        one step ahead of the VQ labels: the first VQ token is predicted from
        the preceding item's last hidden, so each span prepends ``prev[-1:]`` and
        drops the image tail row ``hidden[-1:]``.

        When no real generation images exist (text-only micro-batch), fall back
        to the dummy image row's ``janus_vqvae_labels`` plus a one-row anchor
        hidden taken from any non-dummy item; :meth:`decode` turns that into a
        zero ``hidden.sum() * 0.0`` loss for FSDP graph alignment.
        """

        hidden_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        dummy_data: bool = False
        for sample in conversation_list:
            prev_hidden: torch.Tensor | None = None
            for part in sample:
                hidden_states = part.value
                if self._is_gen_image_item(part):
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
                    label_chunks.append(vq_labels)
                elif is_dummy(part) and part.meta["source"] == "janus_vqvae":
                    hidden_chunks.append(prev_hidden[-1:])
                    label_chunks.append(
                        part.meta["janus_vqvae_labels"].to(device=hidden_states.device, dtype=torch.long).reshape(-1)
                    )
                    dummy_data = True
                prev_hidden = hidden_states

        hidden_states = torch.cat(hidden_chunks, dim=0).unsqueeze(0)
        labels = torch.cat(label_chunks, dim=0).unsqueeze(0)
        return hidden_states, labels, dummy_data

    @staticmethod
    def _is_gen_image_item(part: ConversationItem) -> bool:
        return (
            part.type == "image"
            and part.role == "assistant"
            and not is_dummy(part)
            and isinstance(part.meta.get("janus_vqvae_labels"), torch.Tensor)
        )

    def post_forward(
        self,
        method: str,
        **outputs: Any,
    ) -> Dict[str, Any]:
        """Write encode outputs onto the stashed ``conversation_list`` carrier."""
        assert method in ["encode", "decode"]
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        if method == "encode":
            is_dummy = outputs.get("is_dummy", False)
            image_embeds = outputs.get("image_embeds")
            vq_token_ids = outputs.get("vq_token_ids")
            if is_dummy:  # append dummy item
                assert image_embeds.shape[0] == 1
                image_embeds = image_embeds.squeeze(0)
                vq_token_ids = vq_token_ids.squeeze(0)
                for sample in conversation:
                    sample.append(
                        ConversationItem(
                            type="image",
                            value=image_embeds,
                            role="dummy",
                            meta={
                                "source": "janus_vqvae",
                                "janus_vqvae_labels": vq_token_ids.to(dtype=torch.long),
                            },
                        )
                    )
            else:
                items = list(iter_modality_items(conversation, ["image"], roles=["assistant"]))
                for item, emb, ids in zip(items, image_embeds, vq_token_ids, strict=True):
                    item.value = emb
                    item.meta["janus_vqvae_labels"] = ids.to(dtype=torch.long)
            return {"conversation_list": conversation}

        if method == "decode":
            conversation = self._conversation_carrier
            self._conversation_carrier = None
            loss = outputs.pop("loss", None)
            if loss is not None:
                outputs["_loss"] = loss
            outputs["conversation_list"] = conversation
            return outputs

    def _vq_loss(self, hidden_states: torch.Tensor, gt_token_ids: torch.Tensor) -> torch.Tensor:
        """Token-mean VQ CE from ``hidden_states`` and padded ``gt_token_ids``."""
        labels = gt_token_ids.to(hidden_states.device)
        hidden_states = hidden_states[:, : labels.size(1)]
        logits = self.generation_head(hidden_states)
        v = logits.size(-1)

        if not labels.ne(-100).any():
            return logits[:, -1:, :].mean() * 0.0

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_sum = F.cross_entropy(
            shift_logits.reshape(-1, v),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_valid = (shift_labels != -100).sum().clamp(min=1)
        return ce_sum / n_valid

    # ── Inference (conversation-list aware) ───────────────────────────────────

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """VQ AR step: lm_head → sample → embed → merge ``output`` rows.

        VQ token ids accumulate in :attr:`_vq_buffer`.  On the final patch
        the buffer is decoded to a PIL image and ``image_complete`` is raised.
        """
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
        target = self._num_image_tokens()
        if len(self._vq_buffer) == target:
            # end of image generation, so the hidden states is for the next text ar step
            # do not need to do aligner(embeddings)
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
        """Decode a full VQ grid from ``_vq_buffer`` and clear it."""
        token_ids = torch.tensor([self._vq_buffer], dtype=torch.long, device=self.device)
        self._vq_buffer.clear()
        with torch.inference_mode():
            decoded = self.vqmodel.decode(token_ids).permute(0, 2, 3, 1)
        if self._processor is None:
            raise RuntimeError(
                "JanusVqvae: cannot postprocess VQVAE output — no processor was "
                "loaded. Ensure `preprocessor_config.json` ships next to the weights."
            )
        image_pil = self._processor.postprocess(decoded)[0]
        self._collected_images.append(image_pil)
        return {"type": "image", "value": image_pil}

    def reset_inference_state(self) -> None:
        """Wipe per-request buffers — called by :class:`OmniInferencer` between runs."""
        self._vq_buffer.clear()
        self._collected_images.clear()

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Flush or discard the VQ token buffer based on its fill level.

        Buffer empty → no-op (image may already have been emitted from
        :meth:`generate`).  Buffer full → decode and emit.  Partial grid →
        log a warning, discard, emit nothing.
        """
        target = self._num_image_tokens()
        n = len(self._vq_buffer)
        if n == 0:  # image generation not invoked yet
            return {}
        if n < target:
            logger.warning_rank0(
                f"JanusVqvae.finalize: incomplete VQ grid ({n}/{target} tokens) — "
                "discarding partial sequence (no image emitted)."
            )
            self._vq_buffer.clear()
            return {}
        elif n == target:
            generated = self._emit_buffered_image(device=self.device)
            return {"generated": generated}
        else:
            raise RuntimeError(
                "There's a bug in JanusVqvae.finalize, emit_buffered_image must be invoked when n == target in during generate"
            )
        # ── Internal helpers ─────────────────────────────────────────────────────

    def _num_image_tokens(self) -> int:
        cfg = self.config.vq_config
        if isinstance(cfg, dict):
            return int(cfg.get("num_image_tokens", _DEFAULT_NUM_IMAGE_TOKENS))
        val = getattr(cfg, "num_image_tokens", None)
        return int(val) if val is not None else _DEFAULT_NUM_IMAGE_TOKENS

    @staticmethod
    def _extract_sampling_kwargs(generation_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # `guidance_scale` rides in this dict but is consumed by the CFG
        # mix above (not by `_sample_vq_token`), so we whitelist it here
        # and pop it before sampling.
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
        """Sample one VQ token id per row.  ``logits`` is ``(B, vocab)``;
        returns ``(B,)`` so the caller can ``unsqueeze(-1)`` for the
        codebook embedding lookup (matching the legacy ``decode()``
        ``(B, 1)`` shape contract)."""
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

    def _image_size(self) -> int:
        """VQ spatial side length — from processor ``size`` or ``num_patches * 16``."""
        proc = self._processor
        if proc is not None:
            size = getattr(proc, "size", None)
            if isinstance(size, dict):
                side = size.get("height") or size.get("width")
                if side is not None:
                    return int(side)
            elif size is not None and getattr(size, "height", None) is not None:
                return int(size.height)
        return int(getattr(self._vq_cfg, "num_patches", 24)) * 16

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> torch.Tensor:
        """Raw images → VQVAE-normalised ``(N, 3, H, W)`` (``N=1`` zero row when empty)."""
        if not raw_images:
            return None

        if self._processor is None:
            raise RuntimeError(
                "JanusVqvae: samples carry images but no image processor is loaded. "
                "Assign `module._processor` via OmniModuleTrainer before training."
            )
        processed = self._processor(images=raw_images, return_tensors="pt")["pixel_values"]
        return processed.to(device=self.device, dtype=self.dtype)

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self) -> Dict[str, Any]:
        """Zero ``pixel_values`` for image-free micro-batches."""
        h = self._image_size()
        c = int(getattr(self._vq_cfg, "in_channels", 3))
        return {
            "pixel_values": torch.zeros(1, c, h, h, device=self.device, dtype=self.dtype),
        }
