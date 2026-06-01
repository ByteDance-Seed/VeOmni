"""
JanusSiglip — Janus' SigLIP vision tower + MLP aligner as one OmniModule.

Mixin form: ``class JanusSiglip(OmniModule, PreTrainedModel)``.  HuggingFace
``from_pretrained`` / ``save_pretrained`` work natively against
``<weights_path>/{config.json, model.safetensors[, preprocessor_config.
json]}``; the SeedOmni V2 graph runtime (:class:`OmniModel`) calls
:meth:`forward` / :meth:`pre_forward` / :meth:`post_forward` per
``OmniModule`` mixin protocol.

Connection outputs
------------------
``image_embeds``
    Float tensor of shape ``(batch, num_patches, llm_hidden_size)`` ready
    to be injected into the AR-LLM as understanding image embeddings.

Batch inputs (read from raw batch)
----------------------------------
``pixel_values``
    Float tensor of shape ``(B, 3, H, W)`` — single image per sample, or
    ``(B, N_images, 3, H, W)`` — multiple images per sample.  The latter
    shape is flattened by :meth:`pre_forward`.
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVisionConfig
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ....conversation import (
    ConversationPart,
    TrainConversation,
    collect_modality_values,
    is_raw_training_conversation,
)
from ....image_inputs import build_pixel_values_batch
from ....module import OmniModule
from .configuration import JanusSiglipConfig
from .processing import JanusSiglipProcessor


class JanusSiglip(OmniModule, PreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding.

    Multi-inherits :class:`OmniModule` (V2 mixin) and
    :class:`PreTrainedModel` so HF lifecycle methods work natively.
    Loaded from the ``model.vision_model`` and ``model.aligner`` sub-
    modules of the original ``JanusForConditionalGeneration`` checkpoint
    (split into a standalone folder by ``scripts/split_janus.py``).
    """

    config_class = JanusSiglipConfig
    processor_class = JanusSiglipProcessor
    base_model_prefix = "janus_siglip"
    main_input_name = "pixel_values"
    _no_split_modules = ["JanusVisionEncoderLayer"]

    def __init__(self, config: JanusSiglipConfig):
        super().__init__(config)

        vision_cfg = JanusVisionConfig(**config.vision_config) if config.vision_config else JanusVisionConfig()
        self.vision_model = JanusVisionModel._from_config(vision_cfg)
        self.aligner = JanusVisionAlignerMLP(vision_cfg)

        # Auto-populated by :meth:`OmniModule.from_pretrained` from
        # ``<weights_path>/preprocessor_config.json`` when loading via
        # the HF lifecycle.  Stays ``None`` in trainer-side meta-init
        # paths (training feeds pre-tensorised ``pixel_values`` via the
        # data collator) and when no processor JSON ships next to the
        # weights.
        self._processor: Optional[Any] = None

        self.post_init()

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """Extract + tensorise understanding images, then SP/shape-normalise.

        Training (raw ``conversation_list``): pull every ``image`` turn's raw
        ``(C, H, W)`` uint8 tensor, process it to the SigLIP-normalised
        ``(B, 3, H, W)`` batch (zero placeholder for image-free samples) so
        the encoder runs on every micro-batch (FSDP alignment).  Inference /
        pre-tensorised paths pass ``pixel_values`` straight through.

        Then flatten ``(B, N_images, C, H, W)`` → ``(B*N_images, C, H, W)``
        (the original ``(B, N_images)`` shape is stashed in
        ``_pv_batch_n_images`` so :meth:`forward` can reshape back).
        """
        if pixel_values is None:
            conversation = kwargs.get("conversation_list")
            if is_raw_training_conversation(conversation):
                pixel_values = self._extract_und_pixel_values(conversation)
                # Keep ``conversation_list`` in kwargs: :meth:`forward` wraps the
                # raw conversation into a :class:`TrainConversation` carrier (the
                # single object that flows down the graph) and fills its
                # ``und_embeds``.

        if pixel_values is not None and pixel_values.ndim == 5:
            b, n = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(b * n, *pixel_values.shape[2:])
            kwargs["_pv_batch_n_images"] = (b, n)
        return dict(pixel_values=pixel_values, **kwargs)

    def forward(self, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        """Encode understanding image patches to LLM-space embeddings.

        Training (raw ``conversation_list`` present): build the
        :class:`TrainConversation` carrier and stash the full-batch
        understanding embeds (``(B, P, D)``) on it as ``und_embeds`` —
        ``JanusTextEncoder`` later slices per-sample rows into image segments
        and the backbone uses the whole tensor as an FSDP grad-sync anchor.
        Returns ``{"conversation_list": carrier}``.

        Legacy / pre-tensorised path (``pixel_values`` passed directly, no raw
        conversation): returns ``{"image_embeds": (B, P, D)}`` unchanged.

        Returns ``{}`` for text-only batches (``pixel_values is None``) — the
        *inference* fast path; in training the trainer fills
        :meth:`dummy_inputs` so this branch is never reached.
        """
        conversation = kwargs.get("conversation_list")
        if pixel_values is None:
            return {}

        vision_out = self.vision_model(pixel_values, return_dict=True)
        feats = vision_out.last_hidden_state
        image_embeds = self.aligner(feats)

        b_n = kwargs.pop("_pv_batch_n_images", None)
        if b_n is not None:
            b, n = b_n
            p = image_embeds.size(1)
            image_embeds = image_embeds.reshape(b, n * p, image_embeds.size(2))

        if is_raw_training_conversation(conversation):
            return {"conversation_list": TrainConversation(raw=conversation, und_embeds=image_embeds)}
        return {"image_embeds": image_embeds}

    def _extract_und_pixel_values(self, conversation_list: List[List[dict]]) -> torch.Tensor:
        """Raw ``image`` turns → SigLIP-normalised ``(B, 3, H, W)`` batch."""
        cfg = self.config.vision_config or {}
        image_size = cfg.get("image_size", 384) if isinstance(cfg, dict) else getattr(cfg, "image_size", 384)
        num_channels = cfg.get("num_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "num_channels", 3)
        per_sample = collect_modality_values(conversation_list, ("image",))
        return build_pixel_values_batch(
            per_sample,
            processor=self._processor,
            image_size=image_size,
            num_channels=num_channels,
            device=self._param_device(),
            dtype=self._param_dtype(),
        )

    # ── Inference (conversation-list) ─────────────────────────────────────────

    def generate(
        self,
        *,
        conversation_list: Optional[List[ConversationPart]] = None,
        past_key_values: Optional[Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Encode every ``image_und`` part that doesn't yet carry an ``inputs_embeds``.

        Inference-only entry — training still goes through
        :meth:`forward` (input_ids + masked_scatter contract).  Called once
        on the prompt pass; subsequent FSM iterations short-circuit
        because ``past_key_values is not None`` (no new images get
        injected mid-AR).

        Each ``image_und`` part must either carry a 3D ``pixel_values``
        tensor ``(C, H, W)`` already, or carry a raw PIL ``image`` so we
        can tensorise on the fly via ``self._processor`` — auto-loaded
        from the same checkpoint folder by
        :meth:`OmniModule.from_pretrained`.  The aligner-projected output
        is written back into the same part so downstream
        :meth:`JanusLlama.generate` can concat it next to the text
        embeddings.
        """
        if conversation_list is None or past_key_values is not None:
            return {"conversation_list": conversation_list} if conversation_list is not None else {}

        device = self._param_device()
        dtype = self._param_dtype()
        for part in conversation_list:
            if part.kind != "image_und" or part.inputs_embeds is not None:
                continue
            pv = part.pixel_values
            if pv is None:
                # Fall back to raw PIL → tensor via the wired processor.
                # Cache the result back on the part so a re-run / trace sees
                # the same numerics, and so the FSM's bookkeeping (which may
                # serialise the conversation) stays consistent.
                if part.image is None:
                    continue
                if self._processor is None:
                    raise RuntimeError(
                        "JanusSiglip.generate: image_und part has no `pixel_values` and no "
                        "processor was loaded.  Either pre-tensorise the part or ensure "
                        "`preprocessor_config.json` ships next to the weights checkpoint so "
                        "OmniModule.from_pretrained can auto-load it."
                    )
                out = self._processor(images=[part.image], return_tensors="pt")
                pv = out["pixel_values"]
                if pv.dim() == 4 and pv.size(0) == 1:
                    pv = pv.squeeze(0)
                part.pixel_values = pv
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)
            pv = pv.to(device=device, dtype=dtype, non_blocking=True)
            vision_out = self.vision_model(pv, return_dict=True)
            part.inputs_embeds = self.aligner(vision_out.last_hidden_state)
        return {"conversation_list": conversation_list}

    # ── Internal device / dtype helpers ───────────────────────────────────────

    def _param_device(self) -> torch.device:
        return next(self.parameters()).device

    def _param_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> Dict[str, Any]:
        """Return zero ``pixel_values`` so the full vision path forwards.

        Used by the trainer for micro-batches that have no understanding
        images — keeps the FSDP graph aligned across DP/SP ranks.  The
        zero output flows through the LLM's ``masked_scatter`` as a
        no-op (mask is all-False) and contributes no gradient; the LLM's
        ``pre_forward`` adds an ``image_embeds.sum() * 0.0`` anchor so
        the upstream params still receive a (zero) gradient and FSDP
        sync stays consistent.
        """
        cfg = self.config.vision_config or {}
        h = cfg.get("image_size", 384) if isinstance(cfg, dict) else getattr(cfg, "image_size", 384)
        c = cfg.get("num_channels", 3) if isinstance(cfg, dict) else getattr(cfg, "num_channels", 3)
        return {
            "pixel_values": torch.zeros(batch_size, c, h, h, device=device, dtype=dtype),
        }
