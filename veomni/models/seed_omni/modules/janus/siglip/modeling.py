"""
JanusSiglip — Janus' SigLIP vision tower + MLP aligner as one OmniModule.

Mixin form: ``class JanusSiglip(OmniModule, PreTrainedModel)``.  The vision
stack reuses HuggingFace :class:`~transformers.JanusVisionModel` and
:class:`~transformers.JanusVisionAlignerMLP` — the same pair wired inside
``JanusForConditionalGeneration`` (``aligner(vision_model(pixel_values))``).

Call-site split (V2)
--------------------
* :meth:`pre_forward` — pull raw ``(C,H,W)`` images from ``conversation_list``
  → ``pixel_values``; stash the carrier for :meth:`post_forward`.
* :meth:`forward` — pure encoder: ``pixel_values`` → ``image_embeds`` (HF path).
* :meth:`post_forward` — write ``image_embeds`` back onto ``conversation_list``
  ``type="image"`` items in place.
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel
from transformers.models.janus.configuration_janus import JanusVisionConfig
from transformers.models.janus.modeling_janus import JanusVisionAlignerMLP, JanusVisionModel

from ....conversation import (
    ConversationItem,
    collect_modality_batch,
    item_role,
    iter_modality_items,
)
from ....module import OmniModule
from .configuration import JanusSiglipConfig
from .processing import JanusSiglipProcessor


class JanusSiglip(OmniModule, PreTrainedModel):
    """SigLIP vision tower + MLP aligner for image understanding.

    Composes HF :class:`JanusVisionModel` + :class:`JanusVisionAlignerMLP`
    (weights split from ``JanusForConditionalGeneration`` by
    ``scripts/split_janus.py``).
    """

    config_class = JanusSiglipConfig
    processor_class = JanusSiglipProcessor
    base_model_prefix = "janus_siglip"
    main_input_name = "pixel_values"
    _no_split_modules = ["JanusVisionEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusSiglipConfig):
        super().__init__(config)
        self.config = config

        vision_cfg = JanusVisionConfig(**config.vision_config) if config.vision_config else JanusVisionConfig()
        self.vision_model = JanusVisionModel(vision_cfg)
        self.aligner = JanusVisionAlignerMLP(vision_cfg)

        self._processor: Optional[Any] = None
        self._conversation_carrier: Any = None

        self.post_init()

    # ── JanusSiglip Main Function ───────────────────────────────────────────────────
    def _encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``aligner(vision_model(pixels))`` — mirrors Janus understanding path."""
        vision_out = self.vision_model(pixel_values, return_dict=True)
        return self.aligner(vision_out.last_hidden_state)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """HF-style vision encode: ``pixel_values`` → ``image_embeds``."""
        if pixel_values is None:
            return {"image_embeds": self._encode_pixel_values(**self.dummy_inputs()), "is_dummy": True}
        return {"image_embeds": self._encode_pixel_values(pixel_values)}

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        """Extract ``pixel_values`` from the carrier; stash ``conversation_list``."""
        assert method == "forward"
        self._conversation_carrier = conversation_list
        pixel_values = self._pixels_from_raw_images(
            collect_modality_batch(conversation_list, ["image"], roles=["user"])
        )
        return {"pixel_values": pixel_values}

    def post_forward(
        self,
        method: str,
        image_embeds: torch.Tensor,  # n_image_cross_all_batch, 576, 2048
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        """Write ``image_embeds`` onto the stashed ``conversation_list`` carrier."""
        assert method == "forward"
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        if is_dummy:
            assert image_embeds.shape[0] == 1
            image_embeds = image_embeds.squeeze(0)
            for sample in conversation:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=image_embeds,
                        role="dummy",
                        meta={"source": "janus_siglip"},
                    )
                )
        else:
            items = list(iter_modality_items(conversation, ["image"], roles=["user"]))
            for item, emb in zip(items, image_embeds, strict=True):
                item.value = emb
        return {"conversation_list": conversation}

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> torch.Tensor:
        """Raw images (uint8 ``(C,H,W)`` or PIL) → SigLIP-normalised ``(N, 3, H, W)``."""
        if not raw_images:
            return None
        return self._processor(images=raw_images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Encode every user ``image`` part that is not yet embedded."""

        pending = [
            part
            for part in conversation_list
            if part.type == "image_output"  # newly generated images
        ]
        if not pending:
            pending = [part for part in conversation_list if part.type == "image" and item_role(part) == "user"]

        if not pending:
            return {"conversation_list": conversation_list}

        embeds = self._encode_pixel_values(self._pixels_from_raw_images([part.value for part in pending]))
        for part, emb in zip(pending, embeds, strict=True):
            part.value = emb if emb.dim() == 2 else emb.squeeze(0)
            if part.type == "image_output":
                part.type = "image"
                assert part.role == "assistant"  # debug check

        return {"conversation_list": conversation_list}

    # ── Training-side dummy forward ────────────────────────────────────────────

    def dummy_inputs(self) -> Dict[str, Any]:
        """Zero ``pixel_values`` for image-free micro-batches (FSDP alignment)."""
        cfg = self.config.vision_config or {}
        h = cfg["image_size"]
        c = cfg["num_channels"]
        return {
            "pixel_values": torch.zeros(1, c, h, h, device=self.device, dtype=self.dtype),
        }
