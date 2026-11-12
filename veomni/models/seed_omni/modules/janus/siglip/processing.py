"""Per-module image processor + worker-side preprocessor for :class:`JanusSiglip`.

The Janus SigLIP tower expects 384x384 RGB normalised to its own
mean/std.  Rather than re-deriving the processor here we delegate to
:class:`transformers.JanusImageProcessor` — that's the same class the
original ``JanusForConditionalGeneration`` ships, and it already knows the
right resize / normalise constants.

Exposed as :class:`JanusSiglipProcessor` so the per-module checkpoint
contains a ``preprocessor_config.json`` saved by HF's standard
``save_pretrained`` flow; loading back uses
``JanusSiglipProcessor.from_pretrained(<weights_path>)``.

:class:`JanusSiglipPreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.modules.module_processing_base.ModulePreprocessorBase`) —
built straight off the checkpoint dir via :meth:`JanusSiglipPreprocessor.from_pretrained`,
with no model instance involved.
"""

from typing import Any, Optional

import torch
from transformers.models.janus.image_processing_janus import JanusImageProcessor

from ....utils.conversation import _IMG_TAG_KEY, _OMNI_TOKEN_LEN, ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from .configuration import JanusSiglipConfig


class JanusSiglipProcessor(JanusImageProcessor):
    """Alias — keeps the per-module asset name explicit in the SeedOmni docs."""


class JanusSiglipPreprocessor(ModulePreprocessorBase):
    """Worker-side image normalize for the SigLIP (understanding) tower.

    Holds only the (picklable) HF image processor + a CPU zero-pixel template —
    never the model. Runs the same normalize as ``JanusSiglip._pixels_from_raw_images``
    but on **CPU** (bf16, to halve worker→main IPC); writes the pixel tensor back into
    each ``user``-image item. When a whole micro-batch has no user image, appends one
    ``is_dummy`` user-image placeholder carrying the zero pixels, so the GPU forward never
    builds dummy inputs (the FSDP gradient anchor still runs there).
    """

    def __init__(
        self,
        image_processor: Any,
        dtype: Optional[torch.dtype] = None,
        dummy_pixel_values: Optional[torch.Tensor] = None,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (C, H, W), model dtype

    @property
    def _processor_size(self) -> tuple[int, int]:
        """``(height, width)`` the image processor emits at.

        HF allows ``size`` to be an ``int`` (square) or a dict with
        ``height``/``width`` or ``shortest_edge``; a caller passing a bare int is
        legal, so both spellings are handled here.
        """
        size = self._image_processor.size
        if isinstance(size, int):
            return size, size
        height = size.get("height") or size.get("shortest_edge")
        width = size.get("width") or size.get("shortest_edge")
        return int(height), int(width)

    @property
    def _num_patch_tokens(self) -> int:
        """Patch embeddings one SigLIP image contributes, i.e. its backbone rows.

        ``dyn_bsz`` counts these per image item; the tower emits one row per
        patch and the aligner does not change the row count.
        """
        height, width = self._processor_size
        # A processor always exposes ``patch_size``; a lightweight stand-in
        # (a test double) may not, and one token per pixel is the honest default.
        patch = int(getattr(self._image_processor, "patch_size", 1))
        return (height // patch) * (width // patch)

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> "JanusSiglipPreprocessor":
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        return cls(JanusSiglipProcessor.from_pretrained(module_path))

    def bind_dummy_inputs(self, config: JanusSiglipConfig, dtype: Optional[torch.dtype] = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)``, no live model.

        The shape must match what ``self._image_processor`` produces, because a
        mixed real/dummy batch is ``torch.stack``ed together. A ``vision_config``
        override that moves ``image_size`` away from the checkpoint's
        ``preprocessor_config.json`` would otherwise pair real 384px images with a
        differently sized dummy and fail the stack, so that mismatch is rejected
        here rather than at the first step.
        """
        cfg = config.vision_config
        proc_h, proc_w = self._processor_size
        if (proc_h, proc_w) != (int(cfg.image_size), int(cfg.image_size)):
            raise ValueError(
                f"{type(self).__name__}: vision_config.image_size={cfg.image_size} does not match the image "
                f"processor's {(proc_h, proc_w)} from preprocessor_config.json. Override both together, or "
                "drop the vision_config override — a mixed real/dummy batch would fail to stack."
            )
        self._dtype = dtype
        self._dummy_pixel_values = torch.zeros(cfg.num_channels, proc_h, proc_w, dtype=dtype)

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list:
            sample_image_items = [
                it for it in iter_desired_items([sample], types=["image"], roles=["user"]) if not it.is_dummy
            ]
            if sample_image_items:
                # Real user images present → normalize them; no dummy needed.
                # forward_pre/post re-select them, and the dummy, by the same
                # image + user filter.
                pixel_values = self._image_processor(
                    images=[it.value for it in sample_image_items], return_tensors="pt"
                )["pixel_values"]
                for it, px in zip(sample_image_items, pixel_values, strict=True):
                    it.value = px.to(dtype=self._dtype)
                    # ``dyn_bsz`` counts backbone rows before collation, so each
                    # image item must state how many tokens it contributes:
                    # SigLIP turns one image into ``patches`` patch embeddings.
                    it.meta[_OMNI_TOKEN_LEN] = self._num_patch_tokens
                continue
            if inference:
                continue
            if self._dummy_pixel_values is None:
                raise RuntimeError(
                    f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                    "before training use (pure inference never reaches this branch)."
                )
            # Per sample: this module's forward still runs on a step with no real
            # media so its FSDP collectives stay in step with the other ranks.
            sample.append(
                ConversationItem(
                    type="image",
                    value=self._dummy_pixel_values,
                    role="user",
                    is_dummy=True,
                    meta={_IMG_TAG_KEY: "und", _OMNI_TOKEN_LEN: self._num_patch_tokens},
                )
            )


__all__ = ["JanusSiglipProcessor", "JanusSiglipPreprocessor"]
