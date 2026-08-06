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
counterpart (see :class:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase`) —
built straight off the checkpoint dir via :meth:`JanusSiglipPreprocessor.from_pretrained`,
with no model instance involved.
"""

from typing import Any, Optional

import torch
from transformers.models.janus.image_processing_janus import JanusImageProcessor

from ....processing import ModulePreprocessorBase
from ....utils.conversation import ConversationItem, iter_desired_items
from .configuration import JanusSiglipConfig


class JanusSiglipProcessor(JanusImageProcessor):
    """Alias — keeps the per-module asset name explicit in the V2 docs."""


_SOURCE = "janus_siglip"


class JanusSiglipPreprocessor(ModulePreprocessorBase):
    """Worker-side image normalize for the SigLIP (understanding) tower.

    Holds only the (picklable) HF image processor + a CPU zero-pixel template —
    never the model. Runs the same normalize as ``JanusSiglipModuleMixin._pixels_from_raw_images``
    but on **CPU** (bf16, to halve worker→main IPC); writes the pixel tensor back into
    each ``user``-image item. For each sample without a user image, appends a
    ``role="dummy"`` placeholder carrying the zero pixels, so the GPU
    forward never builds dummy inputs (the FSDP gradient anchor still runs there).
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

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> "JanusSiglipPreprocessor":
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        return cls(JanusSiglipProcessor.from_pretrained(module_path))

    def bind_dummy_inputs(self, config: JanusSiglipConfig, dtype: Optional[torch.dtype] = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)``, no live model."""
        cfg = config.vision_config
        self._dtype = dtype
        self._dummy_pixel_values = torch.zeros(cfg.num_channels, cfg.image_size, cfg.image_size, dtype=dtype)

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], roles=["user"]))
            if sample_image_items:
                # Real user images present → normalize them; no dummy needed. Tag with
                # the module source so forward_pre/post can pick up real images and
                # dummies uniformly (single ``source == _SOURCE`` filter).
                pixel_values = self._image_processor(
                    images=[it.value for it in sample_image_items], return_tensors="pt"
                )["pixel_values"]
                for it, px in zip(sample_image_items, pixel_values, strict=True):
                    it.value = px.to(dtype=self._dtype)
                    it.source = _SOURCE
            elif not inference:
                if self._dummy_pixel_values is None:
                    raise RuntimeError(
                        f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                        "before training use (pure inference never reaches this branch)."
                    )
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values,
                        role="dummy",
                        source=_SOURCE,
                    )
                )


__all__ = ["JanusSiglipProcessor", "JanusSiglipPreprocessor"]
