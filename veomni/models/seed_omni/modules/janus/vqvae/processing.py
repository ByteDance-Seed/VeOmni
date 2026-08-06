"""Per-module image processor + worker-side preprocessor for :class:`JanusVqvae`.

The Janus VQVAE is an asymmetric processor: the *encode* side accepts
PIL → resize → rescale → CLIP-normalise (inherited unchanged from
:class:`transformers.JanusImageProcessor`), but the *decode* side
reconstructs into ``[-1, 1]`` via a plain ``2x - 1`` mapping — NOT the
CLIP-normalised space the encoder lives in.

We therefore inherit the upstream preprocess pipeline as-is and override
:meth:`postprocess` with the VQVAE-decoder convention so callers can chain
``processor.postprocess(decoded_tensor)`` straight into ``img.save(path)``.

Exposed as :class:`JanusVqvaeProcessor` so the per-module checkpoint
ships a ``preprocessor_config.json`` saved by HF's standard
``save_pretrained``; loading back goes through
:meth:`JanusVqvaePreprocessor.from_pretrained`.

:class:`JanusVqvaePreprocessor` is the picklable, weight-free CPU worker
counterpart (see :class:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase`) —
built straight off the checkpoint dir, with no model instance involved.
"""

from typing import Any, List, Optional, Union

import torch
from PIL import Image
from transformers.models.janus.image_processing_janus import JanusImageProcessor

from ....processing import ModulePreprocessorBase
from ....utils.conversation import ConversationItem, iter_desired_items
from .configuration import JanusVqvaeConfig


class JanusVqvaeProcessor(JanusImageProcessor):
    """Janus VQVAE image processor — encode preprocess + decode postprocess."""

    num_image_tokens: int = 576

    def postprocess(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        **_,
    ) -> List[Image.Image]:
        """Convert VQVAE-decoded ``[-1, 1]`` float tensors → list of PIL images.

        Accepts a single ``(H, W, 3)`` / ``(1, H, W, 3)`` / ``(B, H, W, 3)``
        tensor or a list of such tensors.  Returns a flat list of PIL
        images ready for ``img.save(path)``.

        ``(x + 1) / 2 → ×255 → clamp → round → uint8 → PIL.fromarray`` —
        the ``.round()`` before the uint8 cast is load-bearing: uint8
        TRUNCATES toward zero (127.5 → 127, not 128), so dropping the
        round would drift the saved PNG ±1 LSB from the HF baseline.
        """
        if isinstance(images, torch.Tensor):
            images = [images]

        out: List[Image.Image] = []
        for tensor in images:
            img = tensor.detach().to(dtype=torch.float32, device="cpu")
            if img.dim() == 4:
                for b in range(img.size(0)):
                    out.append(_to_pil(img[b]))
            elif img.dim() == 3:
                out.append(_to_pil(img))
            else:
                raise ValueError(
                    f"JanusVqvaeProcessor.postprocess: expected (H,W,3) / (1,H,W,3) / (B,H,W,3), "
                    f"got {tuple(img.shape)}."
                )
        return out


def _to_pil(img: torch.Tensor) -> Image.Image:
    """``[-1, 1]`` float ``(H, W, 3)`` → PIL.Image via inverse Janus ``2x - 1``."""
    if img.dim() != 3 or img.size(-1) != 3:
        raise ValueError(f"_to_pil: expected (H, W, 3), got {tuple(img.shape)}.")
    img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
    arr = (img * 255.0).clamp(0, 255).round().to(torch.uint8).numpy()
    return Image.fromarray(arr)


_SOURCE = "janus_vqvae"


class JanusVqvaePreprocessor(ModulePreprocessorBase):
    """Worker-side image normalize for the VQVAE (generation) codec.

    Holds only the (picklable) VQVAE image processor + a CPU zero-pixel template
    — never the model. Runs the HF image processor on **CPU** (bf16, to halve
    IPC); writes the pixel tensor back into each ``assistant``-image item.
    For each sample without an assistant image, appends a ``role="dummy"``
    placeholder carrying the zero pixels (the codec + generation heads
    still run on it in the GPU forward for the FSDP gradient anchor).
    """

    def __init__(
        self,
        image_processor: JanusVqvaeProcessor,
        dtype: Optional[torch.dtype] = None,
        dummy_pixel_values: Optional[torch.Tensor] = None,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (C, H, W), model dtype

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> "JanusVqvaePreprocessor":
        """Build straight from the checkpoint dir — no model instance needed."""
        del kwargs
        return cls(JanusVqvaeProcessor.from_pretrained(module_path))

    def bind_dummy_inputs(self, config: JanusVqvaeConfig, dtype: Optional[torch.dtype] = None) -> None:
        """Training-only FSDP-anchor dummy — pure ``(config, dtype)`` + the already
        loaded image processor's own ``size`` — no live model needed."""
        cfg = config.vq_config
        size = self._image_processor.size
        height = size.get("height")
        width = size.get("width")
        self._dtype = dtype
        self._dummy_pixel_values = torch.zeros(cfg.in_channels, height, width, dtype=dtype)

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], roles=["assistant"]))
            if sample_image_items:
                # Real assistant images present → normalize them; no dummy needed.
                # Tag with the module source so the decode path can pick up real gen
                # images and dummies uniformly (single ``source == _SOURCE`` filter).
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


__all__ = ["JanusVqvaeProcessor", "JanusVqvaePreprocessor"]
