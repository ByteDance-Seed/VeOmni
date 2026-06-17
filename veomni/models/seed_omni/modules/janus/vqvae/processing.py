"""Per-module image processor for :class:`JanusVqvae`.

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
:meth:`OmniModule.from_pretrained` via the ``image_processor_class`` class
attribute on :class:`JanusVqvae`.
"""

from typing import List, Union

import torch
from PIL import Image
from transformers.models.janus.image_processing_janus import JanusImageProcessor


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


__all__ = ["JanusVqvaeProcessor"]
