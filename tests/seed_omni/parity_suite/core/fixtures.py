"""Model-agnostic deterministic fixtures for SeedOmni parity tests."""

from __future__ import annotations

import numpy as np
from PIL import Image


# Image fixtures ---------------------------------------------------------------


def make_reference_image(width: int, height: int) -> Image.Image:
    """Create the deterministic reference-side input image used by parity tests."""

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(rgb)


__all__ = ["make_reference_image"]
