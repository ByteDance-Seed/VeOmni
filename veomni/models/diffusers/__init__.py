import diffusers

from . import ltx2_3, minimax_h3, qwen_image, wan_t2v


__all__ = ["ltx2_3", "minimax_h3", "qwen_image", "wan_t2v"]

_QWEN_IMAGE21_COMPONENTS = (
    "AutoencoderKLQwenImage21",
    "QwenImage21Pipeline",
    "QwenImage21Transformer2DModel",
)

if all(hasattr(diffusers, component) for component in _QWEN_IMAGE21_COMPONENTS):
    from . import qwen_image21

    __all__.append("qwen_image21")
