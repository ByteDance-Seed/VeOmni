"""Qwen3-VL vision tower OmniModule."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY, OMNI_PROCESSOR_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3vl_vision")
def register_qwen3vl_vision_config():
    from .configuration import Qwen3VLVisionEncoderConfig

    return Qwen3VLVisionEncoderConfig


@OMNI_MODEL_REGISTRY.register("qwen3vl_vision")
def register_qwen3vl_vision_model():
    from .modeling import Qwen3VLVisionEncoder

    return Qwen3VLVisionEncoder


@OMNI_PROCESSOR_REGISTRY.register("qwen3vl_vision")
def register_qwen3vl_vision_processor():
    """Qwen3-VL's vision tower needs TWO processors (image + video), so this
    returns a ``{"image": ..., "video": ...}`` dict rather than a single class.

    Note this registry is only a convenience for convert-time tooling; at
    train / inference time both processors are loaded automatically from the
    ``image_processor_class`` / ``video_processor_class`` attributes on
    :class:`Qwen3VLVisionEncoder` (see ``OmniModule.from_pretrained``), exposed
    as ``self._image_processor`` / ``self._video_processor``.
    """
    from .processing import Qwen3VLVisionImageProcessor, Qwen3VLVisionVideoProcessor

    return {"image": Qwen3VLVisionImageProcessor, "video": Qwen3VLVisionVideoProcessor}
