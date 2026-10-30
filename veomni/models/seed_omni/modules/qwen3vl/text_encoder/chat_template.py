"""Qwen3-VL ChatML template (text + image + video) as readable Python.

Re-exported from :mod:`veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template`
so ``qwen3_text_encoder`` (in ``enable_image`` mode) and ``qwen3vl_text_encoder``
share the exact same class.
"""

from ...qwen3.text_encoder.chat_template import Qwen3VLChatMarkers, Qwen3VLChatTemplate


__all__ = ["Qwen3VLChatMarkers", "Qwen3VLChatTemplate"]
