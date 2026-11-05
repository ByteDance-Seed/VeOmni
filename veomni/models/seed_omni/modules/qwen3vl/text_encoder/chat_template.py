"""Qwen3-VL ChatML template (text + image + video) as readable Python.

Re-exported from :mod:`veomni.models.seed_omni.modules.qwen3.text_encoder.chat_template`
so ``qwen3_text_encoder`` (in ``enable_image`` mode) and ``qwen3vl_text_encoder``
share the exact same class.

Mirrors the upstream ``chat_template.json``:

* each turn is wrapped in ``<|im_start|>{role}\\n … <|im_end|>\\n``;
* image / video become ``<|vision_start|><|image_pad|><|vision_end|>`` /
  ``<|vision_start|><|video_pad|><|vision_end|>`` — in the SeedOmni segment model the
  ``<|*_pad|>`` run is *not* tokenized; the sibling ``image`` / ``video`` item
  already carries the merged vision tokens, so the template emits
  ``<|vision_start|>`` text · the media item · ``<|vision_end|>`` text.

Qwen3-VL has no audio modality (audio-in-video is an Omni feature — see
``docs/seed_omni/design/media.md``, design-only).
"""

from ...qwen3.text_encoder.chat_template import Qwen3VLChatMarkers, Qwen3VLChatTemplate


__all__ = ["Qwen3VLChatMarkers", "Qwen3VLChatTemplate"]
