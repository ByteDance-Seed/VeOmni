"""Per-module processors for the Qwen3-VL vision tower.

Qwen3-VL keeps image and video preprocessing in **two distinct processors with
two distinct configs** (mirroring the upstream Qwen3-VL checkpoint). The vision
module loads **both**:

* **image** → :class:`transformers.Qwen2VLImageProcessor`, config
  ``preprocessor_config.json``.  Qwen3-VL *reuses* the Qwen2-VL image processor
  (the upstream ``preprocessor_config.json`` declares
  ``image_processor_type: Qwen2VLImageProcessorFast``, which resolves to
  ``Qwen2VLImageProcessor`` in transformers v5) — there is **no** Qwen3-VL
  specific image processor.
* **video** → :class:`transformers.Qwen3VLVideoProcessor`, config
  ``video_preprocessor_config.json`` (temporal patchify; not a Qwen2-VL class).

Aliased (not subclassed) to the canonical classes so the saved configs keep
their registered ``*_processor_type`` and stay Auto-loadable.
"""

from transformers import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


# Image: Qwen3-VL reuses the Qwen2-VL image processor.
Qwen3VLVisionImageProcessor = Qwen2VLImageProcessor
# Video: Qwen3-VL's own (non-Qwen2VL) video processor.
Qwen3VLVisionVideoProcessor = Qwen3VLVideoProcessor


__all__ = ["Qwen3VLVisionImageProcessor", "Qwen3VLVisionVideoProcessor"]
