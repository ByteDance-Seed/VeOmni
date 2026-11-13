"""Split a DeepSeek-format Janus checkpoint into SeedOmni modules.

DeepSeek publishes ``deepseek-ai/Janus-1.3B`` under its own tensor names
(``gen_vision_model``, ``language_model.lm_head``, ...) and
``model_type: multi_modality``, which neither ``transformers`` nor
:data:`~.hf_layout.JANUS_HF_LAYOUT` can read. This converter rewrites that format
to the standard HF one and then hands off to the layout, so a DeepSeek source and
an already-converted HF root produce identical modules.

A standard HF checkpoint needs no converter: :func:`~veomni.models.seed_omni.utils.hf_layout.convert_with_hf_layout`
splits it straight through ``JANUS_HF_LAYOUT``, exactly as the Qwen3 family does.
"""

from __future__ import annotations

import tempfile
from typing import Any

from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY

from .convert_janus_weight_to_hf import convert_model as convert_janus_to_hf


def convert_deepseek_janus_checkpoint(model_path: str, **kwargs: Any) -> dict[str, Any]:
    """Rewrite a DeepSeek-layout source to HF, then split it through the layout."""
    from veomni.models.seed_omni.utils.hf_layout import convert_with_hf_layout

    with tempfile.TemporaryDirectory(prefix="janus-hf-") as hf_model_path:
        convert_janus_to_hf(local_dir=model_path, output_dir=hf_model_path)
        return convert_with_hf_layout(hf_model_path, **kwargs)


@OMNI_CONVERT_REGISTRY.register("multi_modality")  # deepseek ckpt
def _register_janus_convert():
    return convert_deepseek_janus_checkpoint


__all__ = ["convert_deepseek_janus_checkpoint"]
