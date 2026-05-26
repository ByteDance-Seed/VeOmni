"""Config for :class:`JanusSiglip` — Janus' SigLIP vision tower + aligner.

The ``model_type`` literal here is the lookup key for
``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY``.

The whole vision-side hyper-parameters travel under ``vision_config`` —
that mirrors the original :class:`transformers.JanusVisionConfig` schema so
the ``scripts/split_janus.py`` extractor can dump the field unchanged.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class JanusSiglipConfig(PretrainedConfig):
    """Top-level config for the Janus SigLIP encoder + aligner."""

    model_type = "janus_siglip"

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.vision_config = vision_config or {}
        super().__init__(**kwargs)
