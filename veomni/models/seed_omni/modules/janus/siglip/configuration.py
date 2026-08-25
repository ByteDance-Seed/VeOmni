"""Config for :class:`JanusSiglip` — Janus' SigLIP vision tower + aligner.

The ``model_type`` literal here is the lookup key for
``OMNI_CONFIG_REGISTRY`` / ``OMNI_MODEL_REGISTRY``.

The whole vision-side hyper-parameters travel under ``vision_config`` —
that mirrors the original :class:`transformers.JanusVisionConfig` schema so
the ``scripts/convert_model.py`` extractor can dump the field unchanged.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from transformers.models.janus.configuration_janus import JanusVisionConfig


class JanusSiglipConfig(PretrainedConfig):
    """Top-level config for the Janus SigLIP encoder + aligner."""

    model_type = "janus_siglip"

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.vision_config = JanusVisionConfig(**vision_config) if vision_config else JanusVisionConfig()
        super().__init__(**kwargs)
