"""Config for :class:`JanusVqvae`.

The whole VQVAE-side hyper-parameters travel under ``vq_config`` — that
mirrors the original :class:`transformers.JanusVQVAEConfig` schema so the
``scripts/split_janus.py`` extractor can dump the field unchanged.

``freeze`` (default ``True``) matches the published Janus recipe — the
module freezes only its inner VQVAE codec (``vqmodel``) and keeps the
generation projection layers (``generation_embeddings``,
``generation_aligner``, ``generation_head``) trainable.  The freeze itself
is applied by :meth:`JanusVqvae.freeze_model` (the trainer calls it once
after build); this config only carries the knob.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from transformers.models.janus.configuration_janus import JanusVQVAEConfig as _hfvqconfig


class JanusVqvaeConfig(PretrainedConfig):
    """Top-level config for the Janus VQVAE + generation head."""

    model_type = "janus_vqvae"

    def __init__(
        self,
        vq_config: Optional[Dict[str, Any]] = None,
        freeze: bool = True,
        **kwargs,
    ):
        self.vq_config = _hfvqconfig(**vq_config) if vq_config else _hfvqconfig()
        # Module-level freeze knob.  ``JanusVqvae.freeze_model`` interprets it
        # as a *partial* freeze (only the inner ``vqmodel``); the generation
        # heads stay trainable.
        self.freeze = freeze
        super().__init__(**kwargs)
