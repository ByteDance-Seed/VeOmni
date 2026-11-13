"""Config for :class:`JanusVqvae`.

The whole VQVAE-side hyper-parameters travel under ``vq_config`` — that
mirrors the original :class:`transformers.JanusVQVAEConfig` schema so the
``scripts/seed_omni/convert_model.py`` extractor can dump the field
unchanged.

``freeze`` (default ``True``) matches the published Janus recipe — the
module freezes only its inner VQVAE codec (``vqmodel``) and keeps the
generation projection layers (``generation_embeddings``,
``generation_aligner``, ``generation_head``) trainable.  The freeze itself
is applied by :meth:`JanusVqvae.freeze_model` (the trainer calls it once
after build); this config only carries the knob.
"""

from typing import Any, Dict, Optional

from transformers.models.janus.configuration_janus import JanusVQVAEConfig as _hfvqconfig

from ...module_configuration_base import OmniModuleConfig


class JanusVqvaeConfig(OmniModuleConfig):
    """Top-level config for the Janus VQVAE + generation head."""

    model_type = "janus_vqvae"

    def __init__(
        self,
        vq_config: Optional[Dict[str, Any]] = None,
        freeze: bool = True,
        **kwargs,
    ):
        if vq_config:
            self.vq_config = _hfvqconfig(**vq_config)
        else:
            # HF's default ``num_patches=32`` expects a 1024-token grid, but the
            # Janus image processor emits 576 (24x24) tokens; decoding a 576-token
            # span with the default grid raises. Every in-repo producer (the family
            # converter and the HF-layout loader) passes ``vq_config``, so this only
            # affects a config that omits it.
            self.vq_config = _hfvqconfig(num_patches=24)
        # Module-level freeze knob.  ``JanusVqvae.freeze_model`` interprets it
        # as a *partial* freeze (only the inner ``vqmodel``); the generation
        # heads stay trainable.
        self.freeze = freeze
        super().__init__(**kwargs)
