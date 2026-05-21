"""Config for :class:`JanusVqvae`.

The whole VQVAE-side hyper-parameters travel under ``vq_config`` — that
mirrors the original :class:`transformers.JanusVQVAEConfig` schema so the
``scripts/split_janus.py`` extractor can dump the field unchanged.

``freeze_vqvae`` (default ``True``) matches the published Janus recipe —
only the generation projection layers (``generation_embeddings``,
``generation_aligner``, ``generation_head``) are trained.
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class JanusVqvaeConfig(PretrainedConfig):
    """Top-level config for the Janus VQVAE + generation head."""

    model_type = "janus_vqvae"

    def __init__(
        self,
        vq_config: Optional[Dict[str, Any]] = None,
        freeze_vqvae: bool = True,
        **kwargs,
    ):
        self.vq_config = vq_config or {}
        self.freeze_vqvae = freeze_vqvae
        super().__init__(**kwargs)
