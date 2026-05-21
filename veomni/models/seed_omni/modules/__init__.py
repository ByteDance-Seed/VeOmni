"""SeedOmni V2 module mixin registry.

Each entry maps a HuggingFace ``model_type`` string (the ``model_type``
field of each module's :class:`PretrainedConfig` subclass) to the
:class:`OmniModule` mixin class that backs it.  At trainer-build time the
flow is::

    cfg     = AutoConfig.from_pretrained(<weights_path>)   # reads model_type
    cls     = MODULE_MIXIN_REGISTRY[cfg.model_type]
    module  = cls.from_pretrained(<weights_path>)          # HF lifecycle
    # → wrapped by build_parallelize_model in OmniTrainer (Step 2)

The module classes also register themselves with HuggingFace
``AutoConfig`` / ``AutoModel`` so plain HF ``from_pretrained`` works
without VeOmni at all (useful for inspection / standalone tests).

File layout
-----------
``modules/<family>/<sub_module>/(configuration.py, modeling.py[,
processing.py])``.  Each sub-module gets its own folder; the folder name
carries the namespace so the inner files use short names rather than
re-spelling ``<family>_<sub_module>`` per file.  Cross-family
lightweight modules live under ``modules/base/<sub_module>/``.
"""

from typing import Dict, Type

from transformers import AutoConfig, AutoModel
from transformers.processing_utils import ProcessorMixin

from ..module import OmniModule
from .base import TextEmbed, TextEmbedConfig
from .janus import (
    JanusLlama,
    JanusLlamaConfig,
    JanusSiglip,
    JanusSiglipConfig,
    JanusSiglipProcessor,
    JanusTextEmbed,
    JanusTextEmbedConfig,
    JanusVqvae,
    JanusVqvaeConfig,
    JanusVqvaeProcessor,
)


# ── HF AutoConfig / AutoModel registration ────────────────────────────────────
# Idempotent: ``AutoConfig.register`` raises on duplicate model_type, so guard
# against re-running this module (pytest, IDE re-imports, ...).
def _register_hf(model_type: str, config_cls, model_cls) -> None:
    try:
        AutoConfig.register(model_type, config_cls)
    except ValueError:
        # Already registered — fine.
        pass
    try:
        AutoModel.register(config_cls, model_cls)
    except ValueError:
        pass


_register_hf("text_embed", TextEmbedConfig, TextEmbed)
_register_hf("janus_siglip", JanusSiglipConfig, JanusSiglip)
_register_hf("janus_vqvae", JanusVqvaeConfig, JanusVqvae)
_register_hf("janus_llama", JanusLlamaConfig, JanusLlama)
_register_hf("janus_text_embed", JanusTextEmbedConfig, JanusTextEmbed)


# ── V2 module mixin registry ──────────────────────────────────────────────────
MODULE_MIXIN_REGISTRY: Dict[str, Type[OmniModule]] = {
    # Cross-family modules
    "text_embed": TextEmbed,
    # Janus-1.3B family
    "janus_siglip": JanusSiglip,
    "janus_vqvae": JanusVqvae,
    "janus_llama": JanusLlama,
    "janus_text_embed": JanusTextEmbed,
}


# ── Per-module processor registry (asset class for save/load) ─────────────────
# Maps model_type → ProcessorMixin/ImageProcessor subclass.  Modules without
# a per-module processor (text_embed, janus_llama) are absent here — the
# checkpoint callback simply skips processor save/load for them.
MODULE_PROCESSOR_REGISTRY: Dict[str, Type[ProcessorMixin]] = {
    "janus_siglip": JanusSiglipProcessor,
    "janus_vqvae": JanusVqvaeProcessor,
}


__all__ = [
    "MODULE_MIXIN_REGISTRY",
    "MODULE_PROCESSOR_REGISTRY",
    # Generic
    "TextEmbed",
    "TextEmbedConfig",
    # Janus family
    "JanusSiglip",
    "JanusSiglipConfig",
    "JanusSiglipProcessor",
    "JanusVqvae",
    "JanusVqvaeConfig",
    "JanusVqvaeProcessor",
    "JanusLlama",
    "JanusLlamaConfig",
    "JanusTextEmbed",
    "JanusTextEmbedConfig",
]
