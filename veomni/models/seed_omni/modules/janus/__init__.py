"""Janus-1.3B OmniModule mixins.

Splits the monolithic ``JanusForConditionalGeneration`` into composable
sub-modules that match the SeedOmni V2 graph runtime (:mod:`veomni.models.
seed_omni`).  Each sub-module lives in its own folder under
``janus/<sub_module>/`` and contains short-named files
(``configuration.py``, ``modeling.py``, optional ``processing.py``) —
the folder name carries the namespace, so the file names don't repeat
``janus_<sub_module>`` again.

Resolve concrete classes at runtime via
:data:`~veomni.models.seed_omni.modules.OMNI_MODEL_REGISTRY` /
:data:`~veomni.models.seed_omni.modules.OMNI_CONFIG_REGISTRY` — do not
import modeling modules directly from this package.
"""

from . import llama, siglip, text_encoder, vqvae  # noqa: F401
