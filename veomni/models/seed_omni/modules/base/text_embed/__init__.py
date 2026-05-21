"""Generic word-token embedding + LM head module.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`TextEmbedConfig`
* :mod:`.modeling`      — :class:`TextEmbed`

The folder name (``text_embed``) carries the namespace, so the inner
files use the short ``configuration.py`` / ``modeling.py`` names rather
than re-spelling the model in each filename.
"""

from .configuration import TextEmbedConfig
from .modeling import TextEmbed


__all__ = ["TextEmbedConfig", "TextEmbed"]
