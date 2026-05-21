"""Janus-specific :class:`TextEmbed` with image boundary-token emitters.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusTextEmbedConfig`
* :mod:`.modeling`      — :class:`JanusTextEmbed`
"""

from .configuration import JanusTextEmbedConfig
from .modeling import JanusTextEmbed


__all__ = ["JanusTextEmbedConfig", "JanusTextEmbed"]
