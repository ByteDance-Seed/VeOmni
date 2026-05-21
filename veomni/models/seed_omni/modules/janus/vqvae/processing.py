"""Per-module image processor for :class:`JanusVqvae`.

Janus' VQVAE expects 384x384 RGB normalised the same way the SigLIP tower
does (the original checkpoint reuses one shared
:class:`transformers.JanusImageProcessor` for both vision paths).  The V2
per-module checkpoint layout still keeps a copy of the processor next to
each module's weights so each sub-checkpoint is self-contained — at the
cost of a few KB of duplicated JSON.
"""

from transformers.models.janus.image_processing_janus import JanusImageProcessor


class JanusVqvaeProcessor(JanusImageProcessor):
    """Alias — keeps the per-module asset name explicit in the V2 docs."""


__all__ = ["JanusVqvaeProcessor"]
