"""Per-module image processor stub for :class:`JanusSiglip`.

The Janus SigLIP tower expects 384x384 RGB normalised to its own
mean/std.  Rather than re-deriving the processor here we delegate to
:class:`transformers.JanusImageProcessor` — that's the same class the
original ``JanusForConditionalGeneration`` ships, and it already knows the
right resize / normalise constants.

Exposed as :class:`JanusSiglipProcessor` so the per-module checkpoint
contains a ``preprocessor_config.json`` saved by HF's standard
``save_pretrained`` flow; loading back uses
``JanusSiglipProcessor.from_pretrained(<weights_path>)``.
"""

from transformers.models.janus.image_processing_janus import JanusImageProcessor


class JanusSiglipProcessor(JanusImageProcessor):
    """Alias — keeps the per-module asset name explicit in the V2 docs."""


__all__ = ["JanusSiglipProcessor"]
