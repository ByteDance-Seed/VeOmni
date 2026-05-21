"""Janus LLaMA backbone OmniModule.

Sub-package layout (per :mod:`veomni.models.seed_omni.modules`):

* :mod:`.configuration` — :class:`JanusLlamaConfig`
* :mod:`.modeling`      — :class:`JanusLlama`

(No per-module processor: the AR backbone consumes hidden tensors only.)
"""

from .configuration import JanusLlamaConfig
from .modeling import JanusLlama


__all__ = ["JanusLlamaConfig", "JanusLlama"]
