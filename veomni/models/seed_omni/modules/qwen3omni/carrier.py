"""Carrier vocabulary shared by the Qwen3-Omni modules.

Two kinds of row on the carrier are *not* thinker input, and every module that
walks a sample skips both:

* **dummy rows** — FSDP anchors (``is_dummy=True``);
* **the speech target** — ``role="assistant"`` audio. This is the waveform the
  talker is taught to say, i.e. a *label*. The thinker has no speech output and
  never consumes its own voice, so the row belongs to the codec alone. It stays
  on the carrier (that is how the codec finds it and stashes its RVQ codes) but
  it must not enter the backbone's packed sequence: it holds a raw 1-D waveform,
  so the packer would read its sample count as a row count, and were it encoded
  instead the target speech would sit inside the talker's own prompt.

The asymmetry with ``role="user"`` audio is the whole point — the same
``type="audio"`` row is an input on one side of the conversation and a target on
the other, and role is what tells them apart.
"""

from __future__ import annotations

from typing import NamedTuple


# The talker stamps its RVQ codes with this ``meta`` key and the codec fetches
# them by it. The codes travel as an ordinary ``type="output"`` row — the
# transient kind a backbone appends mid-generation — because that is what they
# are: an intermediate the next module consumes, not a modality the conversation
# gained. The key is what tells them apart from the thinker's own output rows,
# the same way Bagel marks its flow tensors.
TALKER_CODES_KEY = "qwen3omni_talker_codes"


class PromptSegment(NamedTuple):
    """One prompt row's contribution to the token-level sequence.

    The talker slices the thinker traces by ChatML turn and needs to know which
    positions are multimodal. Upstream recovers both by scanning token ids; this
    module already knows them per item, so they are recorded instead.

    Here rather than beside its producer because three modules build or read it
    — the text encoder on the training path, the backbone's packer, and the
    talker in both.
    """

    role: str
    type: str
    length: int


__all__ = ["PromptSegment", "TALKER_CODES_KEY"]
