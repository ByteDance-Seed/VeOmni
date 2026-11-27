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

from collections.abc import Sequence
from typing import NamedTuple

import torch

from ...utils.conversation import ConversationItem


# The talker stamps its RVQ codes with this ``meta`` key and the codec fetches
# them by it. The codes travel as an ordinary ``type="output"`` row — the
# transient kind a backbone appends mid-generation — because that is what they
# are: an intermediate the next module consumes, not a modality the conversation
# gained. The key is what tells them apart from the thinker's own output rows,
# the same way Bagel marks its flow tensors.
TALKER_CODES_KEY = "qwen3omni_talker_codes"


def is_thinker_input(item: ConversationItem) -> bool:
    """Whether ``item`` takes a position in the thinker's packed sequence.

    False for two kinds of row, both of which stay on the carrier and are skipped
    by every walk that turns it into one sequence:

    * **dummy anchors** (``is_dummy``) — FSDP placeholders with no content;
    * **the assistant's speech** — the waveform the talker is taught to say. It is
      a *label*, so it never enters the thinker's forward: it holds a raw
      waveform, not an embedding, and the talker must not see it inside its own
      prompt.

    The same ``type="audio"`` row is an input on the user's side and a target on
    the assistant's, so ``role`` is what separates them. That is a convention of
    *this* family, which is why the predicate lives here rather than in the
    shared packer helpers — every family's carrier is walked by the same shared
    code, and the other families have no such row to skip.

    Every walk here must agree on it: the packer, the hidden-state scatter and the
    decode walk pair rows to tensors by position, so one that keeps a row another
    skipped shifts every projection after it without raising.
    """
    return not item.is_dummy and not (item.role == "assistant" and item.type == "audio")


def scatter_hidden_states(
    conversation_list: list[list[ConversationItem]],
    hidden_states_list: Sequence[torch.Tensor],
) -> None:
    """Write backbone hidden states back onto the carrier's thinker-input rows.

    The family-local counterpart of
    :func:`~...base.llm_packing.scatter_llm_hidden_states`. It exists because that
    helper skips only dummy rows, and this graph has a second kind to skip —
    see :func:`is_thinker_input`. Pairs by position, so it must skip exactly what
    the packer skipped.
    """
    iterator = iter(hidden_states_list)
    for sample in conversation_list:
        for part in sample:
            if not is_thinker_input(part):
                continue
            part.value = next(iterator)
    if next(iterator, None) is not None:
        raise RuntimeError("scatter_hidden_states: segment count exceeds the packed thinker-input items.")


def thinker_inputs(conversation_list: list[list[ConversationItem]]) -> list[ConversationItem]:
    """Every thinking-time row, in carrier order — the order the packer used."""
    return [item for sample in conversation_list for item in sample if is_thinker_input(item)]


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


__all__ = [
    "PromptSegment",
    "TALKER_CODES_KEY",
    "is_thinker_input",
    "scatter_hidden_states",
    "thinker_inputs",
]
