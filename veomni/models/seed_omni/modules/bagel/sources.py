"""BAGEL-local ``ConversationItem.meta`` routing tags.

Items carry no module ownership (see :mod:`...utils.conversation`), so BAGEL
routes its rows with two BAGEL-private meta keys:

* ``meta[BAGEL_CONTEXT_KEY]`` — which image context copy a row belongs to:
  ``BAGEL_SIGLIP_CONTEXT`` (understanding features, read by SigLIP-NaViT) or
  ``BAGEL_VAE_CONTEXT`` (latent context/target, read by the VAE and flow
  connector). The VAE preprocessor expands each raw image by its data-layer
  ``_IMG_TAG_KEY`` into one or both copies — ``_img_tag`` alone cannot tell the
  two copies of one ``edit`` image apart. The chat template stamps the same tag
  on the vision start/end marker rows around each copy, and MoT reads it to
  pick attention mode and expert routing.
* ``meta[BAGEL_PHASE_KEY]`` — the inference/flow step an ``output`` row is in:
  the understanding start token, then per denoise step flow query -> hidden ->
  velocity, and finally the generated latent handed to VAE decode.

An FSDP placeholder is flagged ``is_dummy=True`` and keeps the context tag of
the rows it stands in for, so each encoder selects real and dummy rows with
one filter.
"""

from __future__ import annotations

from ...utils.conversation import ConversationItem, iter_desired_items


BAGEL_CONTEXT_KEY = "bagel_context"
BAGEL_SIGLIP_CONTEXT = "siglip"
BAGEL_VAE_CONTEXT = "vae"

BAGEL_PHASE_KEY = "bagel_phase"
BAGEL_START_TOKEN = "start_token"
BAGEL_FLOW_QUERY = "flow_query"
BAGEL_FLOW_HIDDEN = "flow_hidden"
BAGEL_FLOW_VELOCITY = "flow_velocity"
BAGEL_GENERATED_LATENT = "generated_latent"


def bagel_context(item: ConversationItem) -> str | None:
    return item.meta.get(BAGEL_CONTEXT_KEY)


def bagel_phase(item: ConversationItem) -> str | None:
    return item.meta.get(BAGEL_PHASE_KEY)


def get_tail_phase_item(conversation: list[ConversationItem], phase: str) -> ConversationItem | None:
    """Return the latest ``output`` row in ``phase`` of a single conversation."""
    return next(
        iter_desired_items([conversation], types=["output"], reverse_item=True, meta={BAGEL_PHASE_KEY: [phase]}),
        None,
    )


__all__ = [
    "BAGEL_CONTEXT_KEY",
    "BAGEL_FLOW_HIDDEN",
    "BAGEL_FLOW_QUERY",
    "BAGEL_FLOW_VELOCITY",
    "BAGEL_GENERATED_LATENT",
    "BAGEL_PHASE_KEY",
    "BAGEL_SIGLIP_CONTEXT",
    "BAGEL_START_TOKEN",
    "BAGEL_VAE_CONTEXT",
    "bagel_context",
    "bagel_phase",
    "get_tail_phase_item",
]
