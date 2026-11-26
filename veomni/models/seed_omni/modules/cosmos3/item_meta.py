"""``ConversationItem.meta`` keys the Cosmos3 modules hand to each other.

A Cosmos3 sample is one user prompt (``type="text"``) and one assistant clip
(``type="video"``). The clip item's ``value`` is rewritten by every module it
passes through — frames, then latent, then generation-tower rows — so whatever a
later module still needs about it travels on ``meta``.
"""

# Set by the VAE once the item's value is the normalized latent. The offline
# cache stores it with the item, so the VAE preprocessor leaves such an item alone
# when ``seedomni_cached`` replays it.
LATENT_KEY = "cosmos3_latent"

# Frame rate of the frames the VAE encoded (``None`` when the clip states none);
# the MoT's mRoPE spaces the latent frames by it.
FPS_KEY = "cosmos3_fps"

# ``(C, T, H, W)`` of the clean latent, written by the flow connector when it turns
# the latent into generation-tower rows; the MoT reads the patch grid from it.
LATENT_SHAPE_KEY = "cosmos3_latent_shape"
