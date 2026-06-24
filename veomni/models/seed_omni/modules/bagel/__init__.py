"""BAGEL SeedOmni V2 modules.

The V2 graph keeps BAGEL modules split by producer/consumer boundaries:

- training: text/SigLIP/VAE/flow hooks write embedded carrier items, then
  ``bagel_qwen2_mot.forward`` packs them into the MoT backbone and scatters
  hidden states back to the same items.
- prompt inference: ``bagel_text_encoder`` and optional VAE/SigLIP context
  feed ``bagel_qwen2_mot.generate`` for prompt prefill or text decoding.
- image denoise: ``flow_connector.prepare_denoise_query`` writes
  ``bagel_flow_query``; text marker wrapping feeds ``qwen2_mot.denoise_branch``;
  ``flow_connector.decode_velocity_from_hidden`` writes ``bagel_flow_velocity``;
  ``qwen2_mot.collect_velocity`` handles CFG branch collection; finally
  ``flow_connector.advance_denoise`` either loops or emits generated latents
  for VAE decode.
"""

from . import (
    convert_model,  # noqa: F401
    flow_connector,  # noqa: F401
    qwen2_mot,  # noqa: F401
    siglip_navit,  # noqa: F401
    text_encoder,  # noqa: F401
    vae,  # noqa: F401
)
