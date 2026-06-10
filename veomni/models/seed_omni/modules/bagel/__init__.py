"""BAGEL SeedOmni V2 modules.

The BAGEL checkpoint is split into five module-owned state dicts:

* ``bagel_text_encoder``: token embeddings and LM head
* ``bagel_siglip_navit``: SigLIP NaViT visual-understanding tower
* ``bagel_qwen2_mot``: Qwen2 MoT backbone blocks
* ``bagel_vae``: latent image autoencoder
* ``bagel_flow_connector``: VAE/LLM flow connector and positional/timestep layers
"""

from . import convert_model, flow_connector, qwen2_mot, siglip_navit, text_encoder, vae  # noqa: F401
