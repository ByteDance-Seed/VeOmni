"""Cosmos3-Nano (mixture-of-towers text-to-video DiT) as SeedOmni modules.

``cosmos3_text_encoder`` -> ``cosmos3_mot`` <- ``cosmos3_flow_connector`` <- ``cosmos3_vae``:

* ``text_encoder`` — the reasoner tower's word embedding; tokenizes the prompt in
  the DataLoader worker.
* ``vae`` — the frozen Wan2.2 VAE; cuts and resizes the clip in the worker and is
  cacheable (``OfflineEncodingMixin``).
* ``mot`` — every decoder block of both towers. They stay one module because each
  block's two passes share one attention softmax.
* ``flow_connector`` — ``proj_in`` / ``proj_out`` / ``time_embedder``, the sigma /
  noise / clean-prefix draws, and the loss.

``core/`` is the framework-neutral numeric code, carried over from the DiT port.
"""

from . import convert_model, flow_connector, mot, text_encoder, vae  # noqa: F401
