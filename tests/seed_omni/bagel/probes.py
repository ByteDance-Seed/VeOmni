"""BAGEL probe bindings for parity_suite."""

from __future__ import annotations

from tests.seed_omni.parity_suite.core.probes import probe_binding


PROBES = {
    "loss.total": probe_binding("OmniModel.forward.loss", "Total scalar training loss."),
    "text.hidden": probe_binding(
        "bagel_qwen2_mot.generate:hidden_state",
        "Text-path final hidden state.",
    ),
    "text.logits": probe_binding(
        "bagel_text_encoder.token_generate:logits",
        "Text logits for one deterministic generation step.",
    ),
    "text.greedy_token": probe_binding(
        "bagel_text_encoder.token_generate:greedy_token",
        "Greedy token selected from text logits.",
    ),
    "image.embeds": probe_binding(
        "bagel_siglip_navit.generate:image_embeds",
        "Visual understanding image embeddings.",
    ),
    "qwen.hidden": probe_binding(
        "bagel_qwen2_mot.generate:hidden_state",
        "Qwen2 MoT packed hidden state.",
    ),
    "qwen.kv_cache": probe_binding(
        "bagel_qwen2_mot.generate:cache_after_step",
        "Layer-wise key/value cache after one generation step.",
    ),
    "gen.latent_embeds": probe_binding(
        "bagel_flow_connector.embed_latent:latent_embeds",
        "Latent embeddings for image generation/editing.",
    ),
    "gen.velocity": probe_binding(
        "bagel_flow_connector.decode_velocity:velocity",
        "Rectified-flow velocity prediction.",
    ),
    "gen.x_t1": probe_binding("bagel_flow_connector.decode_velocity:x_t1", "Euler-updated latent."),
    "vae.latent": probe_binding("bagel_vae.encode:latent", "VAE context latent."),
    "grad.text_embedding": probe_binding(
        "bagel_text_encoder.embed_tokens.grad",
        "Representative text embedding gradient.",
    ),
    "grad.siglip": probe_binding(
        "bagel_siglip_navit.grad",
        "Representative visual-understanding gradient.",
    ),
    "grad.flow": probe_binding("bagel_flow_connector.grad", "Representative flow gradient."),
    "grad.qwen.early": probe_binding(
        "bagel_qwen2_mot.early_layer.grad",
        "Representative early Qwen gradient.",
    ),
    "grad.qwen.late": probe_binding(
        "bagel_qwen2_mot.late_layer.grad",
        "Representative late Qwen gradient.",
    ),
    "grad.qwen.generation": probe_binding(
        "bagel_qwen2_mot.generation_branch.grad",
        "Representative generation-branch Qwen gradient.",
    ),
    "param.after_step": probe_binding(
        "trainer.parameters_after_step",
        "Selected parameter samples after optimizer step.",
    ),
    "scheduler.lr": probe_binding("trainer.scheduler_lr", "Scheduler learning rates."),
    "grad_norm": probe_binding("trainer.grad_norm", "Trainer global gradient norm."),
}
