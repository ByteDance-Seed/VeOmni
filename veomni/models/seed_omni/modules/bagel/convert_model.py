"""Split a BAGEL checkpoint into SeedOmni modules.

Registered under ``OMNI_CONVERT_REGISTRY["bagel"]``; :func:`convert_checkpoint`
writes the omni checkpoint. Graphs come from ``configs/seed_omni/Bagel/bagel_7b_mot/``.
"""

from __future__ import annotations

from typing import Any

from veomni.models.seed_omni.utils.convert_registry import OMNI_CONVERT_REGISTRY
from veomni.models.seed_omni.utils.hf_layout import convert_with_hf_layout

from .hf_layout import BAGEL_GENERATION_KWARGS
from .position_embedding import PositionEmbedding


def convert_bagel_checkpoint(
    model_path: str,
    *,
    max_latent_size: int | str | None = None,
    training_graph: str | None = None,
    generation_graph: str | None = None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Split an upstream BAGEL checkpoint into five SeedOmni modules via :data:`BAGEL_HF_LAYOUT`."""
    converted = convert_with_hf_layout(
        model_path,
        training_graph=training_graph,
        generation_graph=generation_graph,
        attn_implementation=attn_implementation,
    )

    if max_latent_size is not None:
        target_latent_size = int(max_latent_size)
        if target_latent_size <= 0:
            raise ValueError(f"max_latent_size must be positive, got {max_latent_size!r}.")
        flow_connector = converted["modules"]["bagel_flow_connector"]
        if int(flow_connector.config.max_latent_size) != target_latent_size:
            flow_connector.config.max_latent_size = target_latent_size
            hidden_size = int(flow_connector.config.hidden_size)
            pos_embed = PositionEmbedding(max_num_patch_per_side=target_latent_size, hidden_size=hidden_size)
            flow_connector.latent_pos_embed = pos_embed.to(
                device=flow_connector.device,
                dtype=flow_connector.dtype,
            )

    return converted


@OMNI_CONVERT_REGISTRY.register("bagel")
def _register_bagel_convert():
    return convert_bagel_checkpoint


__all__ = ["BAGEL_GENERATION_KWARGS", "convert_bagel_checkpoint"]
