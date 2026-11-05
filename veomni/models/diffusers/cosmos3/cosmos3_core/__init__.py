"""Numeric core of the Cosmos3 mixture-of-towers transformer (torch + VeOmni ops only)."""

from .flow_matching import TrainTimeSampler, flow_matching_loss, interpolate
from .mot_attention import MotAttention, MotRMSNorm
from .mot_decoder_layer import MoTDecoderLayer
from .mot_rope import (
    MotRope,
    build_latent_grid,
    build_latent_position_ids,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
    build_text_position_ids,
    latent_temporal_positions,
)
from .packed_layout import PackedLayout, build_packed_layout
from .timestep_embedder import TimestepEmbedder, add_time_conditioning


__all__ = [
    "MoTDecoderLayer",
    "MotAttention",
    "MotRMSNorm",
    "MotRope",
    "PackedLayout",
    "TimestepEmbedder",
    "TrainTimeSampler",
    "add_time_conditioning",
    "build_latent_grid",
    "build_latent_position_ids",
    "build_packed_layout",
    "build_segmented_latent_position_ids",
    "build_segmented_text_position_ids",
    "build_text_position_ids",
    "flow_matching_loss",
    "interpolate",
    "latent_temporal_positions",
]
