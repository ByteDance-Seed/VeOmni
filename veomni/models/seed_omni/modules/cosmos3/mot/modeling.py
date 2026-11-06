"""Cosmos3 mixture-of-towers backbone: the reasoner (``und``) and generation (``gen``) towers.

Both towers stay in one module because every block's two passes share one
attention softmax: the generation queries attend to the reasoner's keys of the
same block. A micro-batch of several samples runs as one packed (TND) sequence
in which each sample keeps its own positions, fps and attention range.

Parameter names are the released checkpoint's: the reasoner's ``to_q`` /
``to_k`` / ``to_v`` / ``to_out``, the generator's ``add_*`` / ``to_add_out`` and
everything suffixed ``_moe_gen``.
"""

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils import logging

from ...module_modeling_base import PretrainedOmniModule
from ..core import (
    MoTDecoderLayer,
    MotRMSNorm,
    MotRope,
    build_packed_layout,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
)
from .configuration import Cosmos3MotConfig


logger = logging.get_logger(__name__)


# Substring whitelist of the parameters trained when the reasoner tower is frozen:
# the recipe's ``keys_to_select = ["moe_gen", ...]`` in checkpoint spelling, where
# the generation attention is exported as ``add_*`` / ``norm_added_*``.
TRAINABLE_NAME_KEYS = (
    "moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
)


class Cosmos3Mot(PretrainedOmniModule):
    config_class = Cosmos3MotConfig
    main_input_name = "hidden_gen"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _no_split_modules = ["MoTDecoderLayer"]

    def __init__(self, config: Cosmos3MotConfig):
        super().__init__(config)
        # HF's marker; the block loop hands each block to ``_gradient_checkpointing_func``.
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [MoTDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # One final norm per tower; only the generation tower's output is decoded.
        self.norm = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.norm_moe_gen = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.rotary_emb = MotRope(
            config.head_dim,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            interleaved=config.mrope_interleaved,
        )
        self.post_init()

    def freeze_model(self) -> None:
        """Called by the module runtime before the parallel wrap."""
        train_joint = self.config.train_joint_reasoner
        if train_joint and get_parallel_state().dp_mode == "ddp":
            # ``norm`` and the last block's reasoner side never reach the loss, and DDP
            # runs without ``find_unused_parameters``; FSDP2 tolerates them.
            raise NotImplementedError("train_joint_reasoner=True requires fsdp_mode='fsdp2', not 'ddp'.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(train_joint or any(key in name for key in TRAINABLE_NAME_KEYS))

    def forward(
        self,
        hidden_und: torch.Tensor,
        hidden_gen: torch.Tensor,
        *,
        und_lengths: Sequence[int],
        latent_grids: Sequence[tuple[int, int, int]],
        fps: Optional[Sequence[Optional[float]]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Packed ``(sum L_und, H)`` prompt embeddings + ``(sum L_gen, H)`` latent rows in.

        ``und_lengths`` / ``latent_grids`` / ``fps`` describe each sample in pack
        order; ``fps`` is the rate of the frames each latent was encoded from
        (``None`` falls back to ``base_fps``). Returns the generation tower's
        normalized output; the reasoner's is never decoded.
        """
        del kwargs
        device = hidden_gen.device
        fps = list(fps) if fps is not None else [None] * len(und_lengths)
        und_rope = self.rotary_emb(build_segmented_text_position_ids(und_lengths, device=device))
        gen_rope = self.rotary_emb(
            build_segmented_latent_position_ids(
                latent_grids,
                fps=[self._sample_fps(value) for value in fps],
                base_fps=self.config.base_fps,
                temporal_compression_factor=self.config.temporal_compression_factor,
                enable_fps_modulation=self.config.enable_fps_modulation,
                temporal_offsets=[self._gen_temporal_offset(length) for length in und_lengths],
                device=device,
            )
        )
        layout = None
        if len(und_lengths) > 1:
            layout = build_packed_layout(und_lengths, [t * h * w for t, h, w in latent_grids], device=device)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_und, hidden_gen = self._gradient_checkpointing_func(
                    layer, hidden_und, hidden_gen, und_rope=und_rope, gen_rope=gen_rope, layout=layout
                )
            else:
                hidden_und, hidden_gen = layer(
                    hidden_und, hidden_gen, und_rope=und_rope, gen_rope=gen_rope, layout=layout
                )
        return {"hidden_gen": self.norm_moe_gen(hidden_gen)}

    def _gen_temporal_offset(self, und_len: int) -> float:
        """The generation span starts after the prompt plus the modality margin."""
        return float(und_len) + float(self.config.unified_3d_mrope_temporal_modality_margin)

    def _sample_fps(self, fps: Optional[float]) -> Optional[float]:
        """Without a clip fps the reference falls back to ``base_fps`` (step 1.0 per latent frame)."""
        if fps is None:
            if not self.config.enable_fps_modulation:
                return None
            logger.warning_once(
                f"Cosmos3 mRoPE: no per-sample fps reached the MoT; using base_fps={self.config.base_fps}."
            )
            return self.config.base_fps
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {fps!r}.")
        return fps


__all__ = ["TRAINABLE_NAME_KEYS", "Cosmos3Mot"]
