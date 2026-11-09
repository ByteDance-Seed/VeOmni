"""HF PreTrainedModel wrapper for the Cosmos3 mixture-of-towers DiT.

The frozen reasoner tower (Qwen3-VL text) reads the prompt tokens, the trainable
generation tower reads the noised latent patches, and both share one attention per
block. The flow-matching loss is computed internally. Parameter names are the
released checkpoint's, so it loads without a key-mapping table.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group
from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel
from veomni.utils import logging

from ..cosmos3_core import (
    MoTDecoderLayer,
    MotRMSNorm,
    MotRope,
    PackedLayout,
    TimestepEmbedder,
    add_time_conditioning,
    build_latent_grid,
    build_latent_position_ids,
    build_packed_layout,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
    build_text_position_ids,
    flow_matching_loss,
)
from .configuration_cosmos3_transformer import Cosmos3MoTTransformerConfig


logger = logging.get_logger(__name__)


# Substring whitelist of parameters trained when the reasoner tower is frozen. It is
# the recipe's ``keys_to_select = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]``
# in checkpoint spelling: the generation attention is exported as ``add_*`` /
# ``norm_added_*`` and ``vae2llm`` / ``llm2vae`` as ``proj_in`` / ``proj_out``.
TRAINABLE_NAME_KEYS = (
    "moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "time_embedder",
    "proj_in",
    "proj_out",
)

# Frozen even with ``train_joint_reasoner``: there is no AR loss, so ``lm_head`` never
# receives a gradient.
ALWAYS_FROZEN_NAME_PREFIXES = ("lm_head.",)

# Checkpoint tensors this model does not own: the audio and action pathways (unused
# by Nano) and the Qwen3-VL vision encoder, which the repo-root union index lists.
IGNORED_CHECKPOINT_KEY_PREFIXES = (
    "audio_proj_in.",
    "audio_proj_out.",
    "audio_modality_embed",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
    "blocks.",
    "merger.",
    "deepstack_merger_list.",
    "patch_embed.",
    "pos_embed.",
)


@dataclass
class Cosmos3MoTTransformerOutput(ModelOutput):
    loss: Optional[dict[str, torch.Tensor]] = None
    predictions: Optional[list[torch.Tensor]] = None


def patchify_latent(latent: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """``[C, F, H, W]`` -> ``[F * H/p * W/p, p*p*C]`` in the reference's ``thw, (p, q, c)`` order."""
    channels, frames, height, width = latent.shape
    ph, pw = patch_size
    if height % ph or width % pw:
        raise ValueError(f"latent spatial size {(height, width)} is not divisible by the patch size {(ph, pw)}.")
    patches = latent.reshape(channels, frames, height // ph, ph, width // pw, pw)
    patches = torch.einsum("cthpwq->thwpqc", patches)
    return patches.reshape(-1, ph * pw * channels)


def unpatchify_latent(
    patches: torch.Tensor, latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`patchify_latent`."""
    channels, frames, height, width = latent_shape
    ph, pw = patch_size
    shaped = patches.reshape(frames, height // ph, width // pw, ph, pw, channels)
    latent = torch.einsum("thwpqc->cthpwq", shaped)
    return latent.reshape(channels, frames, height, width)


def patch_row_count(latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int]) -> int:
    _, frames, height, width = latent_shape
    return frames * (height // patch_size[0]) * (width // patch_size[1])


class Cosmos3MoTTransformer(PreTrainedModel):
    config_class = Cosmos3MoTTransformerConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _no_split_modules = ["MoTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [f"^{re.escape(prefix)}" for prefix in IGNORED_CHECKPOINT_KEY_PREFIXES]

    def __init__(self, config: Cosmos3MoTTransformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        # HF's marker; the block loop below hands each block to ``_gradient_checkpointing_func``.
        self.gradient_checkpointing = False

        patch = config.latent_patch_size
        self.patch_size = (patch, patch)
        # The time axis is never patched.
        self.latent_grid_patch = (1, patch, patch)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None))
        self.layers = nn.ModuleList(
            [MoTDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # One final norm per tower; only the generation tower's output is decoded.
        self.norm = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.norm_moe_gen = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.proj_in = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_latent_dim)
        # Built so the checkpoint loads completely; never used without an AR loss.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = MotRope(
            config.head_dim,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            interleaved=config.mrope_interleaved,
        )

        logger.info_rank0(
            f"Cosmos3 mRoPE: enable_fps_modulation={config.enable_fps_modulation} base_fps={config.base_fps:g} "
            f"temporal_compression_factor={config.temporal_compression_factor} "
            f"modality_margin={config.unified_3d_mrope_temporal_modality_margin}."
        )
        self._init_modality_heads()
        self.freeze_model()

    @torch.no_grad()
    def _init_modality_heads(self) -> None:
        """The reference's truncated-normal init; only matters without a checkpoint."""
        self.time_embedder.reset_parameters()
        std = 1.0 / math.sqrt(self.config.patch_latent_dim)
        nn.init.trunc_normal_(self.proj_in.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.proj_in.bias)
        std = 1.0 / math.sqrt(self.config.hidden_size)
        nn.init.trunc_normal_(self.proj_out.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(self.proj_out.bias)

    def freeze_model(self) -> None:
        """Called by ``VeOmniModelRuntime`` before FSDP wrapping."""
        train_joint = self.config.train_joint_reasoner
        if train_joint and get_parallel_state().dp_mode == "ddp":
            # `norm` and the last block's reasoner-side parameters never reach the loss, and DDP
            # runs without `find_unused_parameters`; FSDP2 tolerates them.
            raise NotImplementedError("train_joint_reasoner=True requires fsdp_mode='fsdp2', not 'ddp'.")
        for name, parameter in self.named_parameters():
            if name.startswith(ALWAYS_FROZEN_NAME_PREFIXES):
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(train_joint or any(key in name for key in TRAINABLE_NAME_KEYS))

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(extra_parallel_plan={})

    def _gen_temporal_offset(self, und_len: int) -> float:
        """The generation span starts after the prompt plus the modality margin."""
        return float(und_len) + float(self.config.unified_3d_mrope_temporal_modality_margin)

    def _sample_fps(self, fps: Optional[float]) -> Optional[float]:
        """Without a clip fps the reference falls back to ``base_fps`` (step 1.0 per latent frame)."""
        if fps is None:
            if not self.config.enable_fps_modulation:
                return None
            logger.warning_once(
                f"Cosmos3 mRoPE: no per-sample fps reached the transformer; using base_fps={self.config.base_fps}."
            )
            return self.config.base_fps
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {fps!r}.")
        return fps

    def _noisy_patch_row_weight(
        self, condition_mask: Optional[torch.Tensor], latent_shape, device
    ) -> Optional[torch.Tensor]:
        """Per-frame ``condition_mask`` -> per-patch-row ``1 - mask``; clean frames get no timestep."""
        if condition_mask is None:
            return None
        mask = condition_mask.to(device=device, dtype=torch.float32).reshape(-1)
        grid_t, grid_h, grid_w = build_latent_grid(latent_shape, self.latent_grid_patch)
        if mask.numel() != grid_t:
            raise ValueError(f"condition_mask has {mask.numel()} frames but the latent has {grid_t}.")
        return (1.0 - mask).reshape(grid_t, 1).expand(grid_t, grid_h * grid_w).reshape(-1, 1).contiguous()

    def _run_layers(self, hidden_und, hidden_gen, und_rope, gen_rope, layout: Optional[PackedLayout] = None):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_und, hidden_gen = self._gradient_checkpointing_func(
                    layer, hidden_und, hidden_gen, und_rope=und_rope, gen_rope=gen_rope, layout=layout
                )
            else:
                hidden_und, hidden_gen = layer(
                    hidden_und, hidden_gen, und_rope=und_rope, gen_rope=gen_rope, layout=layout
                )
        return self.proj_out(self.norm_moe_gen(hidden_gen))

    def _forward_sample(self, input_ids, latent, timestep, condition_mask=None, fps=None) -> torch.Tensor:
        """One sample: ``LongTensor(L,)`` prompt, ``[C, T, H, W]`` latent -> ``[C, T, H, W]`` prediction."""
        tokens = input_ids.to(device=latent.device, dtype=torch.long).reshape(-1)
        hidden_und = self.embed_tokens(tokens)
        # Latents arrive in the VAE dtype; FSDP does not cast activations.
        hidden_gen = self.proj_in(patchify_latent(latent, self.patch_size).to(self.proj_in.weight.dtype))
        hidden_gen = add_time_conditioning(
            hidden_gen,
            timestep,
            self.time_embedder,
            self.config.timestep_scale,
            row_weight=self._noisy_patch_row_weight(condition_mask, latent.shape, latent.device),
        )

        num_text_tokens = tokens.numel()
        grid_t, grid_h, grid_w = build_latent_grid(latent.shape, self.latent_grid_patch)
        und_rope = self.rotary_emb(build_text_position_ids(num_text_tokens, device=latent.device))
        gen_rope = self.rotary_emb(
            build_latent_position_ids(
                grid_t,
                grid_h,
                grid_w,
                fps=self._sample_fps(fps),
                base_fps=self.config.base_fps,
                temporal_compression_factor=self.config.temporal_compression_factor,
                enable_fps_modulation=self.config.enable_fps_modulation,
                temporal_offset=self._gen_temporal_offset(num_text_tokens),
                device=latent.device,
            )
        )
        rows = self._run_layers(hidden_und, hidden_gen, und_rope, gen_rope)
        return unpatchify_latent(rows, latent.shape, self.patch_size)

    def _forward_packed(self, input_ids, hidden_states, timestep, condition_masks, fps) -> list[torch.Tensor]:
        """All samples of the micro-batch as one packed (TND) sequence; each sample keeps its own positions,
        timestep and fps."""
        device = hidden_states[0].device
        tokens = [ids.to(device=device, dtype=torch.long).reshape(-1) for ids in input_ids]
        patches = [patchify_latent(latent, self.patch_size) for latent in hidden_states]
        latent_shapes = [tuple(latent.shape) for latent in hidden_states]
        und_lengths = [int(row.numel()) for row in tokens]
        gen_lengths = [int(row.shape[0]) for row in patches]
        layout = build_packed_layout(und_lengths, gen_lengths, device=device)

        hidden_und = self.embed_tokens(torch.cat(tokens, dim=0))
        hidden_gen = self.proj_in(torch.cat(patches, dim=0).to(self.proj_in.weight.dtype))

        sigmas = torch.cat([torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1) for t in timestep])
        sigma_per_row = torch.repeat_interleave(sigmas, torch.tensor(gen_lengths, dtype=torch.long, device=device))
        row_weight = None
        if any(mask is not None for mask in condition_masks):
            row_weight = torch.cat(
                [
                    self._noisy_patch_row_weight(mask, shape, device)
                    if mask is not None
                    else torch.ones((length, 1), dtype=torch.float32, device=device)
                    for mask, shape, length in zip(condition_masks, latent_shapes, gen_lengths)
                ]
            )
        hidden_gen = add_time_conditioning(
            hidden_gen, sigma_per_row, self.time_embedder, self.config.timestep_scale, row_weight=row_weight
        )

        und_rope = self.rotary_emb(build_segmented_text_position_ids(und_lengths, device=device))
        gen_rope = self.rotary_emb(
            build_segmented_latent_position_ids(
                [build_latent_grid(shape, self.latent_grid_patch) for shape in latent_shapes],
                fps=[self._sample_fps(value) for value in fps],
                base_fps=self.config.base_fps,
                temporal_compression_factor=self.config.temporal_compression_factor,
                enable_fps_modulation=self.config.enable_fps_modulation,
                temporal_offsets=[self._gen_temporal_offset(length) for length in und_lengths],
                device=device,
            )
        )
        rows = self._run_layers(hidden_und, hidden_gen, und_rope, gen_rope, layout=layout)
        return [
            unpatchify_latent(part, shape, self.patch_size)
            for part, shape in zip(rows.split(gen_lengths), latent_shapes)
        ]

    def forward(
        self,
        input_ids: list[torch.Tensor],
        hidden_states: list[torch.Tensor],
        timestep: list[torch.Tensor],
        training_target: Optional[list[torch.Tensor]] = None,
        condition_mask: Optional[list[torch.Tensor]] = None,
        fps: Optional[Sequence[Optional[float]]] = None,
        **kwargs,
    ) -> Cosmos3MoTTransformerOutput:
        """Per-sample lists in (samples differ in geometry); ``{"dm_loss": mean per-sample loss}`` out.

        ``timestep`` is already ``sigma * num_train_timesteps``. ``condition_mask`` is
        ``[T_lat, 1, 1]`` per sample with ``1`` on clean conditioning frames. A
        micro-batch with more than one sample runs packed.
        """
        group = get_ulysses_sequence_parallel_group()
        if group is not None and dist.get_world_size(group) > 1:
            raise NotImplementedError("Cosmos3 joint attention does not support Ulysses sequence parallelism.")
        sample_count = len(hidden_states)
        masks = list(condition_mask) if condition_mask is not None else [None] * sample_count
        fps = list(fps) if fps is not None else [None] * sample_count

        if sample_count > 1:
            predictions = self._forward_packed(input_ids, hidden_states, timestep, masks, fps)
        else:
            predictions = [self._forward_sample(input_ids[0], hidden_states[0], timestep[0], masks[0], fps[0])]

        loss = None
        if training_target is not None:
            # A plain per-sample mean, also when packed; the trainer divides by the micro-batch count.
            losses = [
                flow_matching_loss(
                    prediction.float().unsqueeze(0),
                    target.float().unsqueeze(0),
                    condition_mask=None
                    if mask is None
                    else [mask.to(device=prediction.device, dtype=torch.float32).unsqueeze(0)],
                    loss_scale=self.config.loss_scale,
                )
                for prediction, target, mask in zip(predictions, training_target, masks)
            ]
            loss = {"dm_loss": torch.stack(losses).mean()}
        return Cosmos3MoTTransformerOutput(loss=loss, predictions=predictions)
