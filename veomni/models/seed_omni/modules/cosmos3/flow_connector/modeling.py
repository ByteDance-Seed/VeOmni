"""Cosmos3 flow connector: everything trainable outside the MoT blocks.

``embed_latent`` draws sigma, the noise and the T2V / I2V / V2V clean-prefix mask
from one CPU generator, builds ``x_t``, patchifies it and adds the timestep
embedding; ``decode_velocity`` projects the generation tower's output back to
latent patches and returns the flow-matching loss. The parameter names
(``proj_in`` / ``proj_out`` / ``time_embedder``) are the released checkpoint's.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from veomni.distributed.parallel_state import get_parallel_state

from ...module_modeling_base import PretrainedOmniModule
from ..core import (
    TimestepEmbedder,
    TrainTimeSampler,
    add_time_conditioning,
    build_latent_grid,
    flow_matching_loss,
    interpolate,
    patch_row_count,
    patchify_latent,
    unpatchify_latent,
)
from .configuration import Cosmos3FlowConnectorConfig


class Cosmos3FlowConnector(PretrainedOmniModule):
    config_class = Cosmos3FlowConnectorConfig
    main_input_name = "latents"

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # Projections and the timestep MLP only; the MoT owns the attention.
        return attn_implementation

    def __init__(self, config: Cosmos3FlowConnectorConfig):
        super().__init__(config)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.proj_in = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.train_time_sampler = TrainTimeSampler(config.train_time_video_distribution)
        # Built on first use, inside this module's ParallelState, so it can be seeded per dp rank.
        self._generator: Optional[torch.Generator] = None
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """The reference's truncated-normal init; only matters without a checkpoint."""
        if module is self.time_embedder:
            module.reset_parameters()
        elif module is self.proj_in or module is self.proj_out:
            std = 1.0 / math.sqrt(module.in_features)
            nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(module.bias)

    @property
    def patch_size(self) -> tuple[int, int]:
        return (self.config.latent_patch_size, self.config.latent_patch_size)

    def embed_latent(self, latents: list[torch.Tensor], **kwargs: Any) -> Dict[str, Any]:
        """Clean ``(C, T, H, W)`` latents -> noised generation-tower rows, plus what the loss needs.

        A micro-batch of one sample embeds its single timestep once and broadcasts
        it, as the reference does; a packed one embeds one timestep per row.
        """
        del kwargs
        noisy, targets, masks, timesteps = self.sample_flow(latents)
        rows = [patchify_latent(latent, self.patch_size) for latent in noisy]
        hidden = self.proj_in(torch.cat(rows, dim=0).to(self.proj_in.weight.dtype))
        gen_lengths = [int(row.shape[0]) for row in rows]
        row_weight = torch.cat(
            [self._noisy_row_weight(mask, tuple(latent.shape), hidden.device) for mask, latent in zip(masks, noisy)]
        )
        if len(noisy) == 1:
            sigma = timesteps[0]
        else:
            sigma = torch.repeat_interleave(
                torch.cat(
                    [timestep.to(device=hidden.device, dtype=torch.float32).reshape(-1) for timestep in timesteps]
                ),
                torch.tensor(gen_lengths, dtype=torch.long, device=hidden.device),
            )
        hidden = add_time_conditioning(
            hidden, sigma, self.time_embedder, self.config.timestep_scale, row_weight=row_weight
        )
        return {
            "hidden_gen": hidden,
            "gen_lengths": gen_lengths,
            "latent_shapes": [tuple(latent.shape) for latent in noisy],
            "training_target": targets,
            "condition_mask": masks,
        }

    def decode_velocity(
        self,
        hidden_gen: torch.Tensor,
        latent_shapes: list[tuple[int, int, int, int]],
        training_target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Generation-tower rows -> per-sample velocity -> the mean per-sample loss.

        A plain mean over the micro-batch's samples, packed or not; the trainer
        divides by the number of micro-batches.
        """
        del kwargs
        rows = self.proj_out(hidden_gen)
        lengths = [patch_row_count(shape, self.patch_size) for shape in latent_shapes]
        predictions = [
            unpatchify_latent(part, shape, self.patch_size) for part, shape in zip(rows.split(lengths), latent_shapes)
        ]
        losses = [
            flow_matching_loss(
                prediction.float().unsqueeze(0),
                target.float().unsqueeze(0),
                condition_mask=[mask.to(device=prediction.device, dtype=torch.float32).unsqueeze(0)],
                loss_scale=self.config.loss_scale,
            )
            for prediction, target, mask in zip(predictions, training_target, condition_mask)
        ]
        return {"_loss": torch.stack(losses).mean()}

    def generate(self, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        raise NotImplementedError(
            "Cosmos3 on SeedOmni is training-only: there is no flow sampler yet, so it cannot generate a video."
        )

    @torch.no_grad()
    def sample_flow(
        self, latents: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """``(x_t, target, condition_mask, timestep)`` per sample.

        ``x_t = noise * sigma + clean * (1 - sigma)`` with ``sigma = 0`` on the clean
        prefix frames; ``target = noise - clean``; ``condition_mask`` is
        ``(T_lat, 1, 1)`` with ``1`` on clean frames; ``timestep = sigma *
        num_train_timesteps``. Draw order — every sigma, then per sample its noise
        and then its mask — is the reference's.
        """
        generator = self._sampling_generator()
        count = len(latents)
        shift = torch.tensor([self.config.shift_for_resolution()] * count, dtype=torch.float32)
        sigmas = self.train_time_sampler(count, generator=generator, shift=shift).to(torch.float32)
        noisy, targets, masks, timesteps = [], [], [], []
        for index, clean in enumerate(latents):
            sigma = sigmas[index : index + 1]
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(device=clean.device, dtype=clean.dtype)
            # Drawn after the noise, so the noise stream does not depend on the mixture.
            mask = self._sample_condition_mask(clean, generator)
            effective_sigma = sigma.to(device=noise.device).reshape(1, 1, 1) * (
                1.0 - mask.to(device=noise.device, dtype=sigma.dtype)
            )
            x_t, target = interpolate(noise, clean, effective_sigma)
            noisy.append(x_t)
            targets.append(target)
            masks.append(mask)
            timesteps.append((sigma * self.config.num_train_timesteps).to(torch.float32))
        return noisy, targets, masks, timesteps

    def _sample_condition_mask(self, clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Clean prefix of 0 (T2V), 1 (I2V) or 2 (V2V) latent frames, at least one frame left noisy."""
        latent_frames = int(clean.shape[1])
        options = self.config.conditioning_probabilities()
        draw = float(torch.rand((), generator=generator, dtype=torch.float32))
        cumulative = 0.0
        num_clean = options[-1][0]
        for frames, probability in options:
            cumulative += probability
            if draw < cumulative:
                num_clean = frames
                break
        num_clean = max(0, min(num_clean, latent_frames - 1))
        # float32: a bf16 mask would round ``1 - mask`` and leak sigma into clean frames.
        mask = torch.zeros((latent_frames, 1, 1), device=clean.device, dtype=torch.float32)
        if num_clean:
            mask[:num_clean] = 1.0
        return mask

    def _noisy_row_weight(self, mask: torch.Tensor, latent_shape: tuple[int, ...], device) -> torch.Tensor:
        """Per-frame mask -> per-patch-row ``1 - mask``: clean frames get no timestep embedding."""
        weight = mask.to(device=device, dtype=torch.float32).reshape(-1)
        grid_t, grid_h, grid_w = build_latent_grid(latent_shape, (1, *self.patch_size))
        if weight.numel() != grid_t:
            raise ValueError(f"condition_mask has {weight.numel()} frames but the latent has {grid_t}.")
        return (1.0 - weight).reshape(grid_t, 1).expand(grid_t, grid_h * grid_w).reshape(-1, 1).contiguous()

    def _sampling_generator(self) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(self.config.seed) + get_parallel_state().dp_rank)
        return self._generator

    def rng_state_dict(self) -> Dict[str, torch.Tensor]:
        """Sampling state to checkpoint per rank; see :meth:`ModuleRuntime.extra_state`."""
        return {"generator": self._sampling_generator().get_state()}

    def load_rng_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._sampling_generator().set_state(state["generator"])


__all__ = ["Cosmos3FlowConnector"]
