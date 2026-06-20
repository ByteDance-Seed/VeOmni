"""Generation-time denoise state for BAGEL flow connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FlowGenerationState:
    """Flow-side latent denoise state.

    The flow connector owns the numerical denoise state: noisy latent tokens,
    timesteps, dt values, and the current FSM phase. Branch/cache/controller
    state belongs to the MoT module and is intentionally not represented here.
    """

    phase: str = "prepare_query"
    x_t: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    dts: torch.Tensor | None = None
    step_index: int = 0
    image_shape: tuple[int, int] | None = None
    latent_grid_shape: tuple[int, int] | None = None
    token_count: int = 0

    @property
    def initialized(self) -> bool:
        return self.x_t is not None

    def reset(self) -> None:
        self.phase = "prepare_query"
        self.x_t = None
        self.timesteps = None
        self.dts = None
        self.step_index = 0
        self.image_shape = None
        self.latent_grid_shape = None
        self.token_count = 0

    def initialize(
        self,
        generation_kwargs: dict[str, Any],
        *,
        resolution: int,
        patch_latent_dim: int,
        device: torch.device,
    ) -> None:
        if bool(generation_kwargs.get("enable_taylorseer", False)):
            raise NotImplementedError("BAGEL infer_gen TaylorSeer is not implemented in the V2 graph path.")

        image_height = int(generation_kwargs.get("image_height", resolution))
        image_width = int(generation_kwargs.get("image_width", image_height))
        latent_downsample = int(generation_kwargs.get("latent_downsample", 16))
        if image_height % latent_downsample != 0 or image_width % latent_downsample != 0:
            raise ValueError("BAGEL infer_gen image size must be divisible by latent_downsample.")

        grid_shape = (image_height // latent_downsample, image_width // latent_downsample)
        token_count = int(grid_shape[0] * grid_shape[1])
        num_timesteps = int(generation_kwargs.get("num_timesteps", 50))
        if num_timesteps < 2:
            raise ValueError("BAGEL infer_gen num_timesteps must be at least 2.")

        timestep_shift = float(generation_kwargs.get("timestep_shift", 1.0))
        timesteps = torch.linspace(1.0, 0.0, num_timesteps, device=device, dtype=torch.float32)
        timesteps = timestep_shift * timesteps / (1.0 + (timestep_shift - 1.0) * timesteps)

        self.timesteps = timesteps[:-1]
        self.dts = timesteps[:-1] - timesteps[1:]
        self.x_t = torch.randn(
            token_count,
            int(patch_latent_dim),
            device=device,
            dtype=torch.float32,
        )
        self.step_index = 0
        self.image_shape = (image_height, image_width)
        self.latent_grid_shape = grid_shape
        self.token_count = token_count
        self.phase = "prepare_query"

    def require_latents(self) -> torch.Tensor:
        if self.x_t is None:
            raise RuntimeError("BAGEL flow denoise latent state is not initialized.")
        return self.x_t

    def require_timesteps(self) -> torch.Tensor:
        if self.timesteps is None:
            raise RuntimeError("BAGEL flow denoise timesteps are not initialized.")
        return self.timesteps

    def require_dts(self) -> torch.Tensor:
        if self.dts is None:
            raise RuntimeError("BAGEL flow denoise dt values are not initialized.")
        return self.dts

    def require_latent_grid_shape(self) -> tuple[int, int]:
        if self.latent_grid_shape is None:
            raise RuntimeError("BAGEL flow denoise latent grid shape is not initialized.")
        return self.latent_grid_shape

    def current_timestep(self) -> torch.Tensor:
        timesteps = self.require_timesteps()
        if self.step_index >= int(timesteps.numel()):
            raise RuntimeError("BAGEL flow denoise timestep index is past the final step.")
        return timesteps[self.step_index]

    def current_timestep_tokens(self) -> torch.Tensor:
        x_t = self.require_latents()
        return self.current_timestep().reshape(1).expand(int(x_t.shape[0]))

    def is_complete(self) -> bool:
        timesteps = self.require_timesteps()
        return self.step_index >= int(timesteps.numel())

    def strip_query_markers(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.token_count and int(hidden.shape[0]) == self.token_count + 2:
            hidden = hidden[1:-1]
        if self.token_count and int(hidden.shape[0]) != self.token_count:
            raise ValueError(
                f"BAGEL flow decode_velocity token count mismatch: got {hidden.shape[0]}, expected {self.token_count}."
            )
        return hidden

    def advance(self, velocity: torch.Tensor) -> bool:
        x_t = self.require_latents()
        dts = self.require_dts()
        velocity = velocity.to(device=x_t.device, dtype=x_t.dtype)
        if velocity.shape != x_t.shape:
            raise ValueError(
                f"BAGEL flow velocity shape mismatch: got {tuple(velocity.shape)}, expected {tuple(x_t.shape)}."
            )
        dt = dts[self.step_index].to(device=x_t.device, dtype=x_t.dtype)
        self.x_t = x_t - velocity * dt
        self.step_index += 1
        return self.step_index >= int(dts.numel())


__all__ = ["FlowGenerationState"]
