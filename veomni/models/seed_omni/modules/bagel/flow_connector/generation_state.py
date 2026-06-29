"""Generation-time denoise state for BAGEL flow connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FlowGenerationState:
    """Flow-side latent denoise state.

    The flow connector owns the numerical denoise state: noisy latent tokens,
    timesteps, dt values, and step progress. Branch/cache/controller state
    belongs to the MoT module and is intentionally not represented here.
    """

    _latents: torch.Tensor | None = None
    _timestep_values: torch.Tensor | None = None
    _dt_values: torch.Tensor | None = None
    _step_index: int = 0
    _image_shape: tuple[int, int] | None = None
    _grid_shape: tuple[int, int] | None = None
    _token_count: int = 0

    def reset(self) -> None:
        self._latents = None
        self._timestep_values = None
        self._dt_values = None
        self._step_index = 0
        self._image_shape = None
        self._grid_shape = None
        self._token_count = 0

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

        self._timestep_values = timesteps[:-1]
        self._dt_values = timesteps[:-1] - timesteps[1:]
        self._latents = torch.randn(
            token_count,
            int(patch_latent_dim),
            device=device,
            dtype=torch.float32,
        )
        self._step_index = 0
        self._image_shape = (image_height, image_width)
        self._grid_shape = grid_shape
        self._token_count = token_count

    @property
    def initialized(self) -> bool:
        return self._latents is not None

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return self._image_shape

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def latents(self) -> torch.Tensor:
        if self._latents is None:
            raise RuntimeError("BAGEL flow denoise latent state is not initialized.")
        return self._latents

    @property
    def timestep_values(self) -> torch.Tensor:
        if self._timestep_values is None:
            raise RuntimeError("BAGEL flow denoise timesteps are not initialized.")
        return self._timestep_values

    @property
    def dt_values(self) -> torch.Tensor:
        if self._dt_values is None:
            raise RuntimeError("BAGEL flow denoise dt values are not initialized.")
        return self._dt_values

    @property
    def grid_shape(self) -> tuple[int, int]:
        if self._grid_shape is None:
            raise RuntimeError("BAGEL flow denoise latent grid shape is not initialized.")
        return self._grid_shape

    def current_timestep(self) -> torch.Tensor:
        timesteps = self.timestep_values
        if self._step_index >= int(timesteps.numel()):
            raise RuntimeError("BAGEL flow denoise timestep index is past the final step.")
        return timesteps[self._step_index]

    def current_timestep_tokens(self) -> torch.Tensor:
        x_t = self.latents
        return self.current_timestep().reshape(1).expand(int(x_t.shape[0]))

    def is_complete(self) -> bool:
        timesteps = self.timestep_values
        return self._step_index >= int(timesteps.numel())

    def advance(self, velocity: torch.Tensor) -> bool:
        x_t = self.latents
        dts = self.dt_values
        velocity = velocity.to(device=x_t.device, dtype=x_t.dtype)
        if velocity.shape != x_t.shape:
            raise ValueError(
                f"BAGEL flow velocity shape mismatch: got {tuple(velocity.shape)}, expected {tuple(x_t.shape)}."
            )

        dt = dts[self._step_index].to(device=x_t.device, dtype=x_t.dtype)
        self._latents = x_t - velocity * dt
        self._step_index += 1
        return self.step_index >= int(dts.numel())


__all__ = ["FlowGenerationState"]
