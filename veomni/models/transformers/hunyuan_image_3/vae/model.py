# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lifecycle-aware container for the official HunyuanImage 3 VAE.

The container preserves the official ``vae.encoder.*`` and future
``vae.decoder.*`` state-dict hierarchy. The initial T2I capability constructs
only the frozen encoder; decoder construction remains an explicit future
capability rather than changing the container or checkpoint prefixes.
"""

from types import SimpleNamespace
from typing import Mapping

import torch
from torch import Tensor, nn

from .encoder import Encoder


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        if parameters.ndim in (4, 5):
            dimension = 1
        elif parameters.ndim == 3:
            dimension = 2
        else:
            raise ValueError(f"Unsupported posterior rank: {parameters.ndim}.")
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dimension)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean


class HunyuanImage3VAE(nn.Module):
    """Stable VAE container whose active submodules follow component policy."""

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        build_encoder: bool,
        build_decoder: bool,
    ) -> None:
        super().__init__()
        if not build_encoder and not build_decoder:
            raise ValueError("At least one HunyuanImage 3 VAE component must be constructed.")
        if build_decoder:
            raise NotImplementedError("The initial HunyuanImage 3 capability does not construct the VAE decoder.")

        self.config = SimpleNamespace(**dict(config))
        self.ffactor_spatial = int(self.config.ffactor_spatial)
        self.ffactor_temporal = int(self.config.ffactor_temporal)
        if build_encoder:
            self.encoder = Encoder(
                in_channels=int(self.config.in_channels),
                z_channels=int(self.config.latent_channels),
                block_out_channels=tuple(self.config.block_out_channels),
                num_res_blocks=int(self.config.layers_per_block),
                ffactor_spatial=self.ffactor_spatial,
                ffactor_temporal=self.ffactor_temporal,
                downsample_match_channel=bool(self.config.downsample_match_channel),
            )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "HunyuanImage3VAE":
        del mode
        return super().train(False)

    @torch.no_grad()
    def encode(self, pixel_values: Tensor) -> DiagonalGaussianDistribution:
        if not hasattr(self, "encoder"):
            raise RuntimeError("The HunyuanImage 3 VAE encoder is absent from this model.")
        if pixel_values.ndim == 4:
            pixel_values = pixel_values[:, :, None]
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must have shape [B, C, H, W] or [B, C, T, H, W].")
        if pixel_values.shape[2] == 1:
            pixel_values = pixel_values.expand(-1, -1, self.ffactor_temporal, -1, -1)
        elif pixel_values.shape[2] % self.ffactor_temporal:
            raise ValueError("The temporal dimension must be one or divisible by the VAE temporal factor.")
        return DiagonalGaussianDistribution(self.encoder(pixel_values))

    @torch.no_grad()
    def sample_latents(
        self,
        pixel_values: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        latents = self.encode(pixel_values).sample(generator)
        shift_factor = getattr(self.config, "shift_factor", None)
        scaling_factor = getattr(self.config, "scaling_factor", None)
        if shift_factor:
            latents = latents - shift_factor
        if scaling_factor:
            latents = latents * scaling_factor
        if latents.shape[2] != 1:
            raise ValueError("Single-image HunyuanImage 3 latents must have temporal length one.")
        return latents.squeeze(2)


__all__ = ["DiagonalGaussianDistribution", "HunyuanImage3VAE"]
