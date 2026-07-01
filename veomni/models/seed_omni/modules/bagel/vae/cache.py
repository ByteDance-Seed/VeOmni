"""Typed tensor views for BAGEL VAE offline cache payloads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....mixins.offline_encoding import OfflineEncodedCache


BAGEL_VAE_POSTERIOR_CACHE_KIND = "bagel_vae_posterior"


@dataclass(frozen=True)
class BagelVAEPosteriorCache(OfflineEncodedCache):
    """DTO for a BAGEL VAE posterior cache tensor."""

    mean: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.mean) or not torch.is_tensor(self.logvar):
            raise TypeError("BAGEL VAE posterior cache fields must be tensors.")
        if self.mean.shape != self.logvar.shape:
            raise ValueError(
                "BAGEL VAE posterior cache mean/logvar shape mismatch: "
                f"{tuple(self.mean.shape)} vs {tuple(self.logvar.shape)}."
            )

    def to_tensor(self) -> torch.Tensor:
        if self.mean.dim() == 3:
            return torch.stack((self.mean, self.logvar), dim=0)
        if self.mean.dim() == 4:
            return torch.stack((self.mean, self.logvar), dim=1)
        raise ValueError(
            "BAGEL VAE posterior cache expects item tensors shaped (C, H, W) "
            f"or batch tensors shaped (B, C, H, W), got {tuple(self.mean.shape)}."
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> BagelVAEPosteriorCache:
        if tensor.dim() == 4 and int(tensor.shape[0]) == 2:
            return cls(mean=tensor[0], logvar=tensor[1])
        if tensor.dim() == 5 and int(tensor.shape[1]) == 2:
            return cls(mean=tensor[:, 0], logvar=tensor[:, 1])
        raise ValueError(
            "BAGEL VAE posterior cache tensor must be shaped (2, C, H, W) "
            f"or (B, 2, C, H, W), got {tuple(tensor.shape)}."
        )


__all__ = ["BAGEL_VAE_POSTERIOR_CACHE_KIND", "BagelVAEPosteriorCache"]
