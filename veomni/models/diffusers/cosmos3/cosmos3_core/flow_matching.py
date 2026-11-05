"""Rectified-flow sampling and loss for Cosmos3 SFT.

Rectified-flow notation: ``x_0`` is noise, ``x_1`` the clean latent, ``sigma`` the
weight of the noise side.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def shift_times(t: Tensor, shift: float | Tensor | Sequence[float]) -> Tensor:
    """Resolution shift ``shift * t / (1 + (shift - 1) * t)``; scalar or per-sample ``shift``."""
    shift = torch.as_tensor(shift).to(device=t.device, dtype=t.dtype).reshape(-1)
    if shift.numel() == 1:
        shift = shift.reshape(())
    elif shift.numel() != t.shape[0]:
        raise ValueError(f"shift must hold 1 or {t.shape[0]} values, got {shift.numel()}")
    else:
        shift = shift.reshape(*t.shape)
    return shift * t / (1 + (shift - 1) * t)


class TrainTimeSampler:
    """Training-time ``sigma`` sampler: ``uniform``, ``logitnormal`` or ``waver``.

    ``waver`` (the video default) is ``t = 1 - u - s * (cos(pi/2 * u) ** 2 - 1 + u)``
    with ``s = 1.29``. Draws happen in fp32 on the CPU generator, then are cast.
    """

    _WAVER_MODE_S = 1.29

    def __init__(self, distribution: str = "uniform") -> None:
        if distribution not in ("uniform", "logitnormal", "waver"):
            raise NotImplementedError(f"Time distribution {distribution!r} is not implemented.")
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        shift: float | Tensor | Sequence[float] | None = None,
    ) -> Tensor:
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), generator=generator)
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), generator=generator))
        else:
            u = torch.rand((batch_size,), dtype=torch.float32, generator=generator)
            t = 1.0 - u - self._WAVER_MODE_S * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
        t = t.to(device=device, dtype=dtype)
        if shift is not None:
            t = shift_times(t, shift)
        return t


def _broadcast_sigma(sigma: Tensor | float, like: Tensor) -> Tensor:
    """Accept a scalar, ``(batch,)``, a pre-broadcast tensor, or a per-frame ``(T, 1, 1)`` against ``(C, T, H, W)``."""
    if not isinstance(sigma, Tensor):
        return torch.as_tensor(sigma, dtype=like.dtype, device=like.device)
    sigma = sigma.to(device=like.device, dtype=like.dtype)
    if sigma.ndim == like.ndim:
        return sigma
    if sigma.numel() == 1:
        return sigma.reshape(1)
    if sigma.ndim == 1:
        return sigma.reshape(sigma.shape[0], *([1] * (like.ndim - 1)))
    if sigma.ndim == like.ndim - 1 and sigma.shape[0] == like.shape[1]:
        return sigma.reshape(1, *sigma.shape)
    raise ValueError(f"sigma shape {tuple(sigma.shape)} does not broadcast to {tuple(like.shape)}")


def interpolate(noise: Tensor, clean: Tensor, sigma: Tensor | float) -> tuple[Tensor, Tensor]:
    """``x_t = noise * sigma + clean * (1 - sigma)`` and ``target = noise - clean``.

    A per-frame ``sigma * (1 - condition_mask)`` puts conditioning frames on the clean latent.
    """
    weight = _broadcast_sigma(sigma, noise)
    return noise * weight + clean * (1 - weight), noise - clean


def flow_matching_loss(
    pred: Tensor | Sequence[Tensor],
    target: Tensor | Sequence[Tensor],
    *,
    condition_mask: Tensor | Sequence[Tensor] | None = None,
    loss_scale: float = 10.0,
) -> Tensor:
    """Mean over samples of ``(sqerr * (1 - condition_mask)).mean()``, times ``loss_scale``.

    Conditioning frames stay in each sample's denominator, as in the reference.
    """
    preds = list(pred.unbind(0)) if isinstance(pred, Tensor) else list(pred)
    targets = list(target.unbind(0)) if isinstance(target, Tensor) else list(target)
    if condition_mask is None:
        masks = [None] * len(preds)
    else:
        masks = list(condition_mask.unbind(0)) if isinstance(condition_mask, Tensor) else list(condition_mask)
    per_sample = []
    for pred_i, target_i, mask_i in zip(preds, targets, masks):
        sqerr = (pred_i - target_i) ** 2
        noisy = torch.ones_like(sqerr) if mask_i is None else (1.0 - mask_i).expand_as(sqerr)
        per_sample.append((sqerr * noisy).mean())
    return torch.stack(per_sample).mean() * loss_scale
