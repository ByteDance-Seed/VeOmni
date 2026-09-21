import math

import torch
from einops import rearrange
from torch import nn


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding."""
    b, _, d, l = encoded_text.shape  # noqa: E741
    eps = 1e-6

    sequence_lengths = attention_mask.sum(dim=-1)
    mask = rearrange(attention_mask.bool(), "b t -> b t 1 1")

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, -1, d * l)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    return normed.masked_fill(~mask_flattened, 0.0)


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization for V2 models."""
    B, T, D, L = encoded_text.shape  # noqa: N806
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


class FeatureExtractorV1(nn.Module):
    """19B: per-segment norm -> aggregate_embed -> 3840"""

    def __init__(self, aggregate_embed: nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.to(hidden_states.device if isinstance(hidden_states, torch.Tensor) else hidden_states[0].device)
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        dtype = encoded.dtype
        device = encoded.device
        normed = _norm_and_concat_padded_batch(encoded, attention_mask)
        features = self.aggregate_embed(normed.to(device=device, dtype=dtype))
        if self.is_av:
            return features, features
        return features, None


class FeatureExtractorV2(nn.Module):
    """22B: per-token RMS norm -> rescale -> dual aggregate embeds"""

    def __init__(
        self,
        video_aggregate_embed: nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.to(hidden_states.device if isinstance(hidden_states, torch.Tensor) else hidden_states[0].device)
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.to(device=encoded.device, dtype=encoded.dtype)
        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
        return video, audio
