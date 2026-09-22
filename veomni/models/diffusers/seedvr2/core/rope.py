"""SeedVR2 multimodal 3D RoPE, evaluated only at the required positions.

Matches models/dit_v2/rope.py at SeedVR e4de8c24441a67e1b7df56abea10645059bb1185.
Preserves the persistent rope.freqs key and rotary-embedding-torch 0.5.3 language
frequency convention without allocating the full 1024 x 128 x 128 grid.
"""

import torch
from torch import nn


def rotate(tensor, frequencies):
    size = frequencies.shape[-1]
    if size > tensor.shape[-1]:
        raise ValueError("Rotary dimensions exceed the attention head dimension.")
    values = tensor[..., :size].float()
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    frequencies = frequencies[:, None, :].float()
    result = values * frequencies.cos() + rotated * frequencies.sin()
    return torch.cat((result.to(tensor.dtype), tensor[..., size:]), dim=-1)


class NaMMRotaryEmbedding3d(nn.Module):
    mm = True

    def __init__(self, dim):
        super().__init__()
        axis_dim = dim // 3
        if axis_dim < 2:
            raise ValueError("SeedVR2 3D RoPE needs at least two dimensions per axis.")
        self.rope = nn.Module()
        self.rope.register_buffer(
            "freqs", 1.0 / (10000 ** (torch.arange(0, axis_dim, 2)[: axis_dim // 2].float() / axis_dim))
        )

    def get_freqs(self, vid_shape, txt_shape):
        video, text = [], []
        freqs = self.rope.freqs
        for (frames, height, width), (length,) in zip(vid_shape.tolist(), txt_shape.tolist()):
            if min(frames, height, width, length) < 1 or frames + length > 1024 or height > 128 or width > 128:
                raise ValueError("SeedVR2 window/text positions exceed the source RoPE bounds.")
            coordinates = torch.meshgrid(
                torch.arange(length, length + frames, device=freqs.device),
                torch.arange(height, device=freqs.device),
                torch.arange(width, device=freqs.device),
                indexing="ij",
            )
            video.append(
                torch.cat(
                    [
                        (axis.reshape(-1, 1).to(freqs.dtype) * freqs).repeat_interleave(2, dim=-1)
                        for axis in coordinates
                    ],
                    dim=-1,
                )
            )
            text.append(
                (torch.arange(length, device=freqs.device)[:, None].to(freqs.dtype) * freqs)
                .repeat_interleave(2, dim=-1)
                .repeat(1, 3)
            )
        return torch.cat(video), torch.cat(text)

    def forward(self, vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache):
        vid_freqs, txt_freqs = cache("mmrope_freqs_3d", lambda: self.get_freqs(vid_shape, txt_shape))
        return rotate(vid_q, vid_freqs), rotate(vid_k, vid_freqs), rotate(txt_q, txt_freqs), rotate(txt_k, txt_freqs)


def get_na_rope(rope_type, dim):
    if rope_type is None:
        return None
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim)
    raise ValueError(f"Unsupported SeedVR2 RoPE: {rope_type}")
