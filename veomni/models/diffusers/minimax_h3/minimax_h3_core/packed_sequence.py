"""FL2VA Packed Sequence Builder.

Layout: [text | cond | audio | video | pad]

Output packed dict fields:
  seq_len, img_pos, audio_pos, text_pos, update_mask,
  img_position_ids [1, seq_len, 3] float64, token_tags [seq_len] int64,
  cu_seqlens [3] int32,
  text_len, audio_channel, audio_t, latent_t, latent_h_patched, latent_w_patched, cond_rows
"""

from __future__ import annotations

import numpy as np
import torch


_VIDEO_FIRST_ID = -3
_VIDEO_ID = -2
_VIDEO_LAST_ID = -4
_PAD_ID = -1
_TEXT_ID = -5
_AUDIO_FIRST_ID = -15
_AUDIO_ID = -14
_IMGVID_COND_ID = -11

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_SEQ_ALIGN = 64
_PATCH_H, _PATCH_W = 2, 2


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Compute 1-D RoPE position grid for one spatial axis."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    """Temporal RoPE position grid for n video latent frames."""
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(temporal_length: int) -> float:
    """Total temporal span (in RoPE units) for temporal_length frames."""
    spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
    for token_index in range(_T_GROUP):
        spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def build_packed_fl2va(
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    keyframe_indices: list[int],
    audio_channel: int = 2,
    text_token_tags: torch.Tensor | None = None,
) -> dict:
    """Build packed sequence for FL2VA layout.

    Args:
        text_len: Number of text prompt tokens.
        latent_t: Video latent temporal dim (after VAE).
        latent_h: Video latent height (after VAE).
        latent_w: Video latent width (after VAE).
        audio_t: Audio latent temporal dim.
        keyframe_indices: List of keyframe indices, e.g. [0, -1].
        audio_channel: Audio channel count (default 2).
        text_token_tags: Presentation tags for the text block (text=1, vision
            tokens=0); overwrites the text region of token_tags.

    Returns:
        Packed dict with all position IDs, token tags, and sequence metadata.
    """
    ph = latent_h // _PATCH_H
    pw = latent_w // _PATCH_W
    frame_rows = ph * pw
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    num_keyframes = len(keyframe_indices)
    cond_rows = num_keyframes * frame_rows
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = ((used + _SEQ_ALIGN - 1) // _SEQ_ALIGN) * _SEQ_ALIGN

    # Slice ranges
    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

    # input_ids (for token_tags, not actual token IDs)
    input_ids = torch.full((seq_len,), _PAD_ID, dtype=torch.int64)
    input_ids[text_sl] = _TEXT_ID
    input_ids[cond_sl] = _IMGVID_COND_ID
    input_ids[audio_sl] = _AUDIO_ID
    input_ids[audio_sl.start] = _AUDIO_FIRST_ID
    input_ids[video_sl] = _VIDEO_ID
    input_ids[video_sl.start] = _VIDEO_FIRST_ID
    input_ids[video_sl.stop - 1] = _VIDEO_LAST_ID

    # img_pos covers both cond AND video rows
    img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
    # update_mask: True for video rows (to be predicted), False for cond rows (fixed)
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True

    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    text_pos = torch.arange(0, text_len)

    # 3-axis RoPE position grid
    g = torch.zeros(seq_len, 3, dtype=torch.float64)
    g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    # Condition rows: temporal position matches keyframe index
    t_grid_video = _video_t_grid(latent_t, float(text_len))
    temporal_span = _temporal_position_span(latent_t)
    for i, idx in enumerate(keyframe_indices):
        sl = slice(i * frame_rows, (i + 1) * frame_rows)
        if idx == 0:
            cond_t = float(text_len)
        else:  # idx == -1
            cond_t = float(text_len) + temporal_span - _FRAME_RESCALE
        cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
        cond_g[:, 0] = cond_t
        cond_g[:, 1:] = frame
        g[cond_sl.start + sl.start : cond_sl.start + sl.stop] = cond_g

    # Video target rows
    video_g = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    video_g[:, :, 0] = t_grid_video[:, None]
    video_g[:, :, 1:] = frame[None]
    g[video_sl] = video_g.reshape(-1, 3)

    # Audio rows: temporal + spatial (W-axis for channel separation)
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    g[audio_sl, 2] = torch.cat(
        [
            torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
            torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
        ]
    )

    # Token tags
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1  # text
    token_tags[audio_sl] = 2  # audio
    token_tags[img_pos] = 0  # video + cond
    # Presentation tags overwrite the text block (vision tokens inside it are
    # tagged 0).
    if text_token_tags is not None:
        token_tags[text_pos] = text_token_tags.long()

    cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

    return {
        "seq_len": int(seq_len),
        "img_pos": img_pos.to(torch.long),
        "audio_pos": audio_pos.to(torch.long),
        "text_pos": text_pos.to(torch.long),
        "update_mask": update_mask,
        "img_position_ids": g.unsqueeze(0),  # [1, seq_len, 3] fp64
        "token_tags": token_tags,
        "cu_seqlens": cu,
        "text_len": int(text_len),
        "audio_channel": audio_channel,
        "audio_t": audio_t,
        "latent_t": latent_t,
        "latent_h_patched": ph,
        "latent_w_patched": pw,
        "cond_rows": cond_rows,
    }
