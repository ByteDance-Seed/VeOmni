"""SeedVR2 window attention must not leak text between packed samples."""

import torch

from veomni.models.diffusers.seedvr2.core.cache import Cache
from veomni.models.diffusers.seedvr2.core.nablocks.attention.mmattn import NaSwinAttention
from veomni.models.diffusers.seedvr2.core.normalization import get_norm_layer


def build_attention():
    torch.manual_seed(0)
    return NaSwinAttention(
        vid_dim=16,
        txt_dim=8,
        heads=2,
        head_dim=8,
        qk_bias=False,
        qk_norm=get_norm_layer("fusedrms"),
        qk_norm_eps=1e-5,
        rope_type="mmrope3d",
        rope_dim=8,
        shared_weights=False,
        window=(2, 2, 2),
        window_method="720pwin_by_size_bysize",
    ).eval()


def run(attention, samples):
    """Attention over one batch; each sample is (video tokens, video shape, text tokens)."""
    with torch.no_grad():
        return attention(
            torch.cat([video for video, _, _ in samples]),
            torch.cat([text for _, _, text in samples]),
            torch.tensor([shape for _, shape, _ in samples]),
            torch.tensor([[len(text)] for _, _, text in samples]),
            Cache(),
        )


def test_seedvr2_window_attention_matches_independent_samples():
    """Packing samples must not change any sample's window attention."""
    attention = build_attention()
    # 8x8x8 video tokens split into two windows, 1x8x8 into a single one, so the windows of
    # the first sample sit before the second sample's only window.
    samples = [
        (torch.randn(8 * 8 * 8, 16), (8, 8, 8), torch.randn(3, 8)),
        (torch.randn(8 * 8, 16), (1, 8, 8), torch.randn(5, 8)),
    ]

    packed_video, packed_text = run(attention, samples)
    independent = [run(attention, [sample]) for sample in samples]

    torch.testing.assert_close(packed_video, torch.cat([video for video, _ in independent]), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(packed_text, torch.cat([text for _, text in independent]), atol=1e-3, rtol=1e-3)
