"""Offloading the SeedVR2 VAE must release its causal-convolution caches."""

import pytest
import torch

from veomni.models.diffusers.seedvr2.conditioning_seedvr2 import SeedVR2ConditionConfig, SeedVR2ConditionModel
from veomni.models.diffusers.seedvr2.core.vae.attn_video_vae import VideoAutoencoderKLWrapper
from veomni.models.diffusers.seedvr2.core.vae.causal_inflation_lib import InflatedCausalConv3d
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


def build_vae(device):
    """The smallest sliced VAE: `split_size=4` cuts the nine frames used below in two."""
    torch.manual_seed(0)
    vae = VideoAutoencoderKLWrapper(
        spatial_downsample_factor=8,
        temporal_downsample_factor=4,
        freeze_encoder=True,
        act_fn="silu",
        block_out_channels=(8, 8, 8),
        down_block_types=("DownEncoderBlock3D",) * 3,
        up_block_types=("UpDecoderBlock3D",) * 3,
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=4,
        slicing_sample_min_size=4,
        temporal_scale_num=2,
        inflation_mode="pad",
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    vae.requires_grad_(False).eval()
    vae.set_causal_slicing(split_size=4, memory_device="same")
    vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
    return vae.to(device)


def causal_caches(vae):
    """The tensors `InflatedCausalConv3d.memory` keeps outside the module state."""
    return [
        module.memory
        for module in vae.modules()
        if isinstance(module, InflatedCausalConv3d) and module.memory is not None
    ]


def cached_bytes(vae, device):
    caches = causal_caches(vae)
    assert caches, "a sliced call is expected to cache causal-convolution state"
    assert all(cache.device.type == device.type for cache in caches)
    return sum(cache.numel() * cache.element_size() for cache in caches)


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="CUDA or NPU is required to observe live accelerator memory",
)
def test_seedvr2_vae_offload_releases_causal_caches():
    device = torch.device(get_device_type())
    device_api = get_torch_device()
    clip = torch.rand(3, 9, 64, 64, device=device)
    assert clip.shape[1] > 5, "the clip must be long enough for the VAE to be sliced"
    vae = build_vae(device)

    # The caches are invisible to `Module.to`, so each stage is offloaded twice - once with
    # the caches still held and once after clearing them. The two offloads differ only by the
    # cached bytes, which keeps the comparison independent of the parameter footprint.
    torch.manual_seed(0)
    latent = vae.encode(clip.unsqueeze(0)).latent.detach()
    encode_cache = cached_bytes(vae, device)

    vae.to("cpu")
    device_api.synchronize()
    leaky_offload = device_api.memory_allocated()
    device_api.reset_peak_memory_stats()
    scratch = torch.empty(1 << 18, dtype=torch.float32, device=device)  # a later stage's working set
    device_api.synchronize()
    leaky_peak = device_api.max_memory_allocated()
    del scratch
    device_api.empty_cache()

    vae.to(device)
    torch.manual_seed(0)
    # Accelerator kernels are not bit-reproducible across offload cycles, so the result
    # comparisons below carry a tolerance; the cache accounting stays exact.
    torch.testing.assert_close(vae.encode(clip.unsqueeze(0)).latent, latent, atol=2e-3, rtol=1e-3)
    assert cached_bytes(vae, device) == encode_cache
    vae.clear_causal_cache()
    vae.to("cpu")
    device_api.synchronize()
    clean_offload = device_api.memory_allocated()
    device_api.reset_peak_memory_stats()
    scratch = torch.empty(1 << 18, dtype=torch.float32, device=device)
    device_api.synchronize()
    clean_peak = device_api.max_memory_allocated()
    del scratch
    assert not causal_caches(vae)
    assert leaky_offload - clean_offload >= encode_cache
    assert leaky_peak - clean_peak >= encode_cache

    # Decoding caches as well, and its offload has to release them too.
    vae.to(device)
    decoded = vae.decode(latent).sample.detach()
    decode_cache = cached_bytes(vae, device)
    vae.to("cpu")
    device_api.synchronize()
    leaky_decode = device_api.memory_allocated()

    vae.to(device)
    torch.testing.assert_close(vae.decode(latent).sample, decoded, atol=2e-3, rtol=1e-3)
    assert cached_bytes(vae, device) == decode_cache
    vae.clear_causal_cache()
    vae.to("cpu")
    device_api.synchronize()
    clean_decode = device_api.memory_allocated()
    assert not causal_caches(vae)
    assert leaky_decode - clean_decode >= decode_cache

    # The restorer reaches the caches through the conditioning model that owns the VAE.
    condition = SeedVR2ConditionModel(SeedVR2ConditionConfig(), meta_init=True)
    condition.vae = vae
    vae.to(device)
    vae.decode(latent)
    assert causal_caches(vae)
    condition.clear_vae_cache()
    assert not causal_caches(vae)
