# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.models.tiny_configs import tiny_movqgan_config as _tiny_config


def test_movqgan_repository_config_and_processor_load_through_registry():
    from veomni.models import build_config, build_processor, get_model_class
    from veomni.models.transformers.movqgan.configuration_movqgan import MoVQGANConfig
    from veomni.models.transformers.movqgan.processing_movqgan import MoVQGANProcessor

    config_path = Path(__file__).parents[2] / "toy_config/movqgan_toy"
    config = build_config(str(config_path))
    processor = build_processor(str(config_path))

    assert type(config) is MoVQGANConfig
    assert get_model_class(config).__name__ == "MoVQGAN"
    assert type(processor) is MoVQGANProcessor

    image = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(12, 8, 3)
    features = processor(images=image, return_tensors="pt")["features"]
    assert features.shape == (1, 3, 64, 64)
    assert features.dtype == torch.float32
    assert features.min() >= -1
    assert features.max() <= 1


def test_movqgan_forward_backward_and_codebook_decode_smoke():
    """Check full-model connectivity and decode API agreement, not independent parity."""
    from veomni.models.transformers.movqgan.modeling_movqgan import MoVQGAN

    torch.manual_seed(0)
    model = MoVQGAN(_tiny_config()).eval()
    features = torch.randn(1, 3, 8, 8, requires_grad=True)

    reconstructed, embedding_loss = model(features)
    assert reconstructed.shape == features.shape
    assert embedding_loss.ndim == 0

    (reconstructed.square().mean() + embedding_loss).backward()
    assert features.grad is not None
    assert model.encoder.conv_in.weight.grad is not None
    assert model.quantize.embedding.weight.grad is not None
    assert model.decoder.conv_out.weight.grad is not None

    with torch.no_grad():
        quantized, _, (_, indices) = model.encode(features.detach())
        decoded = model.decode(quantized)
        decoded_from_codes = model.decode_code(indices, shape=(1, 4, 4, model.config.embed_dim))
    torch.testing.assert_close(decoded_from_codes, decoded)


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy-beta", "commitment-beta"])
def test_movqgan_quantization_matches_explicit_codebook_and_gradients(legacy):
    from veomni.models.transformers.movqgan.modeling_movqgan import VectorQuantizer

    codebook = torch.tensor([[-2.0, -1.0], [0.0, 2.0], [3.0, -2.0], [4.0, 4.0], [9.0, 9.0]])
    indices = torch.tensor([0, 1, 2, 3, 1, 0, 3, 2, 1, 0, 2, 3])
    selected = codebook[indices]
    offsets = torch.linspace(-0.2, 0.3, selected.numel()).reshape_as(selected)
    latent = (selected + offsets).reshape(2, 2, 3, 2).permute(0, 3, 1, 2).contiguous().requires_grad_()
    beta = 0.37
    quantizer = VectorQuantizer(n_e=5, e_dim=2, beta=beta, legacy=legacy)
    with torch.no_grad():
        quantizer.embedding.weight.copy_(codebook)

    quantized, loss, (distances, actual_indices) = quantizer(latent)
    expected = selected.reshape(2, 2, 3, 2).permute(0, 3, 1, 2).contiguous()
    expected_distances = ((selected + offsets)[:, None] - codebook[None]).square().sum(-1)
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    torch.testing.assert_close(distances, expected_distances, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(quantized, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(loss, (1 + beta) * offsets.square().mean(), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(quantizer.get_codebook_entry(indices, shape=(2, 2, 3, 2)), expected, rtol=0, atol=0)

    # The decoder's arbitrary upstream gradient passes only to the encoder.
    # beta weights different sides of the VQ objective in the two supported modes.
    probe = torch.linspace(-0.7, 1.3, latent.numel()).reshape_as(latent)
    ((quantized * probe).sum() + loss).backward()
    encoder_scale, codebook_scale = (1.0, beta) if legacy else (beta, 1.0)
    expected_latent_grad = probe + 2 * encoder_scale * (latent.detach() - expected) / latent.numel()
    expected_codebook_grad = torch.zeros_like(codebook)
    expected_codebook_grad.index_add_(0, indices, -2 * codebook_scale * offsets / latent.numel())
    torch.testing.assert_close(latent.grad, expected_latent_grad, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(quantizer.embedding.weight.grad, expected_codebook_grad, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("add_conv", [False, True])
def test_movqgan_spatial_conditioning_matches_explicit_formula(add_conv):
    from veomni.models.transformers.movqgan.modeling_movqgan import SpatialNorm

    torch.manual_seed(0)
    module = SpatialNorm(4, 3, num_groups=2, eps=1e-6, affine=True, add_conv=add_conv)
    with torch.no_grad():
        module.norm_layer.weight.copy_(torch.tensor([0.7, 1.1, -0.3, 1.4]))
        module.norm_layer.bias.copy_(torch.tensor([0.2, -0.4, 0.1, 0.5]))
    features = torch.randn(2, 4, 4, 6, requires_grad=True)
    quantized = torch.randn(2, 3, 2, 3, requires_grad=True)
    actual = module(features, quantized)

    grouped = features.reshape(2, 2, -1)
    normalized = (grouped - grouped.mean(-1, keepdim=True)) * torch.rsqrt(
        grouped.var(-1, unbiased=False, keepdim=True) + 1e-6
    )
    normalized = normalized.reshape_as(features)
    normalized = (
        normalized * module.norm_layer.weight[None, :, None, None] + module.norm_layer.bias[None, :, None, None]
    )
    conditioning = quantized.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    if add_conv:
        conditioning = F.conv2d(conditioning, module.conv.weight, module.conv.bias, padding=1)
    gain = F.conv2d(conditioning, module.conv_y.weight, module.conv_y.bias)
    shift = F.conv2d(conditioning, module.conv_b.weight, module.conv_b.bias)
    expected = normalized * gain + shift
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    probe = torch.randn_like(actual)
    inputs = (features, quantized, *module.parameters())
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
