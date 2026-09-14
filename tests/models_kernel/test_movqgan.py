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
import torch

from tests.models_kernel.tiny_configs import tiny_movqgan_config as _tiny_config


def test_movqgan_repository_config_and_processor_load_through_registry():
    from veomni.models_kernel import build_config, build_processor, get_model_class
    from veomni.models_kernel.transformers.movqgan.configuration_movqgan import MoVQGANConfig
    from veomni.models_kernel.transformers.movqgan.processing_movqgan import MoVQGANProcessor

    config_path = Path(__file__).parents[1] / "toy_config/movqgan_toy"
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


def test_movqgan_forward_backward_and_codebook_decode():
    from veomni.models_kernel.transformers.movqgan.modeling_movqgan import MoVQGAN

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
