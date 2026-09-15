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
# See the License for the specific language governing limitations
# under the License.

"""Flux models consume tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.models.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.refs import flux as ref_flux
from tests.models.tiny_configs import tiny_flux_config as _tiny_config


def _build_ours_rms(dim: int, *, elementwise_affine: bool = True, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.flux.modeling_flux import RMSNorm

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return RMSNorm(dim, eps=1e-6, elementwise_affine=elementwise_affine)


def _build_ours_model():
    from veomni.models.transformers.flux.modeling_flux import FluxModel

    with ops_config_scope(eager_ops_config()):
        return FluxModel(_tiny_config())


def _rotary_embedding(seq_len: int, head_dim: int):
    pos = torch.arange(seq_len, dtype=torch.float32)
    half = torch.arange(head_dim // 2, dtype=torch.float32)
    angle = torch.outer(pos, half)
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, 1, seq_len, head_dim // 2, 2, 2)


def test_flux_config_defaults_preserve_production_shapes():
    from veomni.models.transformers.flux.config_flux import FluxConfig

    config = FluxConfig()
    assert config.num_attention_heads * config.attention_head_dim == 3072
    assert config.num_blocks == 19
    assert config.num_single_layers == 38
    assert config.joint_attention_dim == 4096
    assert config.pooled_projection_dim == 768
    assert config.input_dim == config.output_dim == 64
    assert config.axes_dims_rope == [16, 56, 56]


def test_flux_registered_config_round_trip(tmp_path):
    from veomni.models import build_config
    from veomni.models.transformers.flux.config_flux import FluxConfig

    expected = _tiny_config()
    expected.save_pretrained(tmp_path)
    loaded = build_config(str(tmp_path))

    assert type(loaded) is FluxConfig
    assert loaded.architectures == ["FluxModel"]
    assert loaded.num_attention_heads == expected.num_attention_heads
    assert loaded.attention_head_dim == expected.attention_head_dim
    assert loaded.num_single_layers == expected.num_single_layers


def test_flux_repository_config_loads_through_registry():
    from veomni.models import build_config, get_model_class
    from veomni.models.transformers.flux.config_flux import FluxConfig

    config_path = Path(__file__).parents[3] / "configs/model_configs/flux/flux.json"
    config = build_config(str(config_path))

    assert type(config) is FluxConfig
    assert config.architectures == ["FluxModel"]
    assert get_model_class(config).__name__ == "FluxModel"


@pytest.mark.parametrize("elementwise_affine", [True, False], ids=["weighted", "unweighted"])
def test_flux_rmsnorm_matches_official(elementwise_affine):
    torch.manual_seed(0)
    official = ref_flux.RMSNorm(32, eps=1e-6, elementwise_affine=elementwise_affine)
    ours = _build_ours_rms(32, elementwise_affine=elementwise_affine)
    assert ours.veomni_rms_norm.variant == ("standard" if elementwise_affine else "unweighted")
    if elementwise_affine:
        with torch.no_grad():
            official.weight.uniform_(0.5, 1.5)
            ours.weight.copy_(official.weight)
    hidden = torch.randn(2, 4, 32)
    official_input = hidden.clone().requires_grad_()
    ours_input = hidden.clone().requires_grad_()

    def call(module):
        return module(ours_input if module is ours else official_input)

    assert_outputs_and_grads_match(official, ours, call)
    torch.testing.assert_close(ours_input.grad, official_input.grad)


def test_flux_joint_attention_matches_official():
    torch.manual_seed(0)
    from veomni.models.transformers.flux import modeling_flux as ours_flux

    # FA2/FA3 are CUDA-only. Pin the models copy to SDPA so it matches
    # the adapted CPU snapshot.
    ours_flags = (ours_flux.FLASH_ATTN_2_AVAILABLE, ours_flux.FLASH_ATTN_3_AVAILABLE)
    ours_flux.FLASH_ATTN_2_AVAILABLE = False
    ours_flux.FLASH_ATTN_3_AVAILABLE = False

    dim = 64
    num_heads = 4
    head_dim = dim // num_heads
    official = ref_flux.FluxJointAttention(dim, dim, num_heads, head_dim)

    with ops_config_scope(eager_ops_config()):
        ours = ours_flux.FluxJointAttention(dim, dim, num_heads, head_dim)
    ours.load_state_dict(official.state_dict())

    hidden_a = torch.randn(2, 4, dim)
    hidden_b = torch.randn(2, 3, dim)
    ids = torch.zeros(2, 7, 3)
    ids[..., 0] = torch.arange(7)
    ids[..., 1] = torch.arange(7)
    ids[..., 2] = torch.arange(7)
    image_rotary_emb = ref_flux.RoPEEmbedding(dim, 10000, [8, 4, 4])(ids)

    def call(module):
        return module(hidden_a, hidden_b, image_rotary_emb)

    try:
        assert_outputs_and_grads_match(official, ours, call)
    finally:
        ours_flux.FLASH_ATTN_2_AVAILABLE, ours_flux.FLASH_ATTN_3_AVAILABLE = ours_flags


def test_flux_tiny_model_forward_backward_smoke():
    """Check end-to-end connectivity, not numerical parity of the whole backbone."""
    from veomni.models.transformers.flux import modeling_flux

    flags = (modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE)
    modeling_flux.FLASH_ATTN_2_AVAILABLE = False
    modeling_flux.FLASH_ATTN_3_AVAILABLE = False
    try:
        torch.manual_seed(0)
        model = _build_ours_model()
        hidden_states = torch.randn(2, 4, 4, 6)
        timestep = torch.rand(2)
        prompt_emb = torch.randn(2, 3, 48)
        pooled_prompt_emb = torch.randn(2, 32)
        guidance = torch.rand(2)
        text_ids = torch.zeros(2, 3, 3)

        output = model(
            hidden_states,
            timestep,
            prompt_emb,
            pooled_prompt_emb,
            guidance,
            text_ids,
        )
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

        output.square().mean().backward()
        assert model.x_embedder.weight.grad is not None
        assert model.blocks[0].attn.a_to_qkv.weight.grad is not None
        assert model.single_blocks[0].to_qkv_mlp.weight.grad is not None
    finally:
        modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE = flags


def test_flux_patch_layout_and_image_coordinates():
    model = _build_ours_model()
    image = torch.arange(2 * 4 * 4 * 6, dtype=torch.float32).reshape(2, 4, 4, 6).requires_grad_()
    # Enumerate 2x2 tiles in row-major order, flattening channel before local pixel.
    expected = torch.stack(
        [image[:, :, row : row + 2, col : col + 2].flatten(1) for row in (0, 2) for col in (0, 2, 4)], dim=1
    )
    patches = model.patchify(image)
    torch.testing.assert_close(patches, expected, rtol=0, atol=0)
    weights = torch.linspace(-1, 2, patches.numel()).reshape_as(patches)
    actual_grad = torch.autograd.grad((patches * weights).sum(), image)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), image)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    # A separate token tensor prevents mutually wrong pack/unpack functions from cancelling.
    tokens = torch.linspace(-2, 3, patches.numel()).reshape_as(patches).requires_grad_()
    reconstructed = model.unpatchify(tokens, height=4, width=6)
    expected_image = F.fold(tokens.transpose(1, 2), output_size=(4, 6), kernel_size=2, stride=2)
    torch.testing.assert_close(reconstructed, expected_image, rtol=0, atol=0)
    image_weights = torch.arange(image.numel()).reshape_as(image)
    actual_grad = torch.autograd.grad((reconstructed * image_weights).sum(), tokens)[0]
    expected_grad = torch.autograd.grad((expected_image * image_weights).sum(), tokens)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    expected_ids = image.new_tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
    torch.testing.assert_close(model.prepare_image_ids(image), expected_ids.expand(2, -1, -1), rtol=0, atol=0)


def test_flux_final_conditioning_matches_explicit_formula():
    from veomni.models.transformers.flux.modeling_flux import AdaLayerNormContinuous

    torch.manual_seed(0)
    module = AdaLayerNormContinuous(8)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    condition = torch.randn(2, 8, requires_grad=True)
    actual = module(hidden, condition)
    modulation = F.linear(condition * condition.sigmoid(), module.linear.weight, module.linear.bias)
    scale, shift = modulation[:, :8], modulation[:, 8:]
    normalized = (hidden - hidden.mean(-1, keepdim=True)) * torch.rsqrt(
        hidden.var(-1, unbiased=False, keepdim=True) + 1e-6
    )
    expected = normalized * (1 + scale[:, None]) + shift[:, None]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    probe = torch.randn_like(actual)
    inputs = (hidden, condition, module.linear.weight, module.linear.bias)
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)


def test_flux_joint_attention_is_non_causal():
    from veomni.models.transformers.flux import modeling_flux

    flags = (modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE)
    modeling_flux.FLASH_ATTN_2_AVAILABLE = False
    modeling_flux.FLASH_ATTN_3_AVAILABLE = False
    try:
        with ops_config_scope(eager_ops_config()):
            block = modeling_flux.FluxJointAttention(32, 32, 4, 8).eval()

        text = torch.randn(1, 1, 32)
        image = torch.randn(1, 2, 32)
        rotary = _rotary_embedding(3, 8)
        with torch.no_grad():
            _, before = block(image, text, rotary)
            image[:, -1].add_(1.0)
            _, after = block(image, text, rotary)
        assert not torch.allclose(before[0, 0], after[0, 0], atol=1e-5)
    finally:
        modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE = flags


def test_flux_single_transformer_block_is_non_causal():
    from veomni.models.transformers.flux import modeling_flux

    flags = (modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE)
    modeling_flux.FLASH_ATTN_2_AVAILABLE = False
    modeling_flux.FLASH_ATTN_3_AVAILABLE = False
    try:
        with ops_config_scope(eager_ops_config()):
            block = modeling_flux.FluxSingleTransformerBlock(32, 4).eval()

        hidden_states = torch.randn(1, 3, 3 * 32)
        rotary = _rotary_embedding(3, 8)
        with torch.no_grad():
            before = block.process_attention(hidden_states, rotary)
            hidden_states[:, -1].add_(1.0)
            after = block.process_attention(hidden_states, rotary)
        assert not torch.allclose(before[0, 0], after[0, 0], atol=1e-5)
    finally:
        modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE = flags


def test_flux_diffusers_converter_preserves_projection_layout():
    from veomni.models.transformers.flux.utils_flux import FluxDiTStateDictConverter

    joint_q = torch.full((4, 4), 1.0)
    joint_k = torch.full((4, 4), 2.0)
    joint_v = torch.full((4, 4), 3.0)
    single_q = torch.full((4, 4), 4.0)
    single_k = torch.full((4, 4), 5.0)
    single_v = torch.full((4, 4), 6.0)
    single_mlp = torch.full((16, 4), 7.0)
    converted = FluxDiTStateDictConverter().from_diffusers(
        {
            "transformer_blocks.0.attn.to_q.weight": joint_q,
            "transformer_blocks.0.attn.to_k.weight": joint_k,
            "transformer_blocks.0.attn.to_v.weight": joint_v,
            "single_transformer_blocks.0.attn.to_q.weight": single_q,
            "single_transformer_blocks.0.attn.to_k.weight": single_k,
            "single_transformer_blocks.0.attn.to_v.weight": single_v,
            "single_transformer_blocks.0.proj_mlp.weight": single_mlp,
        }
    )

    assert torch.equal(converted["blocks.0.attn.a_to_qkv.weight"], torch.cat([joint_q, joint_k, joint_v]))
    assert torch.equal(
        converted["single_blocks.0.to_qkv_mlp.weight"],
        torch.cat([single_q, single_k, single_v, single_mlp]),
    )
