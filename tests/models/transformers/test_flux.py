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

import torch

from tests.models.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
)
from tests.models.refs import flux as ref_flux
from tests.models.tiny_configs import tiny_flux_config as _tiny_config
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _build_ours_rms(dim: int, *, elementwise_affine: bool = True, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.flux.modeling_flux import RMSNorm

    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return RMSNorm(dim, eps=1e-6, elementwise_affine=elementwise_affine)
    finally:
        set_ops_config(previous)


def _build_ours_model():
    from veomni.models.transformers.flux.modeling_flux import FluxModel

    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        return FluxModel(_tiny_config())
    finally:
        set_ops_config(previous)


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


def test_flux_constructs_local_kernels():
    weighted = _build_ours_rms(32)
    unweighted = _build_ours_rms(32, elementwise_affine=False)
    assert isinstance(weighted.veomni_rms_norm, VeomniOp)
    assert weighted.veomni_rms_norm.op == "rms_norm"
    assert weighted.veomni_rms_norm.variant == "standard"
    assert weighted.veomni_rms_norm.impl == "eager"
    assert unweighted.veomni_rms_norm.variant == "unweighted"


def test_flux_instances_keep_distinct_impls():
    eager = _build_ours_rms(32, ops=eager_ops_config())
    other_cfg = eager_ops_config()
    other_cfg.rms_norm_implementation = "liger_kernel"
    other = _build_ours_rms(32, ops=other_cfg)

    assert eager.veomni_rms_norm.impl == "eager"
    assert other.veomni_rms_norm.impl == "liger_kernel"

    set_ops_config(other_cfg)
    assert eager.veomni_rms_norm.impl == "eager"


def test_flux_rmsnorm_matches_official():
    torch.manual_seed(0)
    official = ref_flux.RMSNorm(32, eps=1e-6)
    ours = _build_ours_rms(32)
    ours.weight.data.copy_(official.weight.data)
    hidden = torch.randn(2, 4, 32)

    def call(module):
        return module(hidden)

    assert_outputs_and_grads_match(official, ours, call)


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

    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        ours = ours_flux.FluxJointAttention(dim, dim, num_heads, head_dim)
    finally:
        set_ops_config(previous)
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


def test_flux_tiny_model_forward_backward():
    from veomni.models.transformers.flux import modeling_flux

    flags = (modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE)
    modeling_flux.FLASH_ATTN_2_AVAILABLE = False
    modeling_flux.FLASH_ATTN_3_AVAILABLE = False
    try:
        model = _build_ours_model()
        hidden_states = torch.randn(2, 4, 4, 4)
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

        output.square().mean().backward()
        assert model.x_embedder.weight.grad is not None
        assert model.blocks[0].attn.a_to_qkv.weight.grad is not None
        assert model.single_blocks[0].to_qkv_mlp.weight.grad is not None
    finally:
        modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE = flags


def test_flux_joint_attention_is_non_causal():
    from veomni.models.transformers.flux import modeling_flux

    flags = (modeling_flux.FLASH_ATTN_2_AVAILABLE, modeling_flux.FLASH_ATTN_3_AVAILABLE)
    modeling_flux.FLASH_ATTN_2_AVAILABLE = False
    modeling_flux.FLASH_ATTN_3_AVAILABLE = False
    try:
        previous = get_ops_config()
        set_ops_config(eager_ops_config())
        try:
            block = modeling_flux.FluxJointAttention(32, 32, 4, 8).eval()
        finally:
            set_ops_config(previous)

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
        previous = get_ops_config()
        set_ops_config(eager_ops_config())
        try:
            block = modeling_flux.FluxSingleTransformerBlock(32, 4).eval()
        finally:
            set_ops_config(previous)

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
