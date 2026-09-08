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

"""Flux models_kernel consume tests.

Direct-import the staged classes. Compare RMSNorm and a tiny joint-attention
block against ``tests/models_kernel/refs/flux.py``. Full ``FluxModel``
stays hardcoded at 3072-d / 19+38 blocks.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models_kernel.compare import (
    assert_outputs_and_grads_match,
    eager_ops_config,
)
from tests.models_kernel.refs import flux as ref_flux
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _build_ours_rms(dim: int, *, elementwise_affine: bool = True, ops: SimpleNamespace | None = None):
    from veomni.models_kernel.transformers.flux.modeling_flux import RMSNorm

    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return RMSNorm(dim, eps=1e-6, elementwise_affine=elementwise_affine)
    finally:
        set_ops_config(previous)


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
    from veomni.models_kernel.transformers.flux import modeling_flux as ours_flux

    # FA2/FA3 are CUDA-only. Pin the models_kernel copy to SDPA so it matches
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
