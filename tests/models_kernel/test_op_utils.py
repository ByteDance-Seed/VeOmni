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

"""CPU tests for models_kernel op-construction helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.models_kernel.utils.op_utils import attention_op, linear_bias, resolve_op_impl
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


@pytest.fixture(autouse=True)
def _restore_ops_config():
    previous = get_ops_config()
    yield
    set_ops_config(previous)


def test_resolve_op_impl_defaults_to_eager():
    set_ops_config(None)
    assert resolve_op_impl("rms_norm_implementation") == "eager"


def test_resolve_op_impl_reads_ops_config():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"))
    assert resolve_op_impl("cross_entropy_loss_implementation") == "chunk_loss"


def test_resolve_op_impl_remaps_npu_ce_alias():
    set_ops_config(SimpleNamespace(cross_entropy_loss_implementation="npu"))
    assert resolve_op_impl("cross_entropy_loss_implementation", npu_as="chunk_loss") == "chunk_loss"
    assert resolve_op_impl("cross_entropy_loss_implementation") == "npu"


def test_linear_bias_empty_sentinel():
    linear = torch.nn.Linear(4, 4, bias=False)
    bias = linear_bias(linear)
    assert bias.numel() == 0
    assert bias.device == linear.weight.device
    assert bias.dtype == linear.weight.dtype


def test_attention_op_defaults_to_eager():
    set_ops_config(None)
    op = attention_op()
    assert isinstance(op, VeomniOp)
    assert op.op == "attention"
    assert op.variant == "standard"
    assert op.impl == "eager"


def test_attention_op_reads_ops_config():
    set_ops_config(SimpleNamespace(attn_implementation="veomni_flash_attention_2"))
    op = attention_op()
    assert op.impl == "veomni_flash_attention_2"
    assert attention_op() is op


def test_resolve_moe_impl_reads_ops_config():
    from veomni.models_kernel.utils.op_utils import resolve_moe_impl

    set_ops_config(SimpleNamespace(moe_implementation="fused_triton"))
    assert resolve_moe_impl() == "fused_triton"
    set_ops_config(SimpleNamespace(moe_implementation="eager"))
    assert resolve_moe_impl() == "eager"
