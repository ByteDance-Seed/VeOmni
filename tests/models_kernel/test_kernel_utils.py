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

"""CPU tests for models_kernel construct helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.utils.kernel_utils import linear_bias, resolve_kernel_impl


@pytest.fixture(autouse=True)
def _restore_kernels_config():
    previous = get_kernels_config()
    yield
    set_kernels_config(previous)


def test_resolve_kernel_impl_defaults_to_eager():
    set_kernels_config(None)
    assert resolve_kernel_impl("rms_norm_implementation") == "eager"


def test_resolve_kernel_impl_reads_kernels_config():
    set_kernels_config(SimpleNamespace(cross_entropy_loss_implementation="chunk_loss"))
    assert resolve_kernel_impl("cross_entropy_loss_implementation") == "chunk_loss"


def test_resolve_kernel_impl_remaps_npu_ce_alias():
    set_kernels_config(SimpleNamespace(cross_entropy_loss_implementation="npu"))
    assert resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss") == "chunk_loss"
    assert resolve_kernel_impl("cross_entropy_loss_implementation") == "npu"


def test_linear_bias_empty_sentinel():
    linear = torch.nn.Linear(4, 4, bias=False)
    bias = linear_bias(linear)
    assert bias.numel() == 0
    assert bias.device == linear.weight.device
    assert bias.dtype == linear.weight.dtype
