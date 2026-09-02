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

"""LTX 2.3 models_kernel consume tests.

Direct-import staged helpers. Do not register or use
``build_foundation_model``. Compare ``rms_norm`` against official
``torch.nn.functional.rms_norm``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from tests.models_kernel.compare import eager_kernels_config
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _call_rms(x: torch.Tensor, weight: torch.Tensor | None, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.diffusers.ltx2_3.ltx_core.utils import rms_norm

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return rms_norm(x, weight=weight, eps=1e-6)
    finally:
        set_kernels_config(previous)


def test_ltx2_3_rms_norm_matches_official():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    official_x = x.detach().clone().requires_grad_(True)
    official_weight = weight.detach().clone().requires_grad_(True)

    ours = _call_rms(x, weight)
    official = F.rms_norm(official_x, (official_x.shape[-1],), weight=official_weight, eps=1e-6)
    torch.testing.assert_close(ours, official, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    ours.sum().backward()
    official.sum().backward()
    torch.testing.assert_close(x.grad, official_x.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(weight.grad, official_weight.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_ltx2_3_unweighted_rms_norm_matches_official():
    torch.manual_seed(1)
    x = torch.randn(2, 8, 16, requires_grad=True)
    official_x = x.detach().clone().requires_grad_(True)

    ours = _call_rms(x, None)
    official = F.rms_norm(official_x, (official_x.shape[-1],), weight=None, eps=1e-6)
    torch.testing.assert_close(ours, official, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    ours.sum().backward()
    official.sum().backward()
    torch.testing.assert_close(x.grad, official_x.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
