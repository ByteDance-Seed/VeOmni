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

"""LTX 2.3 models_kernel consume tests."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
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


def test_from_pretrained_does_not_point_at_models_copy():
    from veomni.models_kernel.diffusers.ltx2_3.ltx_transformer.modeling_ltx2_3_transformer import (
        LTXVideoTransformerModel,
    )

    with pytest.raises(NotImplementedError, match="not supported on this wrapper") as excinfo:
        LTXVideoTransformerModel.from_pretrained("unused")
    assert "models/" not in str(excinfo.value)


def test_ltx_core_rebinds_away_from_another_copy(tmp_path):
    fake_root = tmp_path / "fake_ltx"
    fake_pkg = fake_root / "ltx_core"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("")
    (fake_pkg / "utils.py").write_text("MARKER = 'fake'\n")

    saved_path = list(sys.path)
    saved_modules = {
        name: sys.modules[name] for name in list(sys.modules) if name == "ltx_core" or name.startswith("ltx_core.")
    }
    sys.path.insert(0, str(fake_root))
    try:
        for name in list(saved_modules):
            sys.modules.pop(name, None)
        fake_utils = importlib.import_module("ltx_core.utils")
        assert fake_utils.MARKER == "fake"

        binder = importlib.import_module("veomni.models_kernel.diffusers.ltx2_3.ltx_core")
        importlib.reload(binder)
        bound_utils = importlib.import_module("ltx_core.utils")
        package_utils = importlib.import_module("veomni.models_kernel.diffusers.ltx2_3.ltx_core.utils")
        assert bound_utils.__file__ == package_utils.__file__
        assert "VeomniKernel" in bound_utils.rms_norm.__doc__
        assert not hasattr(bound_utils, "MARKER")
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "ltx_core" or name.startswith("ltx_core."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
