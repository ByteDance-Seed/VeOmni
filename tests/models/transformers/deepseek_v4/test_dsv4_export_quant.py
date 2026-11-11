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

"""DSV4 export quantization must reject unsupported GPUs before importing a quantizer backend."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

_EXPORT_QUANT = "veomni.models.transformers.deepseek_v4.export_quant"
_ACT_QUANT = "veomni.models.transformers.deepseek_v4.act_quant"


def test_export_quant_does_not_import_tilelang_eagerly():
    sys.modules.pop(_EXPORT_QUANT, None)
    sys.modules.pop(_ACT_QUANT, None)
    before = "tilelang" in sys.modules

    importlib.import_module(_EXPORT_QUANT)

    assert ("tilelang" in sys.modules) is before
    assert _ACT_QUANT not in sys.modules


def test_export_wrappers_reject_pre_sm90_before_import(monkeypatch):
    sys.modules.pop(_ACT_QUANT, None)
    import veomni.models.transformers.deepseek_v4.export_quant as quant

    error = RuntimeError("requires an SM90 or later NVIDIA CUDA GPU")
    monkeypatch.setattr(quant, "require_tilelang_sm90", lambda device=None: (_ for _ in ()).throw(error))
    monkeypatch.setattr(quant, "fp8_quantizer_backend", lambda device=None: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match="SM90 or later"):
        quant.fp4_act_quant(torch.empty(0))
    with pytest.raises(RuntimeError, match="SM90 or later"):
        quant.fp8_weight_quant(torch.empty(0))
    assert _ACT_QUANT not in sys.modules


def test_export_wrappers_reject_unsupported_rocm_before_import(monkeypatch):
    sys.modules.pop(_ACT_QUANT, None)
    import veomni.models.transformers.deepseek_v4.export_quant as quant

    error = RuntimeError("requires an SM90 or later NVIDIA CUDA GPU or an AMD ROCm gfx942/gfx950 GPU")
    monkeypatch.setattr(quant, "require_tilelang_sm90", lambda device=None: (_ for _ in ()).throw(error))
    monkeypatch.setattr(quant, "fp8_quantizer_backend", lambda device=None: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.fp4_act_quant(torch.empty(0))
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.fp8_weight_quant(torch.empty(0))
    assert _ACT_QUANT not in sys.modules


def test_export_fp8_runs_triton_on_gfx942_and_fp4_stays_nvidia_only(monkeypatch):
    import veomni.models.transformers.deepseek_v4.export_quant as quant

    calls = []
    backend = ModuleType("veomni.ops.qat.triton_quant")
    backend.fp8_weight_quant = lambda x, *args: calls.append(args) or (x, x)
    monkeypatch.setitem(sys.modules, "veomni.ops.qat.triton_quant", backend)
    monkeypatch.setattr(quant, "fp8_quantizer_backend", lambda device=None: "triton")
    error = RuntimeError("requires an SM90 or later NVIDIA CUDA GPU")
    monkeypatch.setattr(quant, "require_tilelang_sm90", lambda device=None: (_ for _ in ()).throw(error))

    quant.fp8_weight_quant(torch.empty(0), 128, "ue8m0", torch.float8_e8m0fnu)
    assert calls == [(128, "ue8m0", torch.float8_e8m0fnu)]
    quant.require_export_quantizers(fp4=False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.require_export_quantizers(fp4=True)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.fp4_act_quant(torch.empty(0))
