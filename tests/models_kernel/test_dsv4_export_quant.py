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

"""DSV4 export quantization must reject non-SM90 before importing TileLang."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch


_EXPORT_QUANT = "veomni.models_kernel.transformers.deepseek_v4.export_quant"
_ACT_QUANT = "veomni.models_kernel.transformers.deepseek_v4.act_quant"


def test_export_quant_does_not_import_tilelang_eagerly():
    sys.modules.pop(_EXPORT_QUANT, None)
    sys.modules.pop(_ACT_QUANT, None)
    before = "tilelang" in sys.modules

    importlib.import_module(_EXPORT_QUANT)

    assert ("tilelang" in sys.modules) is before
    assert _ACT_QUANT not in sys.modules


def test_export_wrappers_reject_pre_sm90_before_import(monkeypatch):
    sys.modules.pop(_ACT_QUANT, None)
    import veomni.models_kernel.transformers.deepseek_v4.export_quant as quant

    monkeypatch.setattr(quant, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(quant, "get_gpu_compute_capability", lambda: 89)

    with pytest.raises(RuntimeError, match="SM90 or later"):
        quant.fp4_act_quant(torch.empty(0))
    with pytest.raises(RuntimeError, match="SM90 or later"):
        quant.fp8_weight_quant(torch.empty(0))
    assert _ACT_QUANT not in sys.modules


def test_export_wrappers_reject_rocm_before_import(monkeypatch):
    sys.modules.pop(_ACT_QUANT, None)
    import veomni.models_kernel.transformers.deepseek_v4.export_quant as quant

    monkeypatch.setattr(quant.torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(quant, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(quant, "get_gpu_compute_capability", lambda: 90)

    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.fp4_act_quant(torch.empty(0))
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        quant.fp8_weight_quant(torch.empty(0))
    assert _ACT_QUANT not in sys.modules
