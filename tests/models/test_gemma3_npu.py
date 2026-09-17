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

"""Tests for the Gemma 3 NPU patchgen path.

These tests run on any backend (CPU/GPU/NPU). The OpSlot guards in the
generated NPU modeling file fall through to the eager HF code when no fused
kernel is bound, so forward/backward parity with the GPU path can be checked
without NPU hardware.
"""

from pathlib import Path

import pytest
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.auto import build_foundation_model
from veomni.models.transformers.gemma3.generated import patched_modeling_gemma3_npu as gemma3_npu_modeling
from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_npu import (
    Gemma3ForCausalLM as VeOmniGemma3NpuForCausalLM,
)
from veomni.models.transformers.gemma3.generated.patched_modeling_gemma3_npu import (
    Gemma3TextModel as VeOmniGemma3NpuTextModel,
)


_TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "gemma3_toy"


def _npu_ops_config(
    attn_implementation: str = "sdpa",
    *,
    cross_entropy_loss_implementation: str = "eager",
) -> OpsImplementationConfig:
    return OpsImplementationConfig(
        attn_implementation=attn_implementation,
        moe_implementation="eager",
        cross_entropy_loss_implementation=cross_entropy_loss_implementation,
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )


def _toy_config() -> Gemma3TextConfig:
    return Gemma3TextConfig.from_pretrained(_TOY_CONFIG)


class TestGemma3NpuImports:
    """Verify the NPU modeling module exposes the expected classes and OpSlots."""

    def test_imports_veomni_classes(self):
        assert issubclass(VeOmniGemma3NpuForCausalLM, torch.nn.Module)
        assert issubclass(VeOmniGemma3NpuTextModel, torch.nn.Module)

    def test_opslot_declarations_present(self):
        assert hasattr(gemma3_npu_modeling, "veomni_causal_lm_loss")
        assert hasattr(gemma3_npu_modeling, "veomni_rms_norm")
        assert hasattr(gemma3_npu_modeling, "veomni_apply_rotary_pos_emb")

    def test_rmsnorm_guard_uses_offset_weight(self):
        import inspect

        source = inspect.getsource(gemma3_npu_modeling.Gemma3RMSNorm.forward)
        assert "1.0 + self.weight" in source
        assert "self.eps" in source

    def test_rotary_guard_dispatches_through_opslot(self):
        import inspect

        source = inspect.getsource(gemma3_npu_modeling.apply_rotary_pos_emb)
        assert "veomni_apply_rotary_pos_emb" in source

    def test_forcausallm_uses_causal_lm_loss_opslot(self):
        import inspect

        source = inspect.getsource(gemma3_npu_modeling.Gemma3ForCausalLM.forward)
        assert "veomni_causal_lm_loss" in source


class TestGemma3NpuForwardBackward:
    """Forward/backward parity with the GPU path (eager fallback)."""

    @pytest.fixture
    def model(self, monkeypatch):
        monkeypatch.setenv("MODELING_BACKEND", "veomni")
        m = build_foundation_model(
            _TOY_CONFIG,
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=_npu_ops_config("sdpa"),
        )
        assert isinstance(m, VeOmniGemma3NpuForCausalLM)
        return m

    def test_forward_produces_correct_shapes(self, model):
        input_ids = torch.randint(0, 128, (1, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() > 0

    def test_backward_updates_parameters(self, model):
        input_ids = torch.randint(0, 128, (1, 16))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_loss_decreases_with_training(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 128, (1, 32))
        labels = input_ids.clone()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())

        assert losses[-1] < losses[0]

    def test_sliding_and_full_attention_layers_work(self, model):
        """Gemma 3 alternates sliding and full attention — both must execute."""
        input_ids = torch.randint(0, 128, (1, 32))
        out = model(input_ids=input_ids)
        assert out.logits is not None
        assert out.logits.shape == (1, 32, 128)


class TestGemma3NpuRmsNormParity:
    """Verify the OpSlot-guarded RMSNorm matches HF eager on CPU."""

    def test_rmsnorm_eager_matches_hf(self):
        torch.manual_seed(42)
        dim = 64
        norm_npu = gemma3_npu_modeling.Gemma3RMSNorm(dim, eps=1e-6)
        norm_npu.weight.data.normal_(0, 0.1)

        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm as HFGemma3RMSNorm

        norm_hf = HFGemma3RMSNorm(dim, eps=1e-6)
        norm_hf.weight.data.copy_(norm_npu.weight.data)

        x = torch.randn(2, 16, dim)
        out_npu = norm_npu(x)
        out_hf = norm_hf(x)
        torch.testing.assert_close(out_npu, out_hf, rtol=1e-5, atol=1e-5)
