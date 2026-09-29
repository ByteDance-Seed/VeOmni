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

"""LTX 2.3 registry, integration, and numerical parity tests."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


# isort: off
# Import the package binder before the vendored top-level ``ltx_core`` modules.
from veomni.models.diffusers.ltx2_3.ltx_transformer import modeling_ltx2_3_transformer as ltx_modeling
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionFunction,
    MaskedAttentionFunction,
    VeomniLTXAttention,
    _hf_attention_mask,
)
from ltx_core.model.transformer.model import LTXModel
# isort: on

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import tiny_ltx2_3_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_ltx2_3_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.ops import resolve_op
from veomni.ops.config import get_ops_config, resolve_op_impl, set_ops_config


def _sdpa_ops_config() -> SimpleNamespace:
    """Portable LTX attention path. This family has no local eager forward."""
    ops = eager_ops_config()
    ops.attn_implementation = "sdpa"
    return ops


_OFFICIAL_ATTENTION_FORWARD = Attention.forward
_OFFICIAL_LTX_MODEL_FORWARD = LTXModel.forward


class _MathSDPAAttention:
    """Independent SDPA reference. Does not go through ``VeomniLTXAttention``."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_q, inner = q.shape
        dim_head = inner // heads
        query = q.view(batch, seq_q, heads, dim_head).transpose(1, 2)
        key = k.view(batch, -1, heads, dim_head).transpose(1, 2)
        value = v.view(batch, -1, heads, dim_head).transpose(1, 2)
        attn_mask = None if mask is None else _hf_attention_mask(mask)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(batch, seq_q, heads * dim_head)


def _bind_math_sdpa(model: torch.nn.Module) -> None:
    math_attn = _MathSDPAAttention()
    for module in model.modules():
        if isinstance(module, Attention):
            module.attention_function = math_attn
            module.masked_attention_function = math_attn


def _call_rms(x: torch.Tensor, weight: torch.Tensor | None, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        impl = resolve_op_impl("rms_norm_implementation")
        if weight is None:
            return resolve_op("rms_norm", "unweighted", impl).wrapper(x, eps=1e-6)
        return resolve_op("rms_norm", "standard", impl).wrapper(x, weight, eps=1e-6)
    finally:
        set_ops_config(previous)


def _ltx_inputs() -> dict:
    return {
        "hidden_states": [torch.randn(1, 4, 1, 2, 2)],
        "timestep": [torch.tensor(0.5)],
        "encoder_hidden_states": [torch.randn(1, 3, 16)],
    }


def test_ltx2_3_condition_builds_from_registered_class_without_assets(monkeypatch):
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.ltx2_3.ltx_condition import modeling_ltx2_3_condition

    monkeypatch.setattr(modeling_ltx2_3_condition.LTXVideoConditionModel, "_load_components", lambda self: None)

    model_class = MODELING_REGISTRY["LTXVideoConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_ltx2_3_condition.LTXVideoConditionModel
    assert model.config.model_type == "LTXVideoConditionModel"


def test_ltx2_3_eager_forward_and_backward_match_vendored_reference():
    torch.manual_seed(2)
    config = _tiny_config()
    with ops_config_scope(_sdpa_ops_config()):
        official = ltx_modeling.LTXVideoTransformerModel(config)
        official.apply(official._init_weights)
        _bind_math_sdpa(official)
        ours = ltx_modeling.LTXVideoTransformerModel(config)
        ours.load_state_dict(official.state_dict())
    official_inputs = _ltx_inputs()
    ours_inputs = {key: [value.detach().clone() for value in values] for key, values in official_inputs.items()}

    previous_attention_forward = Attention.forward
    previous_ltx_model_forward = LTXModel.forward
    try:

        def call(model):
            if model is official:
                Attention.forward = _OFFICIAL_ATTENTION_FORWARD
                LTXModel.forward = _OFFICIAL_LTX_MODEL_FORWARD
                inputs = official_inputs
            else:
                Attention.forward = ltx_modeling.LTXSPAttention_forward
                LTXModel.forward = ltx_modeling.LTXVideoModel_forward
                inputs = ours_inputs
            return model(**inputs).predictions[0]

        assert_outputs_and_grads_match(official, ours, call)
    finally:
        Attention.forward = previous_attention_forward
        LTXModel.forward = previous_ltx_model_forward


def test_ltx2_3_video_only_model_rejects_audio_input():
    with ops_config_scope(_sdpa_ops_config()):
        model = ltx_modeling.LTXVideoTransformerModel(_tiny_config())
    with pytest.raises(ValueError, match="Audio is not enabled"):
        ltx_modeling.LTXVideoModel_forward(
            model,
            video=None,
            audio=object(),
            perturbations=BatchedPerturbationConfig.empty(1),
        )


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


def _explicit_unweighted_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted RMS used by the LTX connector: ``x / sqrt(mean(x^2) + eps)``."""
    x_f = x.float()
    return x * torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)


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


def test_ltx2_3_connector_forwards_use_explicit_unweighted_rms_eps():
    """Condition connector blocks must pass the required RMSNorm ``eps``.

    Zero-layer ``Embeddings1DConnector`` is only the final unweighted RMS.
    ``_BasicTransformerBlock1D`` also runs attention and FF after the same eps.
    """
    from veomni.models.diffusers.ltx2_3.ltx_core.text_encoders.gemma.embeddings_connector import (
        _UNWEIGHTED_RMS_NORM_EPS,
        Embeddings1DConnector,
        _BasicTransformerBlock1D,
    )

    assert _UNWEIGHTED_RMS_NORM_EPS == 1e-6
    # Small activations make 1e-6 and 1e-3 disagree, so the test pins eps.
    hidden = torch.full((1, 4, 8), 1e-4, dtype=torch.float32, requires_grad=True)
    expected = _explicit_unweighted_rms(hidden, _UNWEIGHTED_RMS_NORM_EPS)
    wrong_eps = _explicit_unweighted_rms(hidden, 1e-3)
    assert not torch.allclose(expected, wrong_eps, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    with ops_config_scope(_sdpa_ops_config()):
        connector = Embeddings1DConnector(
            attention_head_dim=4,
            num_attention_heads=2,
            num_layers=0,
            num_learnable_registers=None,
        )
        block = _BasicTransformerBlock1D(dim=8, heads=2, dim_head=4)

        connector_hidden = hidden.detach().clone().requires_grad_(True)
        connector_out, _ = connector(connector_hidden)
        torch.testing.assert_close(connector_out, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
        connector_out.sum().backward()
        torch.testing.assert_close(
            connector_hidden.grad,
            torch.autograd.grad(expected.sum(), hidden)[0],
            atol=EAGER_GRAD_ATOL,
            rtol=EAGER_GRAD_RTOL,
        )

        block_hidden = hidden.detach().clone().requires_grad_(True)
        recorded_eps: list[float] = []
        bound = block.veomni_rms_norm_unweighted

        def _record_eps(x, *args, **kwargs):
            recorded_eps.append(kwargs["eps"])
            return bound(x, *args, **kwargs)

        block.veomni_rms_norm_unweighted = _record_eps
        block_out = block(block_hidden)
        assert recorded_eps == [_UNWEIGHTED_RMS_NORM_EPS, _UNWEIGHTED_RMS_NORM_EPS]
        assert torch.isfinite(block_out).all()
        block_out.sum().backward()
        assert block_hidden.grad is not None
        assert torch.isfinite(block_hidden.grad).all()


def test_ltx_automatic_to_callable_reads_ops_config():
    with ops_config_scope(_sdpa_ops_config()):
        adapter = AttentionFunction.AUTOMATIC.to_callable()
        masked = MaskedAttentionFunction.AUTOMATIC.to_callable()
    assert isinstance(adapter, VeomniLTXAttention)
    assert adapter.veomni_attn.impl == "sdpa"
    assert adapter.veomni_attn_masked is adapter.veomni_attn
    assert masked.veomni_attn.impl == "sdpa"
    assert masked.veomni_attn_masked is masked.veomni_attn


def test_ltx_sdpa_flash_to_callable_collapses_to_sdpa():
    adapter = AttentionFunction.SDPA_FLASH.to_callable()
    assert isinstance(adapter, VeomniLTXAttention)
    assert adapter.veomni_attn.impl == "sdpa"
    assert adapter.veomni_attn_masked is adapter.veomni_attn


def test_ltx_non_sdpa_masked_falls_back_to_sdpa():
    assert AttentionFunction.FLASH_ATTENTION_3.value == "flash_attention_3"
    adapter = VeomniLTXAttention("eager")
    assert adapter.veomni_attn.impl == "eager"
    assert adapter.veomni_attn_masked.impl == "sdpa"
    assert adapter.veomni_attn_masked is not adapter.veomni_attn


def test_ltx_adapter_expands_mask_and_skips_ulysses():
    adapter = VeomniLTXAttention("sdpa")
    captured: dict = {}

    def record(_module, query, _key, _value, attention_mask=None, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        captured["attention_mask"] = attention_mask
        captured["skip_ulysses"] = kwargs.get("skip_ulysses")
        return query.transpose(1, 2), None

    adapter.veomni_attn = record
    adapter.veomni_attn_masked = record
    query = torch.randn(1, 4, 16)
    key = torch.randn(1, 4, 16)
    value = torch.randn(1, 4, 16)

    out = adapter(query, key, value, heads=2)
    assert out.shape == (1, 4, 16)
    assert captured["query_shape"] == (1, 2, 4, 8)
    assert captured["attention_mask"] is None
    assert captured["skip_ulysses"] is True

    mask = torch.ones(4, 4)
    out = adapter(query, key, value, heads=2, mask=mask)
    assert out.shape == (1, 4, 16)
    assert captured["attention_mask"] is not None
    assert tuple(captured["attention_mask"].shape) == (1, 1, 4, 4)
    assert captured["attention_mask"].dtype == mask.dtype
    assert captured["skip_ulysses"] is True


def test_ltx_adapter_matches_sdpa():
    adapter = VeomniLTXAttention("sdpa")
    torch.manual_seed(0)
    query = torch.randn(2, 6, 16, requires_grad=True)
    key = torch.randn(2, 6, 16, requires_grad=True)
    value = torch.randn(2, 6, 16, requires_grad=True)
    heads = 2
    dim_head = 8
    q_ref = query.detach().clone().requires_grad_(True)
    k_ref = key.detach().clone().requires_grad_(True)
    v_ref = value.detach().clone().requires_grad_(True)

    out = adapter(query, key, value, heads)
    query_h = q_ref.view(2, 6, heads, dim_head).transpose(1, 2)
    key_h = k_ref.view(2, 6, heads, dim_head).transpose(1, 2)
    value_h = v_ref.view(2, 6, heads, dim_head).transpose(1, 2)
    ref = F.scaled_dot_product_attention(query_h, key_h, value_h)
    ref = ref.transpose(1, 2).reshape(2, 6, 16)
    torch.testing.assert_close(out, ref, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.clone())
    torch.testing.assert_close(query.grad, q_ref.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(key.grad, k_ref.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    torch.testing.assert_close(value.grad, v_ref.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


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

        binder = importlib.import_module("veomni.models.diffusers.ltx2_3.ltx_core")
        importlib.reload(binder)
        bound_utils = importlib.import_module("ltx_core.utils")
        package_utils = importlib.import_module("veomni.models.diffusers.ltx2_3.ltx_core.utils")
        assert bound_utils.__file__ == package_utils.__file__
        assert not hasattr(bound_utils, "MARKER")
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "ltx_core" or name.startswith("ltx_core."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
