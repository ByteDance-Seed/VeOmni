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

"""Lightning Indexer KL helpers and generated wiring."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.models_kernel.compare import eager_kernels_config
from veomni.distributed.parallel_state import ParallelState
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from veomni.models_kernel.transformers.deepseek_v4.indexer_loss import (
    _builds_indexer_kl,
    _indexer_loss_enabled,
    indexer_kl_terms,
)
from veomni.utils.model_outputs import MoeModelOutputWithIndexerKL


@contextmanager
def _kernels_config_installed(**overrides):
    previous = get_kernels_config()
    installed = eager_kernels_config()
    for key, value in overrides.items():
        setattr(installed, key, value)
    set_kernels_config(installed)
    try:
        yield installed
    finally:
        set_kernels_config(previous)


def _module_with_config(**overrides):
    config = DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
        **overrides,
    )
    return SimpleNamespace(config=config, layer_type="compressed_sparse_attention")


def _sequence_parallel_state(monkeypatch: pytest.MonkeyPatch, **sizes) -> ParallelState:
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 2)
    return ParallelState(dp_size=1, device_type="cpu", device_mesh=MagicMock(), **sizes)


def test_indexer_kl_terms_matches_hand_computation():
    target = torch.tensor([[[0.5, 0.5, 0.0]]])
    index_score = torch.tensor([[[0.0, 0.0, float("-inf")]]])
    kl, _ = indexer_kl_terms(index_score, target)
    assert torch.allclose(kl, torch.zeros(1, 1), atol=1e-6)

    index_score = torch.tensor([[[1.0, 0.0, float("-inf")]]])
    q0 = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    expected = 0.5 * (torch.log(torch.tensor(0.5)) - torch.log(q0[0])) + 0.5 * (
        torch.log(torch.tensor(0.5)) - torch.log(q0[1])
    )
    kl, uniform = indexer_kl_terms(index_score, target)
    assert torch.allclose(kl, expected.view(1, 1), atol=1e-6)
    assert kl.dtype is torch.float32 and uniform.dtype is torch.float32


def test_indexer_kl_terms_gradient_is_finite_when_a_query_sees_no_compressed_slot():
    neg_inf = float("-inf")
    index_score = torch.tensor([[[neg_inf, neg_inf], [0.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [0.4, 0.6]]])

    kl, uniform = indexer_kl_terms(index_score, target)
    assert torch.allclose(kl[0, 0], torch.zeros(()))
    assert torch.isfinite(uniform).all()
    kl.sum().backward()

    assert torch.isfinite(index_score.grad).all()
    assert (index_score.grad[0, 0] == 0).all()

    alone = index_score.detach()[:, 1:].clone().requires_grad_(True)
    indexer_kl_terms(alone, target[:, 1:])[0].sum().backward()
    torch.testing.assert_close(index_score.grad[:, 1:], alone.grad)
    assert alone.grad.abs().sum() > 0


def test_indexer_kl_terms_ignores_zero_target_slots():
    target = torch.tensor([[[1.0, 0.0]]])
    index_score = torch.tensor([[[0.0, float("-inf")]]])
    kl, uniform = indexer_kl_terms(index_score, target)
    assert torch.isfinite(kl).all()
    assert torch.allclose(kl, torch.zeros(1, 1), atol=1e-6)
    assert torch.isfinite(uniform).all()
    assert torch.allclose(uniform, torch.zeros(1, 1), atol=1e-6)


def test_a_row_the_teacher_gave_no_mass_is_excluded_from_both_terms():
    index_score = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0], [0.4, 0.6]]])

    kl, uniform = indexer_kl_terms(index_score, target)
    assert torch.allclose(kl[0, 0], torch.zeros(()))
    assert torch.allclose(uniform[0, 0], torch.zeros(()))
    assert uniform[0, 1] > 0

    kl.sum().backward()
    assert torch.isfinite(index_score.grad).all()
    assert (index_score.grad[0, 0] == 0).all()


def test_the_uniform_reference_is_what_a_zero_information_student_pays():
    neg_inf = float("-inf")
    target = torch.tensor([[[0.7, 0.2, 0.1], [0.25, 0.75, 0.0]]])
    flat = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, neg_inf]]])

    kl_of_a_flat_student, uniform = indexer_kl_terms(flat, target)
    torch.testing.assert_close(uniform, kl_of_a_flat_student, atol=1e-6, rtol=1e-6)

    entropy = -(target * torch.log(target.clamp_min(torch.finfo(torch.float32).tiny))).sum(-1)
    expected = torch.log(torch.tensor([[3.0, 2.0]])) - entropy
    torch.testing.assert_close(uniform, expected, atol=1e-6, rtol=1e-6)

    captured = 1.0 - kl_of_a_flat_student / uniform
    torch.testing.assert_close(captured, torch.zeros_like(captured), atol=1e-6, rtol=0)


def test_the_uniform_reference_carries_no_gradient_into_the_objective():
    target = torch.tensor([[[0.7, 0.2, 0.1]]])
    index_score = torch.tensor([[[0.3, -0.4, 1.2]]], requires_grad=True)

    kl, uniform = indexer_kl_terms(index_score, target)
    assert uniform.grad_fn is None and not uniform.requires_grad

    (kl.sum() + uniform.sum()).backward()
    with_reference = index_score.grad.clone()

    expected = torch.softmax(index_score.detach(), dim=-1) - target
    torch.testing.assert_close(with_reference, expected, atol=1e-6, rtol=1e-6)

    index_score.grad = None
    indexer_kl_terms(index_score, target)[0].sum().backward()
    assert torch.equal(with_reference, index_score.grad)


def test_indexer_loss_enabled_is_off_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: ParallelState(dp_size=1, ulysses_size=1),
    )
    assert _indexer_loss_enabled(_module_with_config()) is False


def test_indexer_loss_enabled_refuses_eager_indexer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: ParallelState(dp_size=1, ulysses_size=1),
    )
    with _kernels_config_installed(dsa_attention_implementation="tilelang"):
        with pytest.raises(ValueError, match="dsa_indexer_implementation"):
            _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True))


def test_indexer_loss_enabled_refuses_eager_attention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: ParallelState(dp_size=1, ulysses_size=1),
    )
    with _kernels_config_installed(dsa_indexer_implementation="tilelang"):
        with pytest.raises(ValueError, match="dsa_attention_implementation"):
            _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True))


@pytest.mark.parametrize(
    ("mode", "sizes", "expected"),
    [
        ("ulysses", {"ulysses_size": 2}, r"requires ulysses_size=1, got ulysses_size=2"),
        ("cp", {"cp_size": 2}, r"requires cp_size=1, got cp_size=2"),
    ],
)
def test_indexer_loss_enabled_refuses_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch, mode: str, sizes: dict, expected: str
):
    del mode
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: _sequence_parallel_state(monkeypatch, **sizes),
    )
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        with pytest.raises(ValueError, match=expected):
            _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True))


def test_a_config_without_the_fields_reads_as_off():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as UpstreamConfig

    upstream = UpstreamConfig(num_hidden_layers=2, layer_types=["compressed_sparse_attention"] * 2)
    assert not hasattr(upstream, "dsa_indexer_loss")
    assert _indexer_loss_enabled(SimpleNamespace(config=upstream)) is False


def test_indexer_loss_enabled_on_the_supported_configuration(monkeypatch: pytest.MonkeyPatch):
    state = ParallelState(dp_size=1, ulysses_size=1, device_type="cpu")
    assert state.ulysses_size == 1
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: state,
    )
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        assert _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True)) is True


@pytest.mark.parametrize("coef", [0.0, -0.0, 0, -1.0])
def test_a_non_positive_coefficient_switches_the_objective_off(coef):
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        assert _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True, dsa_indexer_loss_coef=coef)) is False


def test_a_coefficient_of_zero_does_not_refuse_an_unsupported_configuration():
    with _kernels_config_installed():
        assert _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True, dsa_indexer_loss_coef=0.0)) is False


@pytest.mark.parametrize("coef", [1e-8, 0.5, 1.0])
def test_a_positive_coefficient_leaves_the_objective_on(monkeypatch: pytest.MonkeyPatch, coef: float):
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: ParallelState(dp_size=1, ulysses_size=1, device_type="cpu"),
    )
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        assert _indexer_loss_enabled(_module_with_config(dsa_indexer_loss=True, dsa_indexer_loss_coef=coef)) is True


def test_builds_indexer_kl_only_on_csa_layers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss.get_parallel_state",
        lambda: ParallelState(dp_size=1, ulysses_size=1),
    )
    with _kernels_config_installed(dsa_indexer_implementation="tilelang", dsa_attention_implementation="tilelang"):
        csa = _module_with_config(dsa_indexer_loss=True)
        hca = SimpleNamespace(config=csa.config, layer_type="heavily_compressed_attention")
        assert _builds_indexer_kl(csa) is True
        assert _builds_indexer_kl(hca) is False


def test_the_npu_backend_refuses_the_objective_rather_than_dropping_it():
    from veomni.models_kernel.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as npu

    with pytest.raises(NotImplementedError, match="dsa_indexer_loss is not implemented on NPU"):
        npu.DeepseekV4CSACompressor.forward(
            None,
            hidden_states=None,
            q_residual=None,
            position_ids=None,
            past_key_values=None,
            layer_idx=0,
            build_indexer_loss=True,
        )


def test_causallm_folds_indexer_kl_into_loss_and_aux_metrics(monkeypatch: pytest.MonkeyPatch):
    from veomni.models_kernel.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
        DeepseekV4ForCausalLM,
    )

    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=1,
        n_routed_experts=1,
        max_position_embeddings=16,
        o_groups=4,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=8,
        layer_types=["compressed_sparse_attention"],
        dsa_indexer_loss=True,
        dsa_indexer_loss_coef=0.5,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    previous = get_kernels_config()
    set_kernels_config(eager_kernels_config())
    try:
        model = DeepseekV4ForCausalLM(config)
    finally:
        set_kernels_config(previous)

    hidden = torch.randn(1, 4, config.hidden_size)
    kl_total = torch.tensor(8.0)
    uniform_total = torch.tensor(16.0)
    outputs = MoeModelOutputWithIndexerKL(
        last_hidden_state=hidden,
        indexer_kl_total=kl_total,
        indexer_uniform_total=uniform_total,
        indexer_query_tokens=4,
        indexer_kl_layers=1,
    )
    monkeypatch.setattr(model.model, "forward", lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        model,
        "loss_function",
        lambda **kwargs: (torch.tensor(2.0), None, None),
    )
    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu.get_parallel_state",
        lambda: SimpleNamespace(sp_enabled=False),
    )

    out = model(input_ids=torch.ones(1, 4, dtype=torch.long), labels=torch.ones(1, 4, dtype=torch.long))
    assert out.aux_metrics is not None
    torch.testing.assert_close(out.aux_metrics["indexer_kl"], torch.tensor(2.0))
    torch.testing.assert_close(out.aux_metrics["indexer_kl_uniform"], torch.tensor(4.0))
    torch.testing.assert_close(out.aux_metrics["lm_loss_before_indexer_kl"], torch.tensor(2.0))
    torch.testing.assert_close(out.loss, torch.tensor(3.0))


def test_the_flag_is_refused_when_no_layer_can_build_a_kl(monkeypatch: pytest.MonkeyPatch):
    from veomni.models_kernel.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as gpu

    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=1,
        n_routed_experts=1,
        max_position_embeddings=16,
        o_groups=4,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=8,
        layer_types=["sliding_attention", "sliding_attention"],
        dsa_indexer_loss=True,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    previous = get_kernels_config()
    set_kernels_config(eager_kernels_config())
    try:
        model = gpu.DeepseekV4ForCausalLM(config)
    finally:
        set_kernels_config(previous)

    monkeypatch.setattr(
        "veomni.models_kernel.transformers.deepseek_v4.indexer_loss._indexer_loss_enabled",
        lambda module: True,
    )
    monkeypatch.setattr(gpu, "_indexer_loss_enabled", lambda module: True)
    monkeypatch.setattr(gpu, "get_parallel_state", lambda: ParallelState(dp_size=1, ulysses_size=1))
    input_ids = torch.ones(1, 8, dtype=torch.long)
    with pytest.raises(RuntimeError, match="no layer of this model builds an indexer KL"):
        model(input_ids=input_ids, use_cache=False)
