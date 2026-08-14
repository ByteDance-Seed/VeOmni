# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from veomni.ops.kernels.gated_delta_rule import varlen_metadata as varlen_metadata_module
from veomni.ops.kernels.gated_delta_rule._ascend.varlen_active_segments import (
    build_active_varlen_segments,
    empty_varlen_result,
)


def _plan(points: list[int], initial_state: torch.Tensor | None = None):
    cu = torch.tensor(points, dtype=torch.int32)
    return build_active_varlen_segments(
        cu,
        cu_seqlens_list=points,
        token_count=points[-1],
        initial_state=initial_state,
    )


@pytest.mark.parametrize(
    ("points", "active", "compact"),
    [
        ([0, 0, 32, 64], (1, 2), [0, 32, 64]),
        ([0, 32, 32, 64], (0, 2), [0, 32, 64]),
        ([0, 32, 64, 64], (0, 1), [0, 32, 64]),
        ([0, 0, 0, 32, 32], (2,), [0, 32]),
        ([0, 0, 0], (), [0]),
    ],
)
def test_compacts_empty_segments_without_moving_tokens(points, active, compact):
    plan = _plan(points)
    assert plan.active_indices == active
    assert plan.compact_cu_seqlens.tolist() == compact
    assert list(plan.compact_cu_seqlens_list) == compact


def test_all_active_is_zero_allocation_fast_path():
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    host = [0, 32, 64]
    state = torch.randn(2, 1, 2, 3)
    plan = build_active_varlen_segments(cu, cu_seqlens_list=host, token_count=64, initial_state=state)
    assert plan.all_active
    assert plan.compact_cu_seqlens is cu
    assert plan.compact_cu_seqlens_list is host
    assert plan.compact_initial_state(state) is state


@pytest.mark.parametrize(
    ("points", "token_count", "message"),
    [
        ([1, 2], 2, "start at zero"),
        ([0, 3, 2], 2, "monotonically"),
        ([0, 1], 2, "token count"),
    ],
)
def test_invalid_cu_fails_closed(points, token_count, message):
    cu = torch.tensor(points, dtype=torch.int32)
    with pytest.raises(ValueError, match=message):
        build_active_varlen_segments(
            cu,
            cu_seqlens_list=points,
            token_count=token_count,
            initial_state=None,
        )


def test_non_integer_host_point_and_state_count_fail_closed():
    cu = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(TypeError, match="integers"):
        build_active_varlen_segments(cu, cu_seqlens_list=[0.0, 1.0], token_count=1, initial_state=None)
    with pytest.raises(ValueError, match="initial states"):
        build_active_varlen_segments(
            cu,
            cu_seqlens_list=[0, 1],
            token_count=1,
            initial_state=torch.zeros(2, 1, 1, 1),
        )


def test_restore_final_state_preserves_full_n_autograd_contract():
    initial_state = torch.arange(3.0).reshape(3, 1, 1, 1).requires_grad_()
    plan = _plan([0, 0, 32, 64], initial_state)
    compact_initial = plan.compact_initial_state(initial_state)
    assert compact_initial is not None
    compact_final = compact_initial * 2
    final = plan.restore_final_state(compact_final, initial_state)
    assert final is not None
    assert final.flatten().tolist() == [0.0, 2.0, 4.0]
    final.sum().backward()
    assert initial_state.grad is not None
    assert initial_state.grad.flatten().tolist() == [1.0, 2.0, 2.0]


def test_cpu_cu_compacts_state_on_the_state_device():
    initial_state = torch.empty(3, 1, 2, 3, device="meta")
    plan = _plan([0, 0, 32, 64], initial_state)
    compact = plan.compact_initial_state(initial_state)
    assert compact is not None
    assert compact.device.type == "meta"
    assert compact.shape == (2, 1, 2, 3)
    restored = plan.restore_final_state(compact, initial_state)
    assert restored is not None
    assert restored.device.type == "meta"
    assert restored.shape == initial_state.shape


def test_no_final_state_gives_inactive_state_zero_gradient():
    initial_state = torch.arange(3.0).reshape(3, 1, 1, 1).requires_grad_()
    plan = _plan([0, 0, 32, 64], initial_state)
    compact_initial = plan.compact_initial_state(initial_state)
    assert compact_initial is not None
    compact_initial.sum().backward()
    assert initial_state.grad is not None
    assert initial_state.grad.flatten().tolist() == [0.0, 1.0, 1.0]


def test_all_empty_bypass_keeps_zero_input_grads_and_state_identity():
    output = torch.empty(1, 0, 2, 3, requires_grad=True)
    dependency = torch.empty(1, 0, 2, 4, requires_grad=True)
    initial_state = torch.randn(2, 2, 4, 3, requires_grad=True)
    plan = _plan([0, 0, 0], initial_state)
    empty_output, final = empty_varlen_result(
        output,
        dependencies=[output, dependency, initial_state],
        plan=plan,
        initial_state=initial_state,
        output_final_state=True,
        state_shape=(2, 4, 3),
    )
    assert empty_output.shape == output.shape
    assert final is not None
    torch.testing.assert_close(final, initial_state)
    (empty_output.sum() + final.sum()).backward()
    assert output.grad is not None and output.grad.numel() == 0
    assert dependency.grad is not None and dependency.grad.numel() == 0
    torch.testing.assert_close(initial_state.grad, torch.ones_like(initial_state))


def _package(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _identity_decorator(function):
    return function


def _load_wrapper(monkeypatch: pytest.MonkeyPatch, *, backend: str, fla_npu_available: bool = True):
    root = f"_gdn_active_segment_{backend}_test"
    ascend = f"{root}._ascend"
    _package(monkeypatch, root)
    _package(monkeypatch, ascend)
    _package(monkeypatch, f"{ascend}.triton")
    monkeypatch.setitem(
        sys.modules,
        f"{ascend}.varlen_active_segments",
        sys.modules["veomni.ops.kernels.gated_delta_rule._ascend.varlen_active_segments"],
    )
    monkeypatch.setitem(
        sys.modules,
        f"{root}.varlen_metadata",
        varlen_metadata_module,
    )

    if backend == "mm":
        _module(monkeypatch, f"{root}.normalization", producer_dtype_l2norm=lambda tensor, **_: tensor)
        _module(
            monkeypatch,
            f"{ascend}.triton.chunk_delta_h",
            chunk_gated_delta_rule_bwd_dhu=lambda *args, **kwargs: None,
            chunk_gated_delta_rule_fwd_h=lambda *args, **kwargs: None,
        )
        _module(
            monkeypatch,
            f"{ascend}.triton.chunk_o",
            chunk_bwd_dqkwg=lambda *args, **kwargs: None,
            chunk_bwd_dv_local=lambda *args, **kwargs: None,
            chunk_fwd_o=lambda *args, **kwargs: None,
        )
        _module(
            monkeypatch,
            f"{ascend}.triton.chunk_scaled_dot_kkt",
            chunk_scaled_dot_kkt_fwd=lambda *args, **kwargs: None,
        )
        _module(
            monkeypatch,
            f"{ascend}.triton.wy_fast",
            prepare_wy_repr_bwd=lambda *args, **kwargs: None,
            recompute_w_u_fwd=lambda *args, **kwargs: None,
        )
        _module(monkeypatch, f"{ascend}.triton.solve_tril", solve_tril=lambda *args, **kwargs: None)
        _module(monkeypatch, f"{ascend}.triton.cumsum", chunk_local_cumsum=lambda tensor, **_: tensor)
        _module(
            monkeypatch,
            f"{ascend}.triton.utils",
            autocast_custom_bwd=_identity_decorator,
            autocast_custom_fwd=_identity_decorator,
            input_guard=_identity_decorator,
        )
        filename = "chunk_gated_delta_rule_mm.py"
    else:
        _package(monkeypatch, f"{ascend}.triton_core")
        _module(monkeypatch, "torch_npu")
        if fla_npu_available:
            _module(monkeypatch, "fla_npu")
        else:
            monkeypatch.setitem(sys.modules, "fla_npu", None)
        _module(
            monkeypatch,
            f"{ascend}.triton_core.chunk_scaled_dot_kkt",
            chunk_scaled_dot_kkt_fwd=lambda *args, **kwargs: None,
        )
        _module(
            monkeypatch,
            f"{ascend}.triton_core.l2norm",
            l2norm_bwd=lambda *args, **kwargs: None,
            l2norm_fwd=lambda tensor: (tensor, None),
        )
        _module(monkeypatch, f"{ascend}.triton.cumsum", chunk_local_cumsum=lambda tensor, **_: tensor)
        _module(
            monkeypatch,
            f"{ascend}.triton_core.utils",
            autocast_custom_bwd=_identity_decorator,
            autocast_custom_fwd=_identity_decorator,
            input_guard=_identity_decorator,
        )
        _module(monkeypatch, f"{ascend}.triton.utils", is_arch35=lambda: False)
        _module(monkeypatch, f"{ascend}.triton.solve_tril", solve_tril=lambda *args, **kwargs: None)
        filename = "flash_gated_delta_rule.py"

    path = Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/_ascend" / filename
    name = f"{ascend}.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_mm_wrapper_compacts_native_state_and_restores_full_final(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="mm")
    captured = {}

    def fake_apply(*args):
        captured["initial_state"] = args[6]
        captured["cu_seqlens"] = args[8]
        return args[2].clone(), args[6] * 2

    monkeypatch.setattr(module.ChunkGatedDeltaRuleFunction, "apply", staticmethod(fake_apply))
    q = torch.ones(1, 64, 1, 2, dtype=torch.bfloat16)
    k = q.clone()
    v = torch.ones(1, 64, 1, 3, dtype=torch.bfloat16)
    g = torch.ones(1, 64, 1, dtype=torch.bfloat16)
    beta = g.clone()
    initial_state = torch.arange(18.0).reshape(3, 1, 2, 3)
    _, final = module.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 0, 32, 64], dtype=torch.int32),
        cu_seqlens_list=[0, 0, 32, 64],
    )
    assert captured["cu_seqlens"].tolist() == [0, 32, 64]
    torch.testing.assert_close(captured["initial_state"], initial_state[1:])
    assert final is not None
    torch.testing.assert_close(final[0], initial_state[0])
    torch.testing.assert_close(final[1:], initial_state[1:] * 2)


def test_mm_wrapper_preserves_legacy_positional_chunk_size_and_head_first(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="mm")
    captured = {}

    def fake_apply(*args):
        captured["chunk_size"] = args[10]
        return args[2].clone(), None

    monkeypatch.setattr(module.ChunkGatedDeltaRuleFunction, "apply", staticmethod(fake_apply))
    q = torch.ones(1, 8, 1, 2, dtype=torch.bfloat16)
    module.chunk_gated_delta_rule(
        q,
        q.clone(),
        torch.ones(1, 8, 1, 3, dtype=torch.bfloat16),
        torch.ones(1, 8, 1, dtype=torch.bfloat16),
        torch.ones(1, 8, 1, dtype=torch.bfloat16),
        None,
        None,
        False,
        False,
        None,
        32,
        False,
    )
    assert captured["chunk_size"] == 32


def test_ascendc_wrapper_discards_stale_metadata_after_compaction(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc")
    captured = {}

    def fake_ensure(**kwargs):
        captured["ensure"] = kwargs
        compact_cu = kwargs["cu_seqlens"]
        return compact_cu, [0, 32, 64], {"64": "fresh-tensor"}, {"64": [0, 0, 1, 0]}

    def fake_apply(*args):
        captured["initial_state"] = args[6]
        captured["cu_seqlens"] = args[8]
        captured["chunk_indices"] = args[10]
        return args[2].transpose(1, 2).contiguous(), args[6] * 2

    monkeypatch.setattr(module, "_ensure_varlen_metadata", fake_ensure)
    monkeypatch.setattr(module.ChunkGatedDeltaRuleFunction, "apply", staticmethod(fake_apply))
    q = torch.ones(1, 1, 64, 2, dtype=torch.bfloat16)
    k = q.clone()
    v = torch.ones(1, 1, 64, 3, dtype=torch.bfloat16)
    g = torch.ones(1, 64, 1, dtype=torch.bfloat16)
    beta = g.clone()
    initial_state = torch.arange(18.0).reshape(3, 1, 2, 3)
    _, final = module.flash_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 0, 32, 64], dtype=torch.int32),
        cu_seqlens_list=[0, 0, 32, 64],
        chunk_indices={"64": "stale-tensor"},
        chunk_indices_list={"64": [1, 0, 2, 0]},
    )
    assert captured["ensure"]["chunk_indices"] is None
    assert captured["ensure"]["chunk_indices_list"] is None
    assert captured["cu_seqlens"].tolist() == [0, 32, 64]
    assert captured["chunk_indices"] == {"64": "fresh-tensor"}
    torch.testing.assert_close(captured["initial_state"], initial_state[1:])
    assert final is not None
    torch.testing.assert_close(final[0], initial_state[0])
    torch.testing.assert_close(final[1:], initial_state[1:] * 2)


def test_ascendc_wrapper_rejects_initial_state_gradient_without_varlen_metadata(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc")
    q = torch.ones(1, 1, 8, 2, dtype=torch.bfloat16)
    initial_state = torch.zeros(1, 1, 2, 3, requires_grad=True)

    with pytest.raises(NotImplementedError, match="cannot differentiate initial_state"):
        module.flash_gated_delta_rule(
            q,
            q.clone(),
            torch.ones(1, 1, 8, 3, dtype=torch.bfloat16),
            torch.ones(1, 8, 1, dtype=torch.bfloat16),
            torch.ones(1, 8, 1, dtype=torch.bfloat16),
            initial_state=initial_state,
        )


def test_ascendc_all_empty_bypasses_fla_npu_and_keeps_zero_input_gradients(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc", fla_npu_available=False)
    q = torch.empty(1, 1, 0, 2, dtype=torch.bfloat16, requires_grad=True)
    k = q.detach().clone().requires_grad_(True)
    v = torch.empty(1, 1, 0, 3, dtype=torch.bfloat16, requires_grad=True)
    g = torch.empty(1, 0, 1, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.empty(1, 0, 1, dtype=torch.bfloat16, requires_grad=True)

    output, final_state = module.flash_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens_list=[0, 0],
    )

    assert output.shape == (1, 0, 1, 3)
    assert final_state is None
    output.sum().backward()
    for tensor in (q, k, v, g, beta):
        assert tensor.grad is not None
        assert tensor.grad.numel() == 0


def test_ascendc_all_empty_bypasses_nonempty_vjp_restrictions(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc", fla_npu_available=False)
    q = torch.empty(1, 1, 0, 2, dtype=torch.bfloat16, requires_grad=True)
    k = q.detach().clone().requires_grad_(True)
    v = torch.empty(1, 1, 0, 3, dtype=torch.bfloat16, requires_grad=True)
    g = torch.empty(1, 0, 1, dtype=torch.bfloat16, requires_grad=True)
    beta = torch.empty(1, 0, 1, dtype=torch.bfloat16, requires_grad=True)
    initial_state = torch.randn(1, 1, 2, 3, dtype=torch.float32, requires_grad=True)

    output, final_state = module.flash_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlens_list=[0, 0],
    )

    assert final_state is not None
    torch.testing.assert_close(final_state, initial_state)
    (output.sum() + final_state.sum()).backward()
    for tensor in (q, k, v, g, beta):
        assert tensor.grad is not None
        assert tensor.grad.numel() == 0
    torch.testing.assert_close(initial_state.grad, torch.ones_like(initial_state))


def test_ascendc_wrapper_rejects_final_state_vjp(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc")
    q = torch.ones(1, 1, 8, 2, dtype=torch.bfloat16, requires_grad=True)

    with pytest.raises(NotImplementedError, match="cannot differentiate final_state"):
        module.flash_gated_delta_rule(
            q,
            q.clone(),
            torch.ones(1, 1, 8, 3, dtype=torch.bfloat16),
            torch.ones(1, 8, 1, dtype=torch.bfloat16),
            torch.ones(1, 8, 1, dtype=torch.bfloat16),
            output_final_state=True,
        )


def test_ascendc_all_empty_still_requires_flattened_varlen_batch(monkeypatch):
    module = _load_wrapper(monkeypatch, backend="ascendc")
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        module.flash_gated_delta_rule(
            torch.empty(2, 1, 0, 2, dtype=torch.bfloat16),
            torch.empty(2, 1, 0, 2, dtype=torch.bfloat16),
            torch.empty(2, 1, 0, 3, dtype=torch.bfloat16),
            torch.empty(2, 0, 1, dtype=torch.bfloat16),
            torch.empty(2, 0, 1, dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 0, 0], dtype=torch.int32),
            cu_seqlens_list=[0, 0, 0],
        )
