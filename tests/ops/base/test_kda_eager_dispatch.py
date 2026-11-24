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

"""chunk_kda/kimi_k3 ops-tree dispatch contracts.

The ``eager`` row must stay the pure-PyTorch reference from the ops tree
(``veomni.ops.kernels.kda.kimi_k3.eager``): it never dispatches to
flash-linear-attention's ``chunk_kda`` on non-cuda tensors (fla's autograd
entry indexes ``torch.<device_type>.device``, which ``torch.cpu`` lacks, so
environments that install fla would crash every eager CPU forward through
the KDA layers), it normalizes the historical ``transpose_state_layout``
name onto the naive reference, and the registry keeps ``fla``/``ascendc``
device-gated so accidental cross-device resolution fails loudly.
"""

import sys
import types

import pytest
import torch

import veomni.ops.kernels.kda.kimi_k3.eager as kda_eager_module
from veomni.ops import OP_REGISTRY, resolve_op


def _cpu_inputs():
    torch.manual_seed(1234)
    q = torch.randn(1, 8, 2, 16, dtype=torch.float32, device="cpu")
    k = torch.randn(1, 8, 2, 16, dtype=torch.float32, device="cpu")
    v = torch.randn(1, 8, 2, 32, dtype=torch.float32, device="cpu")
    g = -torch.rand(1, 8, 2, 16, dtype=torch.float32, device="cpu")
    beta = torch.rand(1, 8, 2, dtype=torch.float32, device="cpu")
    return q, k, v, g, beta


def _install_fake_fla(monkeypatch):
    """Make any fla dispatch fail the test with a clear sentinel."""
    fake_kda = types.ModuleType("fla.ops.kda")

    def _fla_crash(*args, **kwargs):
        raise AssertionError("fla chunk_kda must not run for non-cuda tensors")

    fake_kda.chunk_kda = _fla_crash
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", types.ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake_kda)


def test_eager_row_dispatches_to_ops_tree_reference(monkeypatch):
    """The eager row delegates to the ops-tree naive reference, not fla.

    The spy pins the dispatch target exactly (identity of the returned
    tensors), so the contract needs no numerical tolerance: on CPU hosts
    lazy kernel selection cannot perturb a same-call comparison.
    """
    _install_fake_fla(monkeypatch)

    seen = []
    real_naive = kda_eager_module.chunk_kda_naive

    def _spy(*args, **kwargs):
        seen.append(real_naive(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(kda_eager_module, "chunk_kda_naive", _spy)

    q, k, v, g, beta = _cpu_inputs()
    chunk_kda = resolve_op("chunk_kda", "kimi_k3", "eager").wrapper
    o, final_state = chunk_kda(q=q, k=k, v=v, g=g, beta=beta)

    assert len(seen) == 1, "the eager row must dispatch exactly once through the ops-tree reference"
    o_ref, final_ref = seen[0]
    assert o is o_ref, "the eager row must return the reference's result unchanged"
    assert final_state is None and final_ref is None
    assert o.shape == (1, 8, 2, 32)
    assert torch.isfinite(o).all()


def test_eager_row_normalizes_state_layout_kwarg_names(monkeypatch):
    """``state_v_first`` (fla's rename) maps onto the reference's historical kwarg."""
    _install_fake_fla(monkeypatch)

    captured = []
    real_naive = kda_eager_module.chunk_kda_naive

    def _spy(*args, **kwargs):
        captured.append(kwargs)
        return real_naive(*args, **kwargs)

    monkeypatch.setattr(kda_eager_module, "chunk_kda_naive", _spy)

    q, k, v, g, beta = _cpu_inputs()
    chunk_kda = resolve_op("chunk_kda", "kimi_k3", "eager").wrapper

    chunk_kda(q=q, k=k, v=v, g=g, beta=beta, state_v_first=True)
    assert captured[-1].get("transpose_state_layout") is True
    assert "state_v_first" not in captured[-1]

    chunk_kda(q=q, k=k, v=v, g=g, beta=beta, transpose_state_layout=True)
    assert captured[-1].get("transpose_state_layout") is True
    assert "state_v_first" not in captured[-1]

    with pytest.raises(ValueError, match="only one"):
        chunk_kda(q=q, k=k, v=v, g=g, beta=beta, transpose_state_layout=True, state_v_first=True)


def test_fla_row_is_device_gated_to_cuda():
    """fla's autograd entry crashes on non-cuda tensors, so its row must not resolve there."""
    if torch.cuda.is_available():
        pytest.skip("contract is the non-cuda raise; cuda hosts resolve the row")
    with pytest.raises(RuntimeError, match="not registered for device"):
        resolve_op("chunk_kda", "kimi_k3", "fla")


def test_fla_row_rejects_builds_lacking_the_kda_contract(monkeypatch):
    """A flash-linear-attention build without the KDA gating params must fail loudly.

    The K3 callers always pass the beta-sigmoid gating knobs; an installed
    fla whose chunk_kda lacks them would silently drop them (or crash in
    the kernel), so the wrapper feature-detects the signature and raises
    NotImplementedError naming the missing parameters.
    """
    from veomni.ops.kernels.kda import _fla

    def _fla_chunk_kda(q, k, v, g, beta, scale=None):
        return None  # signature deliberately lacks the KDA gating contract

    fake_kda = types.ModuleType("fla.ops.kda")
    fake_kda.chunk_kda = _fla_chunk_kda
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", types.ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake_kda)

    q, k, v, g, beta = _cpu_inputs()
    with pytest.raises(NotImplementedError) as excinfo:
        _fla(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    message = str(excinfo.value)
    # The raise names the gating parameters the installed build lacks.
    for param in ("use_qk_l2norm_in_kernel", "A_log", "cu_seqlens"):
        assert param in message, param


def test_fla_row_rejects_kwargs_swallowing_builds(monkeypatch):
    """The pinned 0.4.1 signature swallows the K3 knobs via ``**kwargs``.

    flash-linear-attention 0.4.1's chunk_kda ends in ``**kwargs`` and spells
    none of use_beta_sigmoid_in_kernel / safe_gate / lower_bound as named
    parameters, so every K3 gating knob the modeling passes would be silently
    dropped (raw beta, unbounded gate). The row must refuse on such builds
    instead of training different math, even though ``**kwargs`` nominally
    accepts the arguments.
    """
    from veomni.ops.kernels.kda import _fla

    def _fla_041_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        cu_seqlens=None,
        chunk_indices=None,
        **kwargs,
    ):
        return None  # exact pinned-0.4.1 signature shape

    fake_kda = types.ModuleType("fla.ops.kda")
    fake_kda.chunk_kda = _fla_041_chunk_kda
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", types.ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake_kda)

    q, k, v, g, beta = _cpu_inputs()
    with pytest.raises(NotImplementedError) as excinfo:
        _fla(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            A_log=torch.randn(2),
            dt_bias=torch.randn(32),
        )
    message = str(excinfo.value)
    for param in ("use_beta_sigmoid_in_kernel", "safe_gate", "lower_bound"):
        assert param in message, param


def test_fla_row_accepts_builds_with_the_full_contract(monkeypatch):
    """A build that names the gating contract runs through, with the layout rename."""
    from veomni.ops.kernels.kda import _fla

    captured = {}

    def _fla_full_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        safe_gate=True,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
        cu_seqlens=None,
        chunk_indices=None,
        disable_recompute=True,
        state_v_first=False,
        **kwargs,
    ):
        captured.update(kwargs)
        captured.update(
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            state_v_first=state_v_first,
        )
        return "SENTINEL"

    fake_kda = types.ModuleType("fla.ops.kda")
    fake_kda.chunk_kda = _fla_full_chunk_kda
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", types.ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake_kda)

    q, k, v, g, beta = _cpu_inputs()
    result = _fla(
        q,
        k,
        v,
        g,
        beta,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
        cu_seqlens_cpu=[0, 8],
    )
    assert result == "SENTINEL"
    assert captured["use_beta_sigmoid_in_kernel"] is True
    assert captured["safe_gate"] is True
    assert captured["lower_bound"] == -5.0
    assert captured["state_v_first"] is True
    assert "cu_seqlens_cpu" not in captured
    assert "transpose_state_layout" not in captured


def test_ascendc_row_is_device_gated_to_npu():
    """The fused AscendC row must not resolve off Ascend NPUs."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        pytest.skip("contract is the non-npu raise; npu hosts resolve the row")
    with pytest.raises(RuntimeError, match="not registered for device"):
        resolve_op("chunk_kda", "kimi_k3", "ascendc")


def test_default_resolution_prefers_the_platform_row():
    entry = resolve_op("chunk_kda", "kimi_k3", None)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        # The fused row is the NPU default only while its out-of-tree kernel
        # packages are importable; stock unit-CI images deliberately ship
        # without them (see test_kda_ascendc_state_layout for the same rule)
        # and the registry then resolves to the eager reference by design
        # (OpRegistry.default_impl). Track that availability instead of
        # assuming it: hosts with the kernels resolve ascendc, hosts without
        # them resolve the documented eager fallback.
        if "ascendc" in OP_REGISTRY.list_available("chunk_kda", "kimi_k3"):
            assert entry.impl == "ascendc"
        else:
            assert entry.impl == "eager"
    elif torch.cuda.is_available():
        assert entry.impl in ("eager", "fla")
    else:
        assert entry.impl == "eager"
