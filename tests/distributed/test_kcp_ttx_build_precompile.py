"""Host contract for KCP TTX build-stage precompile (before AC/TP/FSDP)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.torch_parallelize import (
    _canonical_kcp_ttx_hkv_payload,
    _kcp_gdn_hkv_signatures,
    _precompile_kcp_ttx_before_fsdp,
    build_parallelize_model,
)


class _KcpGdn(nn.Module):
    def __init__(
        self,
        *,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        impl: str = "kcp",
    ) -> None:
        super().__init__()
        self.gdn_context_parallel_implementation = impl
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.weight = nn.Parameter(torch.ones(head_v_dim))


class _ToyModel(nn.Module):
    def __init__(self, modules: list[nn.Module] | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(modules or [_KcpGdn(), _KcpGdn()])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


def _parallel_state(**kwargs):
    values = dict(
        fsdp_enabled=True,
        tp_enabled=False,
        dp_mode="fsdp2",
        ulysses_size=2,
        ulysses_enabled=True,
        local_rank=0,
        tp_mesh=None,
    )
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def _install_precompile_fakes(monkeypatch, *, warmup=None, coord=None, identical=None):
    calls = []

    def default_warmup(key, value, g, beta):
        calls.append(
            {
                "shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "key_dtype": key.dtype,
                "g_dtype": g.dtype,
                "beta_dtype": beta.dtype,
                "device": str(key.device),
            }
        )

    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize._warmup_kcp_ttx_build_signature",
        warmup or default_warmup,
    )
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize._coordinate_kcp_ttx_precompile_success",
        coord or (lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize._assert_identical_kcp_ttx_signatures",
        identical or (lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize._kcp_ttx_precompile_device",
        lambda: torch.device("cpu"),
    )
    return calls


def test_hkv_signatures_are_unique_and_ulysses_adjusted():
    model = _ToyModel([_KcpGdn(), _KcpGdn(), _KcpGdn(head_v_dim=64)])
    assert _kcp_gdn_hkv_signatures(model, ulysses_size=2) == [(16, 128, 128), (16, 128, 64)]
    assert _canonical_kcp_ttx_hkv_payload([(16, 128, 128)]) == "[[16,128,128]]"


def test_hkv_signatures_ignore_non_kcp_and_do_not_read_weights():
    class _MetaGdn(_KcpGdn):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(128, device="meta"))

    model = _ToyModel([_KcpGdn(impl="disabled"), _MetaGdn()])
    assert _kcp_gdn_hkv_signatures(model, ulysses_size=2) == [(16, 128, 128)]


def test_precompile_skips_gpu_and_non_kcp(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    calls = _install_precompile_fakes(monkeypatch)
    monkeypatch.setattr(torch_parallelize, "IS_NPU_AVAILABLE", False)
    _precompile_kcp_ttx_before_fsdp(
        _ToyModel(),
        mixed_precision=MixedPrecisionConfig(),
        parallel_state=_parallel_state(),
    )
    assert calls == []

    monkeypatch.setattr(torch_parallelize, "IS_NPU_AVAILABLE", True)
    _precompile_kcp_ttx_before_fsdp(
        _ToyModel([_KcpGdn(impl="disabled")]),
        mixed_precision=MixedPrecisionConfig(),
        parallel_state=_parallel_state(),
    )
    assert calls == []


def test_identical_gdn_modules_compile_once(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    calls = _install_precompile_fakes(monkeypatch)
    monkeypatch.setattr(torch_parallelize, "IS_NPU_AVAILABLE", True)
    _precompile_kcp_ttx_before_fsdp(
        _ToyModel([_KcpGdn(), _KcpGdn()]),
        mixed_precision=MixedPrecisionConfig(param_dtype="bfloat16"),
        parallel_state=_parallel_state(ulysses_size=2),
    )
    assert len(calls) == 1
    assert calls[0]["shape"] == (1, 0, 16, 128)
    assert calls[0]["value_shape"] == (1, 0, 16, 128)
    assert calls[0]["key_dtype"] is torch.bfloat16
    assert calls[0]["g_dtype"] is torch.float32
    assert calls[0]["beta_dtype"] is torch.bfloat16


def test_signature_mismatch_is_fail_closed(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    def mismatch(payload, *, device):
        raise RuntimeError(f"KCP TTX build-stage precompile signatures differ across ranks; local={payload}")

    _install_precompile_fakes(monkeypatch, identical=mismatch)
    monkeypatch.setattr(torch_parallelize, "IS_NPU_AVAILABLE", True)
    with pytest.raises(RuntimeError, match="signatures differ across ranks"):
        _precompile_kcp_ttx_before_fsdp(
            _ToyModel(),
            mixed_precision=MixedPrecisionConfig(),
            parallel_state=_parallel_state(),
        )


def test_any_rank_compile_failure_is_fail_closed(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic compile failure")

    def coordinate(local_error, *, device, label):
        if local_error is not None:
            raise RuntimeError(f"KCP TTX build-stage precompile failed on this rank ({label})") from local_error

    _install_precompile_fakes(monkeypatch, warmup=boom, coord=coordinate)
    monkeypatch.setattr(torch_parallelize, "IS_NPU_AVAILABLE", True)
    with pytest.raises(RuntimeError, match="failed on this rank"):
        _precompile_kcp_ttx_before_fsdp(
            _ToyModel(),
            mixed_precision=MixedPrecisionConfig(),
            parallel_state=_parallel_state(),
        )


def test_assert_identical_signatures_fail_closed_on_hash_mismatch(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather(gathered, local):
            gathered[0].copy_(local)
            gathered[1].zero_()

    monkeypatch.setattr(torch_parallelize, "dist", FakeDist)
    with pytest.raises(RuntimeError, match="signatures differ across ranks"):
        torch_parallelize._assert_identical_kcp_ttx_signatures("[[16,128,128]]", device=torch.device("cpu"))


def test_coordinate_fail_closed_when_peer_fails(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    class FakeDist:
        ReduceOp = types.SimpleNamespace(MAX="max")

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(flag, op=None):
            flag.fill_(1)

    monkeypatch.setattr(torch_parallelize, "dist", FakeDist)
    with pytest.raises(RuntimeError, match="failed on another rank"):
        torch_parallelize._coordinate_kcp_ttx_precompile_success(
            None, device=torch.device("cpu"), label="H=16,K=128,V=128"
        )


def test_precompile_runs_before_ac_tp_and_fsdp(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    order = []
    monkeypatch.setattr(
        torch_parallelize,
        "_precompile_kcp_ttx_before_fsdp",
        lambda *args, **kwargs: order.append("precompile"),
    )
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: _parallel_state(tp_enabled=True))
    monkeypatch.setattr(
        torch_parallelize,
        "parallelize_module",
        lambda model, **kwargs: order.append("tp") or model,
    )
    monkeypatch.setattr(
        torch_parallelize,
        "parallelize_model_fsdp2",
        lambda model, **kwargs: order.append("fsdp") or model,
    )
    model = _ToyModel()

    def enable_ac(gradient_checkpointing_kwargs=None):
        order.append("ac")
        model.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    model.gradient_checkpointing_enable = enable_ac
    build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_gradient_checkpointing=True,
    )
    assert order == ["precompile", "ac", "tp", "fsdp"]
