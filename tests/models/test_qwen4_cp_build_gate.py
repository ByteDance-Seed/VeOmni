"""Exercise the real model factory CP gate without allocating accelerator state."""

from types import SimpleNamespace

import pytest

from veomni.models import auto


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_qwen_cp_build_gate(monkeypatch, npu, model_type, cp_size):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: npu)
    state = SimpleNamespace(cp_enabled=True, cp_size=cp_size, ulysses_size=8, allow_hybrid_cp=True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: state)
    auto.check_context_parallel_supported(SimpleNamespace(model_type=model_type))
    state.allow_hybrid_cp = False
    with pytest.raises(NotImplementedError, match="explicit allow_hybrid_cp"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type=model_type))
    state.allow_hybrid_cp = True
    state.cp_size = 8
    with pytest.raises(NotImplementedError, match="explicit allow_hybrid_cp"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type=model_type))


def test_other_npu_models_remain_rejected(monkeypatch):
    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True))
    with pytest.raises(NotImplementedError, match="GPU-only"):
        auto.check_context_parallel_supported(SimpleNamespace(model_type="deepseek_v4"))
