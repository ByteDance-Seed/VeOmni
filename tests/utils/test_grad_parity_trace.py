from types import SimpleNamespace

import pytest
import torch

import veomni.distributed.clip_grad_norm as clip_module


class _TinyLinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(2, 2, bias=False)
        self.A_log = torch.nn.Parameter(torch.zeros(2))


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _TinyLinearAttention()
        self.unrelated = torch.nn.Linear(2, 2, bias=False)
        self._extra_parallel_param_groups = {
            "non_extra_parallel": list(self.parameters()),
            "ep": [],
        }


@pytest.mark.parametrize("value", ["", "-1", "1.0", "true"])
def test_grad_parity_trace_rejects_invalid_step_limits(monkeypatch, value):
    monkeypatch.setenv("VEOMNI_GRAD_PARITY_TRACE_STEPS", value)
    with pytest.raises(ValueError, match="non-negative base-10 integer"):
        clip_module._grad_parity_trace_steps()


def test_grad_parity_trace_collects_only_gdn_non_extra_grads(monkeypatch):
    model = _TinyModel()
    model.linear_attn.in_proj_qkv.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    model.linear_attn.A_log.grad = torch.tensor([1.0, -2.0])
    model.unrelated.weight.grad = torch.full_like(model.unrelated.weight, 1000.0)
    monkeypatch.setattr(clip_module, "get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))

    trace = clip_module._collect_gdn_grad_parity_trace(model)

    assert trace["groups"]["in_proj_qkv"] == {
        "l2": 5.0,
        "signed_sum": 7.0,
        "max_abs": 4.0,
        "numel": 4,
    }
    assert trace["groups"]["A_log"] == {
        "l2": pytest.approx(5**0.5),
        "signed_sum": -1.0,
        "max_abs": 2.0,
        "numel": 2,
    }
    assert trace["groups"]["gdn_all"]["l2"] == pytest.approx(30**0.5)
    assert trace["groups"]["gdn_all"]["numel"] == 6
    assert trace["groups"]["non_extra_all"]["l2"] > 1000


def test_grad_parity_trace_rejects_gdn_param_in_extra_parallel_bucket(monkeypatch):
    model = _TinyModel()
    param = model.linear_attn.A_log
    param.grad = torch.ones_like(param)
    model._extra_parallel_param_groups["non_extra_parallel"] = [
        candidate for candidate in model._extra_parallel_param_groups["non_extra_parallel"] if candidate is not param
    ]
    model._extra_parallel_param_groups["ep"].append(param)
    monkeypatch.setattr(clip_module, "get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))

    with pytest.raises(RuntimeError, match="expected a non-extra FSDP parameter"):
        clip_module._collect_gdn_grad_parity_trace(model)


def test_grad_parity_trace_infers_extra_parallel_l2_without_an_extra_collective(monkeypatch):
    model = _TinyModel()
    model.linear_attn.A_log.grad = torch.tensor([3.0, 4.0])
    model.unrelated.weight.grad = torch.zeros_like(model.unrelated.weight)
    monkeypatch.setenv("VEOMNI_GRAD_PARITY_TRACE_STEPS", "1")
    monkeypatch.setattr(clip_module, "get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(13.0))

    result = clip_module.veomni_clip_grad_norm(model, 1.0)

    assert result == 13.0
    assert model._veomni_grad_parity_trace["groups"]["non_extra_all"]["l2"] == 5.0
    assert model._veomni_grad_parity_trace["extra_parallel_inferred_l2"] == 12.0
