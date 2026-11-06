import pytest
import torch

from veomni.optim import optimizer as optimizer_module


def test_muon_rejects_only_trainable_selectively_offloaded_parameters(monkeypatch):
    model = torch.nn.Linear(4, 4, bias=False)
    parameter = model.weight
    parameter._veomni_optimizer_compute_device = "npu"
    with pytest.raises(ValueError, match="DistributedMuon does not support selectively CPU-offloaded"):
        optimizer_module.build_optimizer(model, optimizer_type="muon")
    sentinel = object()
    monkeypatch.setattr(optimizer_module, "_build_muon_with_adamw", lambda *args, **kwargs: sentinel)
    parameter.requires_grad_(False)
    assert optimizer_module.build_optimizer(model, optimizer_type="muon") is sentinel
