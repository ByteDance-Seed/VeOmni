import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


spec = importlib.util.spec_from_file_location(
    "master", Path(__file__).parents[2] / "veomni/optim/master_fp32_adamw.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("optimizer_class", [module.MasterFP32AdamW, module.FullFP32AdamW])
def test_resume_preserves_state_without_bf16_temporary_copies(optimizer_class):
    class BF16CopyCounter(TorchDispatchMode):
        def __init__(self):
            self.copied_elements = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func is torch.ops.aten._to_copy.default and args[0].dtype == torch.float32:
                if result.dtype == torch.bfloat16 and result.numel() > 1:
                    self.copied_elements += result.numel()
            return result

    param = torch.nn.Parameter(torch.linspace(-1, 1, 257).bfloat16())
    dormant = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    original = optimizer_class([param, dormant], lr=1e-4)
    param.grad = torch.linspace(-0.1, 0.1, 257).bfloat16()
    original.step()
    checkpoint = copy.deepcopy(original.state_dict())

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored_dormant = torch.nn.Parameter(dormant.detach().clone())
    resumed = optimizer_class([restored_param, restored_dormant], lr=3e-4)
    with BF16CopyCounter() as copies:
        resumed.load_state_dict(checkpoint)

    assert copies.copied_elements == 0
    assert resumed.param_groups[0]["lr"] == original.param_groups[0]["lr"]
    assert restored_dormant not in resumed.state
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[restored_param][name], original.state[param][name], rtol=0, atol=0)

    next_grad = torch.linspace(0.1, -0.1, 257).bfloat16()
    param.grad = next_grad
    restored_param.grad = next_grad.clone()
    original.step()
    resumed.step()
    torch.testing.assert_close(restored_param, param, rtol=0, atol=0)
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[restored_param][name], original.state[param][name], rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_class", [module.MasterFP32AdamW, module.FullFP32AdamW])
def test_resume_honors_load_hooks_and_unassigned_state(optimizer_class):
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    source = optimizer_class([param], lr=1e-4)
    param.grad = torch.arange(4).bfloat16()
    source.step()
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["state"]["global_metadata"] = {"version": 7}

    target = torch.nn.Parameter(param.detach().clone())
    resumed = optimizer_class([target], lr=3e-4)
    events = []

    def before_load(optimizer, incoming):
        events.append("before")
        assert optimizer is resumed
        key = incoming["param_groups"][0]["params"][0]
        assert incoming["state"][key]["master_weight"].dtype == torch.float32
        replacement = copy.deepcopy(incoming)
        replacement["param_groups"][0]["lr"] = 0.002
        replacement["state"][key]["exp_avg"].add_(1)
        return replacement

    def after_load(optimizer):
        events.append("after")
        assert optimizer.state[target]["master_weight"].dtype == torch.float32
        assert optimizer.state["global_metadata"] == {"version": 7}
        optimizer.param_groups[0]["post_load_seen"] = True

    resumed.register_load_state_dict_pre_hook(before_load)
    resumed.register_load_state_dict_post_hook(after_load)
    resumed.load_state_dict(checkpoint)

    assert events == ["before", "after"]
    assert resumed.param_groups[0]["lr"] == 0.002
    assert resumed.param_groups[0]["post_load_seen"]
    torch.testing.assert_close(resumed.state[target]["exp_avg"], source.state[param]["exp_avg"] + 1, rtol=0, atol=0)


def test_master_updates_and_reload():
    p = torch.nn.Parameter(torch.ones(257, dtype=torch.bfloat16))
    opt = module.MasterFP32AdamW([p], lr=1e-5, weight_decay=0.1, chunk_size=64)
    ref = p.detach().float().clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    for step in range(1, 301):
        grad = (torch.sin(torch.arange(p.numel()).float() + step) * 0.01).bfloat16()
        p.grad = grad
        first = m.float().lerp_(grad.float(), 0.1)
        second = v.float().mul_(0.95).addcmul_(grad.float(), grad.float(), value=0.05)
        m.copy_(first)
        v.copy_(second)
        denom = second.sqrt().div_((1 - 0.95**step) ** 0.5).add_(1e-8)
        ref.mul_(1 - 1e-6).addcdiv_(first, denom, value=-1e-5 / (1 - 0.9**step))
        opt.step()
        state = opt.state[p]
        assert state["master_weight"].dtype == torch.float32
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.bfloat16
        torch.testing.assert_close(state["master_weight"], ref, rtol=0, atol=0)
        torch.testing.assert_close(p, ref.bfloat16(), rtol=0, atol=0)
        if step == 150:
            resumed = module.MasterFP32AdamW([p])
            resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
            opt = resumed
    assert not torch.equal(ref, torch.ones_like(ref))


def test_full_fp32_moments_reference_and_reload():
    torch.manual_seed(17)
    p = torch.nn.Parameter(torch.randn(257, dtype=torch.bfloat16))
    ref = torch.nn.Parameter(p.detach().float().clone())
    opt = module.FullFP32AdamW([p], lr=1e-5, weight_decay=0.1, chunk_size=64)
    reference = torch.optim.AdamW([ref], lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    for step in range(50):
        grad = torch.randn_like(p) * 0.01
        p.grad = grad
        ref.grad = grad.float()
        opt.step()
        reference.step()
        torch.testing.assert_close(opt.state[p]["master_weight"], ref, rtol=1e-6, atol=1e-7)
        for key in ("exp_avg", "exp_avg_sq"):
            assert opt.state[p][key].dtype == torch.float32
            torch.testing.assert_close(opt.state[p][key], reference.state[ref][key], rtol=1e-6, atol=1e-10)
        if step == 24:
            state = copy.deepcopy(opt.state_dict())
            resumed = module.FullFP32AdamW([p])
            resumed.load_state_dict(state)
            for key in ("master_weight", "exp_avg", "exp_avg_sq"):
                assert torch.equal(resumed.state[p][key], opt.state[p][key])
            opt = resumed
