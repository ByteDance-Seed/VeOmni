import copy
import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "master", Path(__file__).parents[2] / "veomni/optim/master_fp32_adamw.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
