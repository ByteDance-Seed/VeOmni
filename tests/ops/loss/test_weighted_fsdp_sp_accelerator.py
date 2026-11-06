"""Distributed accelerator parity for weighted CE through FSDP2 and SP."""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh


if os.environ.get("VEOMNI_TEST_REQUIRE_NPU") == "1":
    importlib.import_module("torch_npu")

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities  # noqa: E402
from veomni.utils.device import get_device_type  # noqa: E402


_ACCELERATOR_16 = LaunchSpec(nproc=16, device="accelerator")
_CAPABILITIES = runner_capabilities()


def _check_fsdp_sp(dtype, all_zero):
    """Compare sharded accelerator loss and gradients with a dense reference."""
    from veomni.models.loss_utils import cross_entropy_loss as causal
    from veomni.ops import VeomniOp

    device = torch.device(get_device_type(), dist.get_rank())
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 16 and device.type != "cpu"
    torch.manual_seed(42)
    all_hidden = torch.randn(size * 2, 8, dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)
    initial = torch.randn(32, 8, dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)
    labels = (torch.arange(size * 2, device=device) % 32).long()
    weights = (torch.arange(size * 2, device=device) % 5).float() / 4
    weights[:2] = 0  # rank zero has no supervised tokens
    if all_zero:
        weights.zero_()
    mesh = init_device_mesh(device.type, (size,))
    causal.get_parallel_state = lambda: SimpleNamespace(sp_enabled=True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(initial.clone())
            self.op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")

        def forward(self, hidden, target, loss_weights):
            loss, _, _ = causal.ForCausalLMLoss(
                hidden_states=hidden,
                labels=target,
                weights=self.weight,
                loss_weights=loss_weights,
                op=self.op,
                loss_reduction_group=dist.group.WORLD,
            )
            return loss

    model = Model().to(device)
    fully_shard(model, mesh=mesh)
    local_hidden = all_hidden[rank * 2 : rank * 2 + 2].clone().requires_grad_()
    actual = model(local_hidden, labels[rank * 2 : rank * 2 + 2], weights[rank * 2 : rank * 2 + 2])
    actual.backward()
    actual_weight_grad = model.weight.grad.full_tensor()

    ref_hidden = all_hidden.clone().requires_grad_()
    ref_weight = initial.clone().requires_grad_()
    expected = (
        F.cross_entropy(F.linear(ref_hidden, ref_weight).float(), labels, reduction="none") * weights
    ).sum() / weights.sum().clamp_min(1)
    expected.backward()
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-5
    if all_zero:
        assert actual.item() == 0
        assert torch.count_nonzero(actual_weight_grad).item() == 0
        assert torch.count_nonzero(local_hidden.grad).item() == 0
    else:
        ref_local = ref_hidden.grad[rank * 2 : rank * 2 + 2]
        for got, ref in ((actual_weight_grad, ref_weight.grad), (local_hidden.grad / size, ref_local)):
            error = (got.float() - ref.float()).norm()
            scale = ref.float().norm()
            assert error <= 0.05 * scale + 1e-7
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_weight_grad, ref_weight.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        local_hidden.grad / size, ref_hidden.grad[rank * 2 : rank * 2 + 2], atol=tolerance, rtol=tolerance
    )


@pytest.mark.skipif(
    os.environ.get("VEOMNI_TEST_REQUIRE_NPU") != "1"
    and (_CAPABILITIES.backend == "cpu" or len(_CAPABILITIES.devices) < 16),
    reason="requires 16 accelerators",
)
@pytest.mark.launch_plan(provider=_ACCELERATOR_16)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("all_zero", [False, True], ids=["mixed", "zero"])
def test_weighted_loss_fsdp2_sequence_parallel_on_accelerator(dtype, all_zero):
    """Exercise mixed and zero supervision masses through FSDP2 plus SP."""
    from tests.tools.launch_utils import torchrun

    if os.environ.get("VEOMNI_TEST_REQUIRE_NPU") == "1":
        assert get_device_type() == "npu" and torch.npu.is_available() and torch.npu.device_count() == 16
    elif get_device_type() == "cpu":
        pytest.skip("requires 16 accelerators")
    torchrun(_check_fsdp_sp, _ACCELERATOR_16.nproc, dtype, all_zero)
