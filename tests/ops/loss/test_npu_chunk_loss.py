"""Native chunk CE parity and conservative fallback coverage."""

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import resolve_op
from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss, npu
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU")
@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(13, 8), (2, 13, 8)])
@pytest.mark.parametrize("frozen", [None, "hidden", "weight", "both"])
@pytest.mark.parametrize("ignore_index", [-100, 2, 999])
def test_npu_chunk_parity(dtype, shape, frozen, ignore_index):
    """Compare eager loss and gradients across dtypes, shapes and frozen inputs."""
    torch.manual_seed(127)
    hidden = torch.randn(shape, device="npu", dtype=dtype).requires_grad_(frozen not in ("hidden", "both"))
    weight = torch.randn(19, 8, device="npu", dtype=dtype).requires_grad_(frozen not in ("weight", "both"))
    labels = torch.randint(0, 19, shape[:-1], device="npu")
    labels[..., :5] = ignore_index
    logits = F.linear(hidden, weight).float()
    reference = (
        F.cross_entropy(logits.reshape(-1, 19), labels.reshape(-1), ignore_index=ignore_index, reduction="sum") / 31
    )
    op = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    actual = op(hidden, labels, weight, ignore_index=ignore_index, num_items_in_batch=torch.tensor(31), chunk_size=4)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    inputs = tuple(value for value in (hidden, weight) if value.requires_grad)
    if inputs:
        expected = torch.autograd.grad(reference * 2.75, inputs)
        grads = torch.autograd.grad(actual * 2.75, inputs)
        for value, target in zip(grads, expected, strict=True):
            torch.testing.assert_close(value, target, atol=2e-2 if dtype != torch.float32 else 2e-6, rtol=2e-2)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU")
@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.parametrize("count", [None, 0, 37, torch.tensor(0), torch.tensor(37)])
def test_npu_all_ignore_compacts_before_projection(monkeypatch, count):
    """Verify ignored chunks project zero rows and produce zero loss and gradients."""
    hidden = torch.randn(2, 9, 7, device="npu", requires_grad=True)
    weight = torch.randn(19, 7, device="npu", requires_grad=True)
    labels = torch.full((2, 9), -100, device="npu")
    original = F.linear
    rows = []

    def record(input, weight, bias=None):
        """Record the compacted row count before forwarding to the real Linear."""
        rows.append(input.shape[0])
        return original(input, weight, bias)

    monkeypatch.setattr(npu.F, "linear", record)
    loss = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper(
        hidden, labels, weight, chunk_size=4, num_items_in_batch=count
    )
    grads = torch.autograd.grad(loss, (hidden, weight))
    assert rows == [0, 0, 0]
    assert loss == 0
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("mode", ["bias", "factored_bias", "observer"])
def test_special_arguments_bypass_native_ce(monkeypatch, mode):
    """Keep full/factored bias and token observation on the eager CE path."""
    hidden, weight = torch.randn(3, 4), torch.randn(7, 4)
    labels = torch.tensor([0, 1, 2])
    monkeypatch.setattr(chunk_loss, "can_use_fused_ce", lambda *args: True)

    def fail(*args):
        """Fail if a special-argument call reaches native CE."""
        raise AssertionError("native CE must not consume bias or observer calls")

    monkeypatch.setattr(chunk_loss, "fused_linear_ce", fail)
    kwargs = {}
    if mode == "observer":
        observed = []
        kwargs["observe"] = lambda logits, labels: observed.append(logits.shape)
    else:
        kwargs["logits_bias"] = torch.randn(3, 2 if mode == "factored_bias" else 7)
        if mode == "factored_bias":
            kwargs["logits_bias_weight"] = torch.randn(7, 2)
    loss = chunk_loss._ce_loss_func(hidden, weight, labels, 3, -100, **kwargs)
    assert torch.isfinite(loss)
    if mode == "observer":
        assert observed == [torch.Size([3, 7])]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_native_guard_rejects_cpu():
    """Reject CPU tensors even when the optional native NPU backend is installed."""
    assert not npu.can_use_fused_ce(torch.randn(3, 4), torch.randn(7, 4), torch.tensor([0, 1, 2]))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("ignore_index", [2, 999, -1])
def test_custom_ignore_rejects_remaining_native_sentinel(ignore_index):
    """Reject an unmasked native sentinel when the caller selects another ignore index."""
    with pytest.raises(IndexError, match="out of bounds"):
        npu.fused_linear_ce(
            torch.randn(3, 4), torch.randn(7, 4), torch.tensor([0, -100, ignore_index]), 2, ignore_index
        )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU")
@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_native_guard_rejects_unsupported_inputs(monkeypatch):
    """Reject unsupported labels, devices, dtypes, row counts and compilation."""
    hidden = torch.randn(3, 4, device="npu")
    weight = torch.randn(7, 4, device="npu")
    labels = torch.tensor([0, 1, 2], device="npu")
    assert npu.can_use_fused_ce(hidden, weight, labels)
    assert not npu.can_use_fused_ce(hidden, weight, labels.to(torch.int32))
    assert not npu.can_use_fused_ce(hidden, weight.cpu(), labels)
    assert not npu.can_use_fused_ce(hidden.to(torch.bfloat16), weight, labels)
    assert not npu.can_use_fused_ce(hidden[:1].expand(200001, -1), weight, labels[:1].expand(200001))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not npu.can_use_fused_ce(hidden, weight, labels)
