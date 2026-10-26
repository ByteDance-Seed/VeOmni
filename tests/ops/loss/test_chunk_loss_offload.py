"""Exercise the registered chunk CE under production offload/checkpoint hooks."""

from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from veomni.distributed.offloading import OffloadPolicy, build_activation_offloading_context
from veomni.ops import VeomniOp
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from veomni.utils.loss_observer import capture_chunk_loss_per_token


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires CUDA")),
    pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU")),
]


@pytest.fixture(scope="module", autouse=True)
def _deterministic_npu_comparisons():
    """Keep exact offload comparisons independent of NPU reduction scheduling."""
    if not IS_NPU_AVAILABLE:
        yield
        return
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _run(device, dtype, *, offload, recompute, frozen, all_ignored, bias_kind=None, observed=None):
    torch.manual_seed(127)
    hidden = torch.randn(2, 23, 64).to(device=device, dtype=dtype).requires_grad_(frozen != "hidden")
    weight = torch.randn(137, 64).to(device=device, dtype=dtype).requires_grad_(frozen != "weight")
    labels = torch.randint(0, 137, (2, 23), device=device)
    labels[:, 1:8] = -100
    if all_ignored:
        labels.fill_(-100)
    bias = bias_weight = None
    if bias_kind is not None:
        bias = torch.randn(2, 23, 4 if bias_kind == "factored" else 137, device=device, dtype=dtype)
        bias.requires_grad_()
        if bias_kind == "factored":
            bias_weight = torch.randn(137, 4, device=device, dtype=dtype).requires_grad_()
    op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
    # Even the last (2 x 3) token chunk is eligible for the >1 KiB offload policy.
    assert 2 * 3 * weight.size(0) * torch.float32.itemsize > 1024
    offloaded, restored = [], []
    contexts = (nullcontext(), nullcontext())
    if offload:
        contexts = build_activation_offloading_context(enable_activation=True, enable_gradient_checkpointing=recompute)
        for context in contexts:
            if not hasattr(context, "pack_hook"):
                continue
            original_pack, original_unpack = context.pack_hook, context.unpack_hook

            def pack(tensor, original=original_pack):
                packed = original(tensor)
                if packed[0] == OffloadPolicy.OFFLOAD:
                    assert packed[2].device.type == "cpu"
                    offloaded.append((tuple(tensor.shape), tensor.device.type))
                return packed

            def unpack(packed, original=original_unpack):
                tensor = original(packed)
                if packed[0] == OffloadPolicy.OFFLOAD:
                    assert tensor.device == packed[1]
                    restored.append(tensor.device.type)
                return tensor

            context.pack_hook, context.unpack_hook = pack, unpack

    def body(x, head, bias, bias_weight):
        return op(x.sin(), labels, head, bias, bias_weight, chunk_size=5)

    capture = capture_chunk_loss_per_token(observed.append) if observed is not None else nullcontext()
    with capture, contexts[0]:
        inputs = (hidden, weight, bias, bias_weight)
        loss = checkpoint(body, *inputs, use_reentrant=False) if recompute else body(*inputs)
    with contexts[1]:
        loss.backward()
    if offload:
        assert offloaded and restored
        assert all(
            len(shape) < 2 or shape[-1] != 137 or (bias_kind == "full" and shape == tuple(bias.shape))
            for shape, _ in offloaded
        ), "Transient vocabulary logits escaped the chunk"
        if device != "cpu":
            assert any(source == device for _, source in offloaded)
            assert device in restored
    return (
        loss.detach(),
        hidden.grad,
        weight.grad,
        bias.grad if bias is not None else None,
        bias_weight.grad if bias_weight is not None else None,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("frozen", [None, "hidden", "weight"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_registered_chunk_loss_offload_preserves_loss_and_gradients(device, dtype, recompute, frozen, all_ignored):
    expected = _run(device, dtype, offload=False, recompute=recompute, frozen=frozen, all_ignored=all_ignored)
    actual = _run(device, dtype, offload=True, recompute=recompute, frozen=frozen, all_ignored=all_ignored)
    for value, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert value is None
        else:
            assert torch.isfinite(value).all()
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("bias_kind", ["full", "factored"])
def test_npu_chunk_observer_bias_offload_parity(dtype, recompute, bias_kind):
    """Keep loss, all gradients and observations exact with native bias/offload."""
    kwargs = dict(recompute=recompute, frozen=None, all_ignored=False, bias_kind=bias_kind)
    expected = _run("npu", dtype, offload=False, **kwargs)
    reference_tokens, actual_tokens = [], []
    observed = _run("npu", dtype, offload=False, observed=reference_tokens, **kwargs)
    actual = _run("npu", dtype, offload=True, observed=actual_tokens, **kwargs)
    for result in (observed, actual):
        for value, reference in zip(result, expected, strict=True):
            if reference is None:
                assert value is None
            else:
                assert torch.isfinite(value).all()
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert len(reference_tokens) == len(actual_tokens) == 1
    torch.testing.assert_close(actual_tokens[0], reference_tokens[0], rtol=0, atol=0)
