"""Exercise the registered chunk CE under production offload/checkpoint hooks."""

from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from veomni.distributed.offloading import OffloadPolicy, build_activation_offloading_context
from veomni.ops import VeomniOp
from veomni.utils.device import IS_NPU_AVAILABLE


DEVICES = ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires native NPU"))]


def _run(device, dtype, *, offload, recompute, frozen, all_ignored):
    torch.manual_seed(127)
    hidden = torch.randn(2, 23, 64).to(device=device, dtype=dtype).requires_grad_(frozen != "hidden")
    weight = torch.randn(37, 64).to(device=device, dtype=dtype).requires_grad_(frozen != "weight")
    labels = torch.randint(0, 37, (2, 23), device=device)
    labels[:, 1:8] = -100
    if all_ignored:
        labels.fill_(-100)
    op = VeomniOp("cross_entropy_loss", "standard", "chunk_loss")
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

    def body(x, head):
        return op(x.sin(), labels, head, chunk_size=5)

    with contexts[0]:
        loss = checkpoint(body, hidden, weight, use_reentrant=False) if recompute else body(hidden, weight)
    with contexts[1]:
        loss.backward()
    if offload:
        assert offloaded and restored
        assert all(len(shape) < 2 or shape[-1] != 37 for shape, _ in offloaded), (
            "Transient vocabulary logits escaped the chunk"
        )
        if device == "npu":
            assert any(source == "npu" for _, source in offloaded)
            assert "npu" in restored
    return loss.detach(), hidden.grad, weight.grad


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
