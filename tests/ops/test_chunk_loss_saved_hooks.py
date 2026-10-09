"""CPU parity gates for chunk loss under saved tensor hooks and checkpointing."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_module
from veomni.distributed.offloading import OffloadPolicy, build_activation_offloading_context
from veomni.distributed.sequence_parallel.loss import ReduceLoss
from veomni.utils.device import IS_NPU_AVAILABLE


DEVICES = ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires NPU"))]


def _legacy_forward(ctx, hidden, weight, bias, loss_forward, kwargs_chunks, chunk_size):
    loss = torch.tensor(0.0, device=hidden.device)
    dh, dw = torch.empty_like(hidden), torch.zeros_like(weight)
    for x, target, kwargs in zip(hidden.split(chunk_size, 1), dh.split(chunk_size, 1), kwargs_chunks):
        (dx, delta), (value, _) = torch.func.grad_and_value(loss_forward, (0, 1), has_aux=True)(
            x, weight, bias, **kwargs
        )
        loss.add_(value)
        target.copy_(dx)
        dw.add_(delta)
    ctx.save_for_backward(dh, dw)
    return loss


class LegacyChunkLoss(torch.autograd.Function):
    forward = staticmethod(_legacy_forward)
    backward = staticmethod(chunk_module.ChunkLoss.backward)


def _run(monkeypatch, implementation, dtype, sp, hooks, gc, frozen_head=False, all_ignored=False, device="cpu"):
    monkeypatch.setattr(chunk_module, "ChunkLoss", implementation)
    monkeypatch.setattr(chunk_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=sp))
    # Exercise the real reducer including its zero-token guard, with a one-rank
    # collective seam. Multi-rank communication is outside this CPU unit gate.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(
        chunk_module, "reduce_sequence_parallel_loss", lambda loss, count: ReduceLoss.apply(loss, count, object())
    )
    torch.manual_seed(127)
    if device == "npu":
        torch.npu.manual_seed_all(127)
    h = torch.randn(2, 11, 64, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(19, 64, dtype=dtype, device=device, requires_grad=not frozen_head)
    labels = torch.randint(0, 19, (2, 11), device=device)
    labels[:, 1:7] = -100  # whole ignored chunks and a short final chunk
    if all_ignored:
        labels.fill_(-100)
    packed = []
    unpacked = []
    offloaded_devices = []
    restored_devices = []

    def pack(t):
        packed.append(t.shape)
        return t.detach().clone()

    def unpack(t):
        unpacked.append(t.shape)
        return t

    def forward(x, head):
        # Upstream graph ensures local autograd does not accidentally consume
        # the decoder graph or its accumulated gradients.
        return chunk_module.chunk_loss_function(x.sin(), head, labels, chunk_size=3, vocab_size=19)[0]

    fwd_context = torch.autograd.graph.saved_tensors_hooks(pack, unpack) if hooks else nullcontext()
    bwd_context = nullcontext()
    if hooks == "production":
        fwd_context, bwd_context = build_activation_offloading_context(
            enable_activation=True, enable_gradient_checkpointing=gc
        )
        # Both BF16 buffers exceed the production 1024-byte threshold. Audit
        # the real pack/unpack hooks rather than replacing their transfers.
        assert h.numel() * h.element_size() > 1024
        assert w.numel() * w.element_size() > 1024
        for context in (fwd_context, bwd_context):

            def audit_pack(tensor, original=context.pack_hook):
                value = original(tensor)
                if value[0] == OffloadPolicy.OFFLOAD:
                    assert value[2].device.type == "cpu"
                    offloaded_devices.append(value[1].type)
                return value

            def audit_unpack(value, original=context.unpack_hook):
                tensor = original(value)
                if value[0] == OffloadPolicy.OFFLOAD:
                    assert tensor.device == value[1]
                    restored_devices.append(tensor.device.type)
                return tensor

            context.pack_hook = audit_pack
            context.unpack_hook = audit_unpack
    with fwd_context:
        loss = checkpoint(forward, h, w, use_reentrant=False) if gc else forward(h, w)
    with bwd_context:
        (loss * 0.37).backward()  # non-unit upstream gradient (e.g. accumulation)
    if hooks == "production":
        assert device in offloaded_devices
        assert device in restored_devices
    if hooks and hooks != "production":
        assert packed and unpacked
    return loss.detach(), h.grad, w.grad


def test_original_transform_rejects_saved_tensor_hooks():
    x = torch.ones(2)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        with pytest.raises(RuntimeError, match="don't yet support saved tensor hooks"):
            torch.func.grad_and_value(lambda t: t.square().sum())(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize(
    "hooks,gc", [(False, False), (True, False), (False, True), (True, True), ("production", True)]
)
def test_loss_and_gradients_match_legacy(monkeypatch, dtype, sp, hooks, gc, device):
    fixed = chunk_module.ChunkLoss
    expected = _run(monkeypatch, LegacyChunkLoss, dtype, sp, False, False, device=device)
    actual = _run(monkeypatch, fixed, dtype, sp, hooks, gc, device=device)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_head_and_upstream_graph(monkeypatch, dtype, device):
    fixed = chunk_module.ChunkLoss
    expected = _run(monkeypatch, LegacyChunkLoss, dtype, False, False, False, frozen_head=True, device=device)
    actual = _run(monkeypatch, fixed, dtype, False, True, True, frozen_head=True, device=device)
    assert actual[2] is None
    for got, ref in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_valid_sp_slice(monkeypatch, dtype, device):
    fixed = chunk_module.ChunkLoss
    expected = _run(monkeypatch, LegacyChunkLoss, dtype, True, False, False, all_ignored=True, device=device)
    actual = _run(monkeypatch, fixed, dtype, True, True, True, all_ignored=True, device=device)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        assert torch.isfinite(got).all()
        assert torch.count_nonzero(got) == 0
