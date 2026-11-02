"""Submit independent dX before ordered communication waits, including errors."""

import gc
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.parallel.test_moe_ep_load_balance_dispatch import ep_load_balance as ep
from tests.parallel.test_moe_ep_load_balance_dispatch import moe_layer as moe


@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("fail_dx", [False, True])
def test_down_wait_follows_dx_and_precedes_gate_weight_grad(monkeypatch, needs, deferred, fail_dx):
    events = []
    tokens = torch.randn(4, 3, dtype=torch.bfloat16)
    gate = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    down = torch.randn(1, 3, 2, dtype=torch.bfloat16)
    pre_gate = torch.randn(4, 2, dtype=torch.bfloat16)
    pre_up = torch.randn(4, 2, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        saved_tensors=(tokens, torch.tensor([4]), gate, down, pre_gate, pre_up, torch.empty(0), torch.empty(0)),
        swiglu_limit=None,
    )
    calls = 0

    def nk(*, a, b, **kwargs):
        nonlocal calls
        calls += 1
        events.append(f"dx{calls}")
        if fail_dx and calls == 2:
            raise RuntimeError("independent dx failure")
        return torch.ones(a.shape[0], b.shape[-1], dtype=a.dtype)

    def sync(**kwargs):
        events.append("gate_sync" if kwargs["c"].shape == gate.shape else "down_sync")
        kwargs["c"].zero_()
        # The original non-deferred helper returns a Tensor, not a work handle.
        return kwargs["c"]

    if deferred:

        def start(**kwargs):
            events.append("down_start")

            def wait():
                events.append("down_wait")
                kwargs["c"].zero_()

            return SimpleNamespace(wait=wait)

        sync.start = start

    monkeypatch.setattr(moe, "group_gemm_same_nk", nk)

    def backward():
        return moe.EPMergedFc1GroupGemm.backward(
            ctx,
            torch.ones_like(tokens),
            weight_gradient_dtype=torch.float32,
            weight_needs_grad=needs,
            weight_grad_fn=sync,
        )

    if fail_dx:
        with pytest.raises(RuntimeError, match="independent dx failure"):
            backward()
    else:
        outputs = backward()
        assert (outputs[2] is not None) == needs[0]
        assert (outputs[3] is not None) == needs[1]
    expected = ["dx1"]
    if needs[1]:
        expected.append("down_start" if deferred else "down_sync")
    expected.append("dx2")
    if needs[1] and deferred:
        expected.append("down_wait")
    if needs[0] and not fail_dx:
        expected.append("gate_sync")
    assert events == expected


def test_pending_handle_retains_operands_and_waits_once():
    spec = importlib.util.spec_from_file_location("pending_ep_grad_test", Path(ep.__file__).with_name("ep_grad.py"))
    grad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grad)
    events = []

    class Operand:
        value = 7

    operand = Operand()
    reference = weakref.ref(operand)

    class Work:
        def wait(self):
            events.append("wait")

    def stages(value):
        events.append("submitted_empty_stage")
        yield []
        events.append("submitted_next_stage")
        yield [Work()]
        assert value.value == 7
        events.append("suffix")

    pending = grad._PendingOrderedWeightGrad(stages(operand))
    del operand
    gc.collect()
    assert reference() is not None
    events.append("independent_dx")
    pending.wait()
    pending.wait()
    gc.collect()
    assert reference() is None
    assert events == ["submitted_empty_stage", "independent_dx", "submitted_next_stage", "wait", "suffix"]
