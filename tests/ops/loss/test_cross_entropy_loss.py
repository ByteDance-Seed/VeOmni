# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-entropy eager vs HF ``fixed_cross_entropy``, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.loss.loss_utils import fixed_cross_entropy

from tests.ops.tol import (
    CE_FUSED_ATOL,
    CE_FUSED_GRAD_ATOL,
    CE_FUSED_GRAD_RTOL,
    CE_FUSED_RTOL,
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
)
from tests.ops.utils import make_grad_leaf
from veomni.ops import resolve_op
from veomni.utils.device import IS_CUDA_AVAILABLE


class _FullWeightAddCounter(TorchDispatchMode):
    def __init__(self, weight_shape):
        super().__init__()
        self.weight_shape = weight_shape
        self.out_of_place_adds = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        output = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.add.Tensor and isinstance(output, Tensor) and output.shape == self.weight_shape:
            self.out_of_place_adds += 1
        return output


def _empty_weight(device: torch.device | str) -> Tensor:
    return torch.empty(0, device=device)


@pytest.mark.parametrize(
    ("seed", "num_tokens", "num_classes", "ignore_first", "num_items_in_batch"),
    (
        (0, 8, 16, True, None),
        (2, 10, 7, False, 6),
    ),
    ids=("mean-reduction", "explicit-item-count"),
)
def test_eager_logits_match_hf(seed, num_tokens, num_classes, ignore_first, num_items_in_batch):
    torch.manual_seed(seed)
    logits = torch.randn(num_tokens, num_classes, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (num_tokens,))
    if ignore_first:
        labels[0] = -100

    logits_h = make_grad_leaf(logits)
    out_h = fixed_cross_entropy(logits_h, labels, num_items_in_batch=num_items_in_batch)

    logits_e = make_grad_leaf(logits)
    out_e = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(
        logits_e,
        labels,
        _empty_weight(logits.device),
        num_items_in_batch=num_items_in_batch,
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    assert torch.allclose(logits_e.grad, logits_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


@pytest.mark.parametrize(
    ("hidden_requires_grad", "weight_requires_grad", "expected_argnums"),
    (
        (True, True, (0, 1)),
        (True, False, (0,)),
        (False, True, (1,)),
        (False, False, None),
    ),
    ids=("all-gradients", "frozen-weight", "frozen-hidden", "inference"),
)
def test_eager_hidden_skips_unneeded_grad_and_value(
    monkeypatch: pytest.MonkeyPatch,
    hidden_requires_grad: bool,
    weight_requires_grad: bool,
    expected_argnums: tuple[int, ...] | None,
):
    """Function.forward disables grad, so unused V×H grads must use requires_grad."""
    calls: list[object] = []
    real = torch.func.grad_and_value

    def wrapped(fn, *args, **kwargs):
        calls.append(kwargs.get("argnums", args[1] if len(args) > 1 else None))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(torch.func, "grad_and_value", wrapped)
    torch.manual_seed(4)
    hidden = torch.randn(2, 4, 8, requires_grad=hidden_requires_grad)
    weight = torch.randn(6, 8, requires_grad=weight_requires_grad)
    labels = torch.randint(0, 6, (2, 4))
    loss = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(hidden, labels, weight)
    assert calls == ([] if expected_argnums is None else [expected_argnums])
    assert torch.isfinite(loss).all()
    if hidden_requires_grad or weight_requires_grad:
        loss.backward()
    if hidden_requires_grad:
        assert hidden.grad is not None
    else:
        assert hidden.grad is None
    if weight_requires_grad:
        assert weight.grad is not None
    else:
        assert weight.grad is None


@pytest.mark.parametrize(
    "impl",
    (
        pytest.param("chunk_loss", id="chunk-loss"),
        pytest.param(
            "liger_kernel",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU"),
            id="liger-kernel",
        ),
    ),
)
@pytest.mark.parametrize(
    ("hidden_requires_grad", "weight_requires_grad", "expected_argnums"),
    (
        (True, True, (0, 1)),
        (True, False, (0,)),
        (False, True, (1,)),
        (False, False, None),
    ),
    ids=("all-gradients", "frozen-weight", "frozen-hidden", "inference"),
)
def test_ce_rows_skip_unneeded_grads(
    monkeypatch: pytest.MonkeyPatch,
    impl: str,
    hidden_requires_grad: bool,
    weight_requires_grad: bool,
    expected_argnums: tuple[int, ...] | None,
):
    """Every CE row honors caller mode and per-input requires_grad."""
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")
    calls: list[object] = []
    real = torch.func.grad_and_value

    def wrapped(fn, *args, **kwargs):
        calls.append(kwargs.get("argnums", args[1] if len(args) > 1 else None))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(torch.func, "grad_and_value", wrapped)
    torch.manual_seed(4)
    device = "cuda" if impl == "liger_kernel" else "cpu"
    dtype = torch.bfloat16 if impl == "liger_kernel" else torch.float32
    hidden = torch.randn(2, 4, 8, device=device, dtype=dtype, requires_grad=hidden_requires_grad)
    weight = torch.randn(6, 8, device=device, dtype=dtype, requires_grad=weight_requires_grad)
    labels = torch.randint(0, 6, (2, 4), device=device)
    loss = resolve_op("cross_entropy_loss", "standard", impl).wrapper(hidden, labels, weight)
    if impl != "liger_kernel":
        if expected_argnums is None:
            assert calls == []
        else:
            assert calls and all(call == expected_argnums for call in calls)
    assert loss.requires_grad is (hidden_requires_grad or weight_requires_grad)
    if hidden_requires_grad or weight_requires_grad:
        loss.backward()
    if hidden_requires_grad:
        assert hidden.grad is not None
    else:
        assert hidden.grad is None
    if weight_requires_grad:
        assert weight.grad is not None
    else:
        assert weight.grad is None


@pytest.mark.parametrize(
    "impl",
    (
        pytest.param("eager", id="eager"),
        pytest.param("chunk_loss", id="chunk-loss"),
        pytest.param(
            "liger_kernel",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU"),
            id="liger-kernel",
        ),
    ),
)
def test_ce_rows_skip_grads_under_no_grad(monkeypatch: pytest.MonkeyPatch, impl: str):
    """Trainable leaves under a real no_grad must not build unused grad buffers."""
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")
    calls: list[object] = []
    real = torch.func.grad_and_value

    def wrapped(fn, *args, **kwargs):
        calls.append(kwargs.get("argnums", args[1] if len(args) > 1 else None))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(torch.func, "grad_and_value", wrapped)
    torch.manual_seed(5)
    device = "cuda" if impl == "liger_kernel" else "cpu"
    dtype = torch.bfloat16 if impl == "liger_kernel" else torch.float32
    hidden = torch.randn(2, 4, 8, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(6, 8, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 6, (2, 4), device=device)
    with torch.no_grad():
        loss = resolve_op("cross_entropy_loss", "standard", impl).wrapper(hidden, labels, weight)
    if impl != "liger_kernel":
        assert calls == []
    assert loss.requires_grad is False
    assert torch.isfinite(loss).all()


def test_liger_adapter_detaches_vendor_inputs_when_grad_disabled(monkeypatch: pytest.MonkeyPatch):
    """Adapter contract: vendor sees detached inputs when the caller disables grads."""
    liger_ce = pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import liger_kernel as liger_row

    captured: dict[str, object] = {}

    def spy(_input, weight, target, **kwargs):
        captured["input_requires_grad"] = _input.requires_grad
        captured["weight_requires_grad"] = weight.requires_grad
        captured["grad_weight"] = None
        return _input.new_zeros(()), None, None, None, None, None

    monkeypatch.setattr(liger_ce, "fused_linear_cross_entropy_forward", spy)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    weight = torch.randn(6, 8, requires_grad=True)
    labels = torch.randint(0, 6, (2, 4))
    loss, _saved = liger_row.forward(hidden, labels, weight, grad_enabled=False)
    assert captured["input_requires_grad"] is False
    assert captured["weight_requires_grad"] is False
    assert captured["grad_weight"] is None
    assert loss.requires_grad is False


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU")
def test_liger_skips_vendor_weight_grad_buffer_under_no_grad(monkeypatch: pytest.MonkeyPatch):
    """Detach vendor inputs so Liger does not allocate the ``V×H`` weight buffer."""
    pytest.importorskip("liger_kernel")
    from liger_kernel.ops import fused_linear_cross_entropy as liger_ce

    captured: dict[str, object] = {}
    real = liger_ce.fused_linear_cross_entropy_forward

    def spy(*args, **kwargs):
        hidden = args[0] if args else kwargs["_input"]
        weight = args[1] if len(args) > 1 else kwargs["weight"]
        result = real(*args, **kwargs)
        captured["input_requires_grad"] = hidden.requires_grad
        captured["weight_requires_grad"] = weight.requires_grad
        captured["grad_hidden"] = result[3]
        captured["grad_weight"] = result[4]
        return result

    monkeypatch.setattr(liger_ce, "fused_linear_cross_entropy_forward", spy)
    torch.manual_seed(6)
    hidden = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(6, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 6, (2, 4), device="cuda")
    with torch.no_grad():
        loss = resolve_op("cross_entropy_loss", "standard", "liger_kernel").wrapper(hidden, labels, weight)
    assert captured["input_requires_grad"] is False
    assert captured["weight_requires_grad"] is False
    assert captured["grad_weight"] is None
    assert loss.requires_grad is False
    assert torch.isfinite(loss).all()


def test_eager_fp16_logits_do_not_overflow_outside_cross_entropy():
    logits = torch.ones(128, 1024, dtype=torch.float16)
    labels = torch.zeros(128, dtype=torch.long)

    logits_h = make_grad_leaf(logits)
    out_h = F.cross_entropy(logits_h.float(), labels)
    logits_e = make_grad_leaf(logits)
    out_e = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(
        logits_e, labels, _empty_weight(logits.device)
    )

    torch.testing.assert_close(out_e, out_h)
    out_h.backward()
    out_e.backward()
    torch.testing.assert_close(logits_e.grad, logits_h.grad)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_eager_logits_allow_infinite_mask_values(dtype):
    logits = torch.tensor([[0.0, -torch.inf, 1.0]], dtype=dtype)
    labels = torch.tensor([2])

    logits_h = make_grad_leaf(logits)
    out_h = F.cross_entropy(logits_h.float(), labels)
    logits_e = make_grad_leaf(logits)
    out_e = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(
        logits_e, labels, _empty_weight(logits.device)
    )

    torch.testing.assert_close(out_e, out_h)
    out_h.backward()
    out_e.backward()
    torch.testing.assert_close(logits_e.grad, logits_h.grad)


@pytest.mark.parametrize(
    ("num_tokens", "num_items_in_batch"),
    (
        pytest.param(0, None, id="empty"),
        pytest.param(3, None, id="all-ignored"),
        pytest.param(3, 0, id="all-ignored-zero-items"),
        pytest.param(3, torch.tensor(0), id="all-ignored-zero-items-tensor"),
    ),
)
def test_eager_empty_or_all_ignored_returns_connected_zero(num_tokens: int, num_items_in_batch: int | Tensor | None):
    logits = torch.randn(num_tokens, 4, requires_grad=True)
    labels = torch.full((num_tokens,), -100, dtype=torch.long)

    loss = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(
        logits,
        labels,
        _empty_weight(logits.device),
        num_items_in_batch=num_items_in_batch,
    )

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    ("impl", "device"),
    (
        pytest.param("chunk_loss", "cpu", id="chunk-loss"),
        pytest.param(
            "liger_kernel",
            "cuda",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU"),
            id="liger-kernel",
        ),
    ),
)
@pytest.mark.parametrize("tensor_count", (False, True), ids=("integer-count", "tensor-count"))
def test_fused_all_ignored_zero_items_returns_connected_zero(impl: str, device: str, tensor_count: bool):
    """Fused CE rows preserve the eager contract for an explicit zero count."""
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")
    hidden = torch.randn(3, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.full((3,), -100, device=device, dtype=torch.long)
    num_items_in_batch = torch.tensor(0, device=device) if tensor_count else 0

    loss = resolve_op("cross_entropy_loss", "standard", impl).wrapper(
        hidden,
        labels,
        weight,
        num_items_in_batch=num_items_in_batch,
    )

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def test_eager_matches_hf_hidden_weight():
    torch.manual_seed(1)
    hidden = torch.randn(4, 8, 32, dtype=torch.float32)
    weight = torch.randn(16, 32, dtype=torch.float32)
    labels = torch.randint(0, 16, (4, 8))
    labels[:, 0] = -100

    hidden_h, weight_h = make_grad_leaf(hidden), make_grad_leaf(weight)
    out_h = fixed_cross_entropy(F.linear(hidden_h.reshape(-1, 32), weight_h), labels.reshape(-1))

    hidden_e, weight_e = make_grad_leaf(hidden), make_grad_leaf(weight)
    out_e = resolve_op("cross_entropy_loss", "standard", "eager").wrapper(hidden_e, labels, weight_e)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    assert torch.allclose(hidden_e.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(weight_e.grad, weight_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_chunk_loss_matches_eager_with_uneven_valid_tokens():
    eager = resolve_op("cross_entropy_loss", "standard", "eager").wrapper
    other = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper
    torch.manual_seed(0)
    hidden = torch.randn(2, 20, 16, dtype=torch.float32)
    weight = torch.randn(8, 16, dtype=torch.float32)
    labels = torch.randint(0, 8, (2, 20))
    labels.reshape(-1)[[0, 1, 2, 7, 8, 20, 27, 28, 29, 30]] = -100

    hidden_e, weight_e = make_grad_leaf(hidden), make_grad_leaf(weight)
    hidden_o, weight_o = make_grad_leaf(hidden), make_grad_leaf(weight)
    out_e = eager(hidden_e, labels, weight_e)
    counter = _FullWeightAddCounter(weight.shape)
    with counter:
        out_o = other(hidden_o, labels, weight_o, chunk_size=7)
    assert counter.out_of_place_adds == 0
    assert torch.allclose(out_e, out_o, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_e.backward()
    out_o.backward()
    assert torch.allclose(hidden_e.grad, hidden_o.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(weight_e.grad, weight_o.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_chunk_loss_does_not_copy_full_noncontiguous_hidden():
    from veomni.ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    hidden_shape = (2, 7, 4)

    class FullHiddenCloneCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.full_hidden_clones = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            output = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.clone.default and isinstance(output, Tensor) and output.shape == hidden_shape:
                self.full_hidden_clones += 1
            return output

    torch.manual_seed(9)
    hidden = torch.randn(7, 2, hidden_shape[-1]).transpose(0, 1)
    assert hidden.shape == hidden_shape
    assert not hidden.is_contiguous()
    labels = torch.randint(0, 16, hidden_shape[:-1])
    weight = torch.randn(16, hidden_shape[-1])
    counter = FullHiddenCloneCounter()

    with counter:
        chunk_loss.forward(hidden, labels, weight, chunk_size=2)

    assert counter.full_hidden_clones == 0


@pytest.mark.parametrize(
    ("hidden_shape", "labels_shape"),
    (
        ((2, 3, 4), (2, 0)),
        ((2, 0, 4), (2, 3)),
    ),
)
def test_chunk_loss_rejects_mismatched_empty_token_counts(hidden_shape, labels_shape):
    hidden = torch.randn(hidden_shape)
    labels = torch.zeros(labels_shape, dtype=torch.long)
    weight = torch.randn(8, hidden_shape[-1])

    with pytest.raises(ValueError, match="token count"):
        resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper(hidden, labels, weight)


def test_chunk_loss_empty_returns_connected_zero():
    hidden = torch.empty(2, 0, 4, requires_grad=True)
    labels = torch.empty(2, 0, dtype=torch.long)
    weight = torch.randn(8, 4, requires_grad=True)

    loss = resolve_op("cross_entropy_loss", "standard", "chunk_loss").wrapper(hidden, labels, weight)

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU")
@pytest.mark.parametrize(
    ("seed", "hidden_requires_grad", "weight_requires_grad", "num_items_in_batch", "noncontiguous"),
    (
        pytest.param(0, True, True, None, False, id="all-gradients"),
        pytest.param(5, True, False, None, False, id="frozen-weight"),
        pytest.param(6, False, True, None, False, id="frozen-hidden"),
        pytest.param(3, True, True, 12, False, id="num-items"),
        pytest.param(1, True, True, None, True, id="noncontiguous-hidden"),
    ),
)
def test_liger_matches_eager(
    seed: int,
    hidden_requires_grad: bool,
    weight_requires_grad: bool,
    num_items_in_batch: int | None,
    noncontiguous: bool,
):
    pytest.importorskip("liger_kernel")
    eager = resolve_op("cross_entropy_loss", "standard", "eager").wrapper
    other = resolve_op("cross_entropy_loss", "standard", "liger_kernel").wrapper
    torch.manual_seed(seed)
    batch_size = 4 if noncontiguous else 2
    hidden = torch.randn(batch_size, 16, 32, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        hidden = hidden.transpose(0, 1).contiguous().transpose(0, 1)
        assert not hidden.is_contiguous()
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 64, (batch_size, 16), device="cuda")
    labels[:, 0] = -100

    hidden_e = hidden.detach().requires_grad_(hidden_requires_grad)
    hidden_o = hidden.detach().requires_grad_(hidden_requires_grad)
    weight_e = weight.detach().requires_grad_(weight_requires_grad)
    weight_o = weight.detach().requires_grad_(weight_requires_grad)
    kwargs = {} if num_items_in_batch is None else {"num_items_in_batch": num_items_in_batch}
    out_e = eager(hidden_e, labels, weight_e, **kwargs)
    out_o = other(hidden_o, labels, weight_o, **kwargs)
    assert torch.allclose(out_e.float(), out_o.float(), atol=CE_FUSED_ATOL, rtol=CE_FUSED_RTOL)

    out_e.backward()
    out_o.backward()
    for eager_input, other_input, requires_grad in (
        (hidden_e, hidden_o, hidden_requires_grad),
        (weight_e, weight_o, weight_requires_grad),
    ):
        if requires_grad:
            assert torch.allclose(
                eager_input.grad.float(),
                other_input.grad.float(),
                atol=CE_FUSED_GRAD_ATOL,
                rtol=CE_FUSED_GRAD_RTOL,
            )
        else:
            assert eager_input.grad is None
            assert other_input.grad is None


@pytest.mark.parametrize(
    ("impl", "device"),
    (
        pytest.param("chunk_loss", "cpu", id="chunk-loss"),
        pytest.param(
            "liger_kernel",
            "cuda",
            marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs a GPU"),
            id="liger",
        ),
    ),
)
def test_hidden_state_implementations_require_weight(impl: str, device: str):
    if impl == "liger_kernel":
        pytest.importorskip("liger_kernel")
    with pytest.raises(RuntimeError, match="nonempty ``weight``"):
        resolve_op("cross_entropy_loss", "standard", impl).wrapper(
            torch.randn(4, 8, device=device),
            torch.zeros(4, dtype=torch.long, device=device),
            _empty_weight(device),
        )
