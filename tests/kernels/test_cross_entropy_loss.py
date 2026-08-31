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

from tests.kernels.tol import (
    CE_FUSED_ATOL,
    CE_FUSED_GRAD_ATOL,
    CE_FUSED_GRAD_RTOL,
    CE_FUSED_RTOL,
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE


def _empty_weight(device: torch.device | str) -> Tensor:
    return torch.empty(0, device=device)


def _hf_ce(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch: int | None = None,
) -> Tensor:
    """Match HuggingFace ``fixed_cross_entropy``."""
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss


def _clone(tensor: Tensor) -> Tensor:
    return tensor.detach().requires_grad_(True)


def test_eager_matches_hf_logits():
    torch.manual_seed(0)
    logits = torch.randn(8, 16, dtype=torch.float32)
    labels = torch.randint(0, 16, (8,))
    labels[0] = -100

    logits_h = _clone(logits)
    out_h = _hf_ce(logits_h, labels)

    logits_e = _clone(logits)
    out_e = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper(
        logits_e, labels, _empty_weight(logits.device)
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    assert torch.allclose(logits_e.grad, logits_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_hf_hidden_weight():
    torch.manual_seed(1)
    hidden = torch.randn(4, 8, 32, dtype=torch.float32)
    weight = torch.randn(16, 32, dtype=torch.float32)
    labels = torch.randint(0, 16, (4, 8))
    labels[:, 0] = -100

    hidden_h, weight_h = _clone(hidden), _clone(weight)
    out_h = _hf_ce(F.linear(hidden_h.reshape(-1, 32), weight_h), labels.reshape(-1))

    hidden_e, weight_e = _clone(hidden), _clone(weight)
    out_e = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper(hidden_e, labels, weight_e)
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    assert torch.allclose(hidden_e.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(weight_e.grad, weight_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_matches_hf_num_items():
    torch.manual_seed(2)
    logits = torch.randn(10, 7, dtype=torch.float32)
    labels = torch.randint(0, 7, (10,))
    num_items = 6

    logits_h = _clone(logits)
    out_h = _hf_ce(logits_h, labels, num_items_in_batch=num_items)
    logits_e = _clone(logits)
    out_e = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper(
        logits_e, labels, _empty_weight(logits.device), num_items_in_batch=num_items
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_h.backward()
    out_e.backward()
    assert torch.allclose(logits_e.grad, logits_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_chunk_loss_matches_eager():
    eager = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper
    other = resolve_kernel("cross_entropy_loss", "standard", "chunk_loss").wrapper
    torch.manual_seed(0)
    hidden = torch.randn(2, 20, 16, dtype=torch.float32)
    weight = torch.randn(8, 16, dtype=torch.float32)
    labels = torch.randint(0, 8, (2, 20))
    labels[:, :3] = -100

    hidden_e, weight_e = _clone(hidden), _clone(weight)
    hidden_o, weight_o = _clone(hidden), _clone(weight)
    out_e = eager(hidden_e, labels, weight_e)
    out_o = other(hidden_o, labels, weight_o, chunk_size=7)
    assert torch.allclose(out_e, out_o, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    out_e.backward()
    out_o.backward()
    assert torch.allclose(hidden_e.grad, hidden_o.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(weight_e.grad, weight_o.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_chunk_loss_requires_weight():
    with pytest.raises(RuntimeError, match="nonempty ``weight``"):
        resolve_kernel("cross_entropy_loss", "standard", "chunk_loss").wrapper(
            torch.randn(4, 8), torch.zeros(4, dtype=torch.long), _empty_weight("cpu")
        )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs CUDA")
def test_liger_matches_eager():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper
    other = resolve_kernel("cross_entropy_loss", "standard", "liger_kernel").wrapper
    torch.manual_seed(0)
    hidden = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 64, (2, 16), device="cuda")
    labels[:, 0] = -100

    hidden_e, weight_e = _clone(hidden), _clone(weight)
    hidden_o, weight_o = _clone(hidden), _clone(weight)
    out_e = eager(hidden_e, labels, weight_e)
    out_o = other(hidden_o, labels, weight_o)
    assert torch.allclose(out_e.float(), out_o.float(), atol=CE_FUSED_ATOL, rtol=CE_FUSED_RTOL)

    out_e.backward()
    out_o.backward()
    assert torch.allclose(
        hidden_e.grad.float(), hidden_o.grad.float(), atol=CE_FUSED_GRAD_ATOL, rtol=CE_FUSED_GRAD_RTOL
    )
    assert torch.allclose(
        weight_e.grad.float(), weight_o.grad.float(), atol=CE_FUSED_GRAD_ATOL, rtol=CE_FUSED_GRAD_RTOL
    )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs CUDA")
def test_liger_matches_eager_noncontiguous_hidden():
    pytest.importorskip("liger_kernel")
    eager = resolve_kernel("cross_entropy_loss", "standard", "eager").wrapper
    other = resolve_kernel("cross_entropy_loss", "standard", "liger_kernel").wrapper
    torch.manual_seed(1)
    hidden = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16).transpose(0, 1).contiguous().transpose(0, 1)
    assert not hidden.is_contiguous()
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 64, (4, 16), device="cuda")
    hidden_e, weight_e = _clone(hidden), _clone(weight)
    hidden_o, weight_o = _clone(hidden), _clone(weight)
    out_e = eager(hidden_e, labels, weight_e)
    out_o = other(hidden_o, labels, weight_o)
    assert torch.allclose(out_e.float(), out_o.float(), atol=CE_FUSED_ATOL, rtol=CE_FUSED_RTOL)
    out_e.backward()
    out_o.backward()
    assert torch.allclose(
        hidden_e.grad.float(), hidden_o.grad.float(), atol=CE_FUSED_GRAD_ATOL, rtol=CE_FUSED_GRAD_RTOL
    )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="liger fused CE needs CUDA")
def test_liger_requires_weight():
    pytest.importorskip("liger_kernel")
    with pytest.raises(RuntimeError, match="nonempty ``weight``"):
        resolve_kernel("cross_entropy_loss", "standard", "liger_kernel").wrapper(
            torch.randn(4, 8, device="cuda"),
            torch.zeros(4, dtype=torch.long, device="cuda"),
            _empty_weight("cuda"),
        )
