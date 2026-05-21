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
r"""Single-process unit tests for extra_parallel_fsdp2_clip_grad_norm.

Covers the inf-norm reduction path with multiple ``extra_parallel_names``
entries -- the historical bug was a variadic ``torch.maximum`` call that
silently worked only when the combined dict size was 1, and crashed with
``TypeError`` on the Mode-2 shared-LoRA case (``ep`` populated in both
``extra_parallel_total`` and ``extra_parallel_replicated_total``) or any
future multi-axis setup (``ep`` + ``emb`` + ...).

No distributed init is needed: we monkey-patch ``get_parallel_state`` so
every group / mesh is None, ``_fsdp2_reduce_group`` skips its
``dist.all_reduce`` calls, and the reduction runs in pure single-process
Python. The bug-or-no-bug signal is exactly the reduction line, which is
what we want to nail down.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from unittest import mock

import pytest
import torch
import torch.nn as nn


# Direct ``importlib`` lookup: ``veomni.distributed.fsdp2.__init__`` shadows
# the submodule name with ``from .clip_grad_norm import clip_grad_norm`` (the
# function), so ``import veomni.distributed.fsdp2.clip_grad_norm`` would
# resolve to the function, not the module we need to monkey-patch.
cgn_mod = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
extra_parallel_fsdp2_clip_grad_norm = cgn_mod.extra_parallel_fsdp2_clip_grad_norm


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeParallelState:
    """Minimal stub for ``get_parallel_state()`` consumed by clip_grad_norm.

    All groups and meshes are ``None`` so ``_fsdp2_reduce_group`` skips
    every ``dist.all_reduce`` -- the reduction runs single-process and
    only the in-process maths is exercised.
    """

    extra_parallel_names: list[str] = field(default_factory=lambda: ["ep"])
    fsdp_group: object | None = None
    extra_parallel_fsdp_device_mesh: dict[str, object | None] = field(default_factory=dict)

    def extra_parallel_enabled(self, name: str) -> bool:
        return False

    def extra_parallel_group(self, name: str) -> object | None:
        return None


def _make_model(param_groups: dict[str, list[nn.Parameter]], replicated_ids: set) -> nn.Module:
    """Wrap a ``param_groups`` dict in the duck-typed object the clipper expects."""
    m = nn.Module()
    m._extra_parallel_param_groups = param_groups
    m._ep_replicated_lora_param_ids = replicated_ids
    return m


def _param_with_grad(grad: torch.Tensor) -> nn.Parameter:
    p = nn.Parameter(torch.zeros_like(grad), requires_grad=True)
    p.grad = grad.clone()
    return p


# ---------------------------------------------------------------------------
# Inf-norm reduction (the historical bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra_names", [["ep"], ["ep", "emb"]])
def test_inf_norm_with_replicated_bucket_does_not_crash(extra_names):
    """``norm_type=inf`` with both EP and EP-replicated buckets populated.

    Pre-fix: ``torch.maximum(non_ep, *ep.values(), *ep_rep.values())`` is a
    3+ arg call on a 2-arg op -- raises ``TypeError`` at the first crash
    point. Post-fix: iterative ``torch.maximum`` reduces over any number
    of bucket values. The Mode-2 shared-LoRA EP case is exactly the
    smallest failing instance.
    """
    p_non = _param_with_grad(torch.tensor([1.0, -2.0, 0.5]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=list(extra_names),
        extra_parallel_fsdp_device_mesh=dict.fromkeys(extra_names),
    )
    grads_seen = [p_non.grad.abs().max()]

    param_groups: dict[str, list[nn.Parameter]] = {"non_extra_parallel": [p_non]}
    replicated_ids: set = set()
    for i, name in enumerate(extra_names):
        # One EP-bucket param + one EP-replicated-bucket param per axis --
        # for ``extra_names=["ep"]`` this is the Mode-2 shared-LoRA case
        # (1 sharded + 1 replicated == 2 entries that get unpacked alongside
        # ``non_extra_parallel_total``, total 3 args to torch.maximum).
        p_ep = _param_with_grad(torch.tensor([3.0 + i, -1.0, 0.0]))
        p_ep_rep = _param_with_grad(torch.tensor([-(7.0 + i), 0.1, 4.0]))
        param_groups[name] = [p_ep, p_ep_rep]
        replicated_ids.add(id(p_ep_rep))
        grads_seen.append(p_ep.grad.abs().max())
        grads_seen.append(p_ep_rep.grad.abs().max())

    model = _make_model(param_groups, replicated_ids)
    expected = torch.stack(grads_seen).max().to(torch.float32)
    with mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    # The expected inf-norm is the elementwise max across every grad.
    # Single-process + max_norm=1e9 means no clipping is actually applied,
    # so this just validates the reduction maths.
    assert total.item() == pytest.approx(expected.item(), abs=1e-6), (
        f"inf-norm mismatch with {extra_names=}: got {total.item()}, expected {expected.item()}"
    )


def test_inf_norm_single_axis_no_replicated_still_works():
    """Sanity: the legacy K1+K2==1 happy path still passes after the fix.

    Pre-fix this was the only working combination. Keep it asserted so a
    future refactor that breaks the 1-arg reduction case fails fast.
    """
    p_non = _param_with_grad(torch.tensor([0.25]))
    p_ep = _param_with_grad(torch.tensor([5.0, -6.0]))
    fake_ps = _FakeParallelState(
        extra_parallel_names=["ep"],
        extra_parallel_fsdp_device_mesh={"ep": None},
    )
    model = _make_model({"non_extra_parallel": [p_non], "ep": [p_ep]}, replicated_ids=set())

    with mock.patch.object(cgn_mod, "get_parallel_state", return_value=fake_ps):
        total = extra_parallel_fsdp2_clip_grad_norm(
            model, max_norm=1e9, norm_type=math.inf, error_if_nonfinite=False, foreach=False
        )

    assert total.item() == pytest.approx(6.0, abs=1e-6)
