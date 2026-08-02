# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""``ParallelPlan.fsdp_ignored_param_fqn_patterns`` -- the typed declaration of
parameters that must stay OUT of the root FSDP2 shard (e.g. a frozen FP32
sub-encoder). No FSDP2 wrapping happens here: only the pattern-to-param
resolution and the nested-sharded assertion in ``parallelize_model_fsdp2``. The
full FSDP path is exercised by the GPU-only
``tests/utils/test_extra_parallel_clip_grad_norm.py`` suite."""

from __future__ import annotations

import pytest
import torch.nn as nn
from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.torch_parallelize import _assert_ignored_params_not_in_sharded_submodules


class _FakeFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(4, 4)
        self.norm = nn.Linear(4, 4)


class _ToyModel(nn.Module):
    """Rough shape of an image model with a frozen encoder sibling."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        self.vae = _FakeFrozenEncoder()


def test_ignored_patterns_resolve_to_matching_params():
    """A ``vae.*`` pattern picks up every VAE param, no trainable-body param."""
    model = _ToyModel()
    plan = ParallelPlan(
        extra_parallel_plan={"ep": {"model.0.weight": Shard(0)}},
        fsdp_ignored_param_fqn_patterns=["vae.*"],
    )
    ignored = plan.get_fsdp_ignored_params(model)
    assert ignored is not None
    ignored_ids = {id(p) for p in ignored}
    for fqn, param in model.named_parameters():
        if fqn.startswith("vae."):
            assert id(param) in ignored_ids, f"expected {fqn!r} to be ignored"
        else:
            assert id(param) not in ignored_ids, f"unexpected non-vae {fqn!r} ignored"


def test_no_patterns_returns_none():
    """Empty declaration keeps the caller's default (no ``ignored_params`` kwarg)."""
    model = _ToyModel()
    plan = ParallelPlan(extra_parallel_plan={"ep": {"model.0.weight": Shard(0)}})
    assert plan.get_fsdp_ignored_params(model) is None


def test_patterns_matching_nothing_warns_and_returns_none():
    """Declared-but-unused patterns log a warning and don't wrongly return ``set()``."""
    model = _ToyModel()
    plan = ParallelPlan(
        extra_parallel_plan={"ep": {"model.0.weight": Shard(0)}},
        fsdp_ignored_param_fqn_patterns=["nonexistent.*"],
    )
    ignored = plan.get_fsdp_ignored_params(model)
    # ``None`` signals "no ignored set" to the FSDP call site so it stays default.
    assert ignored is None or ignored == set()


def _mark_fsdp_wrapped(module):
    """Reproduce the marker a nested ``fully_shard`` leaves behind.

    ``fully_shard`` itself needs a process group; the guard only inspects how
    FSDP2 tags a wrapped module -- ``__class__`` rewritten to a subclass with
    ``FSDPModule`` ahead of the original in the MRO.
    """
    from torch.distributed.fsdp import FSDPModule

    cls = type(module)
    module.__class__ = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})


def test_nested_sharded_assertion_flags_param_inside_wrapped_submodule():
    """If an ignored param ends up inside an already-``fully_shard``-wrapped submodule,
    the guard must raise -- ``ignored_params`` on the root call would be silently
    ignored for it, still sharding + MP-casting the param."""
    model = _ToyModel()
    _mark_fsdp_wrapped(model.vae)

    ignored = {p for _, p in model.vae.named_parameters()}
    with pytest.raises(AssertionError, match="lives inside submodule"):
        _assert_ignored_params_not_in_sharded_submodules(model, ignored)


def test_nested_sharded_assertion_is_noop_for_disjoint_ignored_set():
    """A wrapped submodule holding none of the ignored params must not trigger."""
    model = _ToyModel()
    _mark_fsdp_wrapped(model.model)

    _assert_ignored_params_not_in_sharded_submodules(model, {model.vae.conv.weight})


def test_empty_ignored_set_is_short_circuit():
    """Empty / falsy ``ignored`` must not scan the module tree at all."""
    model = _ToyModel()
    _assert_ignored_params_not_in_sharded_submodules(model, set())
    _assert_ignored_params_not_in_sharded_submodules(model, None)

