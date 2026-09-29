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

"""``EmbParallelMixin`` with the ``emb`` extra-parallel group absent.

Every module that mixes this in runs this path in single-replica training and
in eager inference, so lookup and projection must stay exactly the dense ops.
The sharded paths need a process group and live with the distributed tests; the
kernels they call are covered in ``tests/ops/test_vocab_parallel_embed.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from veomni.models.seed_omni.mixins import emb_parallel_mixin
from veomni.models.seed_omni.mixins.emb_parallel_mixin import EmbParallelMixin


class _NoExtraParallelState:
    """Parallel state as it looks before any ``emb`` group is requested."""

    extra_parallel_sizes = ()

    def extra_parallel_enabled(self, name: str) -> bool:
        return False


def test_emb_parallel_is_inactive_without_the_group(monkeypatch):
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", _NoExtraParallelState)
    assert EmbParallelMixin.emb_parallel_active() is False


def test_a_plain_tensor_weight_passes_through_untouched(monkeypatch):
    """Single replica / eager: nothing to gather, and no copy of a possibly huge table."""
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", _NoExtraParallelState)
    weight = torch.randn(6, 4)
    assert EmbParallelMixin.emb_local_weight(weight) is weight


def test_lookup_is_the_module_call_when_emb_is_off(monkeypatch):
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", _NoExtraParallelState)
    embedding = nn.Embedding(6, 4)
    ids = torch.tensor([[0, 3], [5, 1]])

    assert torch.equal(EmbParallelMixin.emb_parallel_lookup(embedding, ids), embedding(ids))


def test_projection_is_a_dense_linear_when_emb_is_off(monkeypatch):
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", _NoExtraParallelState)
    weight = torch.randn(6, 4)
    hidden = torch.randn(2, 4)

    assert torch.equal(EmbParallelMixin.emb_parallel_project(hidden, weight), F.linear(hidden, weight))


def test_projection_casts_the_weight_to_the_activation_dtype(monkeypatch):
    """``full_tensor()`` hands back the fp32 master param; the matmul needs the activation dtype."""
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", _NoExtraParallelState)
    weight = torch.randn(6, 4, dtype=torch.float32)
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)

    assert EmbParallelMixin.emb_parallel_project(hidden, weight).dtype is torch.bfloat16
