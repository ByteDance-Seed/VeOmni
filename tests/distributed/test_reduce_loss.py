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
# See the License for the specific language governing limitations
# under the License.

"""Sequence-parallel ``ReduceLoss`` graph contract."""

from unittest.mock import MagicMock, patch

import torch

from veomni.distributed.sequence_parallel.loss import ReduceLoss


def test_reduce_loss_no_nan_when_sp_group_all_padding():
    """The distributed reducer must return graph-connected zero for 0/0 tokens."""
    with (
        patch(
            "veomni.distributed.sequence_parallel.loss.get_unified_sequence_parallel_group", return_value=MagicMock()
        ),
        patch("veomni.distributed.sequence_parallel.loss.dist.get_world_size", return_value=2),
        patch("veomni.distributed.sequence_parallel.loss.dist.all_reduce", side_effect=lambda *_args, **_kwargs: None),
    ):
        value = torch.tensor(0.5, requires_grad=True)
        result = ReduceLoss.apply(value * 1.0, torch.tensor(0.0))
        assert torch.isfinite(result)
        assert result.item() == 0.0
        result.backward()
        assert value.grad is not None
        assert torch.isfinite(value.grad)
        assert value.grad.item() == 0.0
