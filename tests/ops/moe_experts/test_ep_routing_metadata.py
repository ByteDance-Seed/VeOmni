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

"""Portable EP metadata/collective regressions; fused expert kernels are outside this suite."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import from_parameter
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.distributed.moe import preprocess, token_pre_all2all, tokens_post_all2all


def _routing_worker(rank, world_size, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        store=endpoint.connect(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        for case in ("uneven", "empty_experts"):
            torch.manual_seed(10 + rank)
            tokens, experts = 5 + rank, 8
            selected = (
                torch.randint(experts, (tokens, 2)) if case == "uneven" else torch.zeros(tokens, 2, dtype=torch.long)
            )
            selected[0] = 0  # repeated top-k expert counts as one dispatched row
            mask = torch.nn.functional.one_hot(selected, num_classes=experts).permute(2, 1, 0)
            host_handoffs = []
            original_to = torch.Tensor.to

            def track_to(tensor, *args, _handoffs=host_handoffs, _to=original_to, **kwargs):
                device = kwargs.get("device", args[0] if args else None)
                if isinstance(device, (str, torch.device)) and torch.device(device).type == "cpu":
                    _handoffs.append(tensor.shape)
                return _to(tensor, *args, **kwargs)

            # Count explicit host handoffs while retaining real tensor/collective operations. This is not a CUDA
            # timing claim; on CUDA these metadata handoffs must not multiply synchronization points per layer.
            with patch.object(torch.Tensor, "to", track_to):
                inputs, outputs, counts, expert_counts = preprocess(mask, experts, dist.group.WORLD)
            assert len(host_handoffs) <= 1, f"Routing metadata uses {len(host_handoffs)} separate host handoffs"
            assert counts.device.type == expert_counts.device.type == "cpu"
            assert counts.shape == (world_size, experts // world_size)
            assert inputs == mask.any(dim=1).sum(dim=1).reshape(world_size, -1).sum(dim=1).tolist()
            assert outputs == counts.sum(dim=1).tolist()
            torch.testing.assert_close(expert_counts, counts.sum(dim=0), rtol=0, atol=0)

            hidden = torch.randn(tokens, 4, requires_grad=True)
            weights = torch.rand(tokens, 2, requires_grad=True)
            reference_hidden = hidden.detach().clone().requires_grad_()
            reference_weights = weights.detach().clone().requires_grad_()
            permuted, routing_map, mapping, shape = token_pre_all2all(
                hidden, mask, experts, inputs, outputs, counts, dist.group.WORLD
            )
            scales = torch.arange(rank * (experts // world_size) + 1, (rank + 1) * (experts // world_size) + 1)
            scales = scales.repeat_interleave(expert_counts).to(permuted.dtype)
            actual = tokens_post_all2all(
                permuted * scales[:, None],
                weights,
                selected,
                experts,
                inputs,
                outputs,
                counts,
                routing_map,
                mapping,
                shape,
                dist.group.WORLD,
            )
            expected = reference_hidden * (reference_weights * (selected + 1)).sum(dim=1, keepdim=True)
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(hidden.grad, reference_hidden.grad)
            torch.testing.assert_close(weights.grad, reference_weights.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.launch_plan(provider=from_parameter("world_size", device="cpu"))
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_ep_routing_metadata_roundtrip(world_size, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    spawn_with_rendezvous(_routing_worker, args=(world_size,), nprocs=world_size)
