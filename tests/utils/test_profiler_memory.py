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

import pickle

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from veomni.utils.helper import create_profiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory history requires a GPU")
def test_profiler_memory_snapshot_keeps_live_allocation_stacks(tmp_path):
    profiler = create_profiler(
        start_step=1,
        end_step=2,
        trace_dir=str(tmp_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=False,
        global_rank=0,
    )

    profiler.start()
    try:
        values = torch.randn(128, 64, device="cuda", requires_grad=True)
        indices = torch.arange(128, device="cuda") % 16
        forward_calls = 0

        def recomputed_forward(inputs):
            nonlocal forward_calls
            forward_calls += 1
            output = torch.zeros_like(inputs)
            output.index_add_(0, indices, inputs)
            return output.square()

        checkpoint(recomputed_forward, values, use_reentrant=False).mean().backward()
        assert forward_calls > 1
        torch.cuda.synchronize()
        profiler.step()
    finally:
        profiler.stop()

    with next(tmp_path.glob("*.pkl")).open("rb") as file:
        snapshot = pickle.load(file)

    live_blocks = [
        block
        for segment in snapshot["segments"]
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]
    assert any(block["frames"] for block in live_blocks)

    # Free-event traceback capture can fail during checkpoint recomputation
    # and is unnecessary for locating the tensors still occupying memory.
    events = snapshot["device_traces"][torch.cuda.current_device()]
    assert events
    assert all(not event.get("frames") for event in events)
    assert list(tmp_path.glob("*.pt.trace.json.gz"))
