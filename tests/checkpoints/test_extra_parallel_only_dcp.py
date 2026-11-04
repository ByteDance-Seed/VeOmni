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

"""DCP round trip of a parameter that ExtraParallel shards but FSDP does not wrap.

Under ``fsdp_config.extra_parallel_only`` (DeepSeek-V4.1's row-sharded embedding
and n-gram tables) every rank holds a plain local slice and the ``*_fsdp`` mesh
axis has size 1. If the slice is described on that axis, DCP sees a whole
tensor the size of one slice, saves a single copy, and every rank resumes with
rank 0's rows.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = 4
ROWS, COLS = 3, 5


def _worker(rank: int, init_file: str, ckpt_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE)
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.device_mesh import init_device_mesh

        from veomni.checkpoint.dcp_checkpointer import drop_extra_parallel_dim, restore_extra_parallel_dim

        mesh = init_device_mesh("cpu", (1, WORLD_SIZE), mesh_dim_names=("emb_fsdp", "emb"))
        fsdp_submesh = mesh["emb_fsdp"]
        local = torch.arange(ROWS * COLS, dtype=torch.float32).view(ROWS, COLS) + 100 * rank

        saved = restore_extra_parallel_dim(local.clone(), mesh, fsdp_submesh, ep_shard_dim=0)
        assert saved.shape == (WORLD_SIZE * ROWS, COLS)
        dcp.save({"table": saved}, checkpoint_id=ckpt_dir)

        target = restore_extra_parallel_dim(torch.zeros_like(local), mesh, fsdp_submesh, ep_shard_dim=0)
        state = {"table": target}
        dcp.load(state, checkpoint_id=ckpt_dir)
        torch.testing.assert_close(drop_extra_parallel_dim(state["table"], fsdp_submesh), local)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo required")
def test_unwrapped_extra_parallel_slices_survive_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, "ckpt")
        mp.spawn(_worker, args=(os.path.join(tmp, "init"), ckpt_dir), nprocs=WORLD_SIZE, join=True)
