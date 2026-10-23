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

"""``_init_parallel_state`` registry and CPU/Gloo mesh behavior."""

from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    get_parallel_state,
    get_parallel_state_by_name,
    set_parallel_state,
    use_parallel_state,
)
from veomni.distributed.sequence_parallel import comm as sp_comm
from veomni.distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group


_CI_CPU_4 = LaunchSpec(nproc=4, device="cpu")


@pytest.fixture
def single_rank_group(tmp_path):
    # A file store under tmp_path rather than a fixed port: the rendezvous is
    # then unique per test, so a concurrent run of this suite cannot collide with
    # it. Torch creates the file and never removes it, hence a fresh directory.
    dist.init_process_group(backend="gloo", init_method=f"file://{tmp_path / 'rendezvous'}", world_size=1, rank=0)
    try:
        yield
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


def test_name_none_claims_no_registry_key(single_rank_group):
    state = _init_parallel_state(dp_size=1, device_type="cpu", name=None)

    with pytest.raises(ValueError, match="is not registered"):
        get_parallel_state_by_name("base")
    # Opting out of the registry must not opt out of becoming the ambient state,
    # which every SP/DP group getter resolves from.
    assert get_parallel_state() is state


def test_named_init_still_registers(single_rank_group):
    state = _init_parallel_state(dp_size=1, device_type="cpu", name="vision_tower")

    assert get_parallel_state_by_name("vision_tower") is state


def test_a_later_named_init_adopts_the_cached_anonymous_state(single_rank_group):
    anonymous = _init_parallel_state(dp_size=1, device_type="cpu", name=None)

    # Same topology, so the cache hits: name=None declines a key for itself, it
    # does not keep the state out of the registry forever.
    named = _init_parallel_state(dp_size=1, device_type="cpu", name="base")

    assert named is anonymous
    assert get_parallel_state_by_name("base") is anonymous


def _check_mesh_rank_layout(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        state = _init_parallel_state(dp_size=1, cp_size=2, ulysses_size=2, device_type="cpu", name=None)
        mesh = state.device_mesh
        assert state.sp_size == 4
        assert state.sp_rank == rank
        assert state.cp_size == 2
        assert state.cp_rank == rank // 2
        assert state.ulysses_size == 2
        assert state.ulysses_rank == rank % 2
        assert mesh.mesh_dim_names == ("dp_shard", "cp", "ulysses")
        assert mesh.mesh.tolist() == [[[0, 1], [2, 3]]]
        assert dist.get_process_group_ranks(mesh.get_group("ulysses")) == [rank // 2 * 2, rank // 2 * 2 + 1]
        assert dist.get_process_group_ranks(mesh.get_group("cp")) == [rank % 2, rank % 2 + 2]
        for name in ("sp", "dp_sp", "dp_shard_sp"):
            assert dist.get_process_group_ranks(mesh.get_group(name)) == [0, 1, 2, 3]

        sp_state = _init_parallel_state(dp_size=1, ulysses_size=4, device_type="cpu", name="sp")
        dp_state = _init_parallel_state(dp_size=4, ulysses_size=1, device_type="cpu", name="dp")
        assert sp_state.ulysses_group is not None
        assert dp_state.ulysses_group is None
        with use_parallel_state("sp"):
            assert get_ulysses_sequence_parallel_group() is sp_state.ulysses_group
        with use_parallel_state("dp"):
            assert get_ulysses_sequence_parallel_group() is None
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_CPU_4)
@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the CPU Gloo backend")
def test_mesh_places_ulysses_inside_cp_with_combined_parallelism(tmp_path):
    mp.spawn(_check_mesh_rank_layout, args=(str(tmp_path / "rendezvous"),), nprocs=_CI_CPU_4.nproc, join=True)


def test_ulysses_group_resolves_only_from_current_parallel_state(single_rank_group):
    assert not hasattr(sp_comm, "_ULYSSES_SP_GROUP_OVERRIDE")
    assert not hasattr(sp_comm, "set_ulysses_sequence_parallel_group")

    state = _init_parallel_state(dp_size=1, ulysses_size=1, device_type="cpu", name=None)
    assert get_ulysses_sequence_parallel_group() is state.ulysses_group

    set_parallel_state(None)
    assert get_ulysses_sequence_parallel_group() is None
