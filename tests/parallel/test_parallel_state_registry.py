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

"""``init_parallel_state`` registry behaviour, including ``name=None``.

Single-rank gloo, so this runs anywhere. The registry, the topology cache and
the ambient state are process globals, hence the teardown.
"""

import pytest
import torch.distributed as dist

from veomni.distributed.parallel_state import (
    clear_parallel_state,
    get_parallel_state,
    get_parallel_state_by_name,
    init_parallel_state,
)


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
    state = init_parallel_state(dp_size=1, device_type="cpu", name=None)

    with pytest.raises(ValueError, match="is not registered"):
        get_parallel_state_by_name("base")
    # Opting out of the registry must not opt out of becoming the ambient state,
    # which every SP/DP group getter resolves from.
    assert get_parallel_state() is state


def test_named_init_still_registers(single_rank_group):
    state = init_parallel_state(dp_size=1, device_type="cpu", name="vision_tower")

    assert get_parallel_state_by_name("vision_tower") is state


def test_a_later_named_init_adopts_the_cached_anonymous_state(single_rank_group):
    anonymous = init_parallel_state(dp_size=1, device_type="cpu", name=None)

    # Same topology, so the cache hits: name=None declines a key for itself, it
    # does not keep the state out of the registry forever.
    named = init_parallel_state(dp_size=1, device_type="cpu", name="base")

    assert named is anonymous
    assert get_parallel_state_by_name("base") is anonymous
