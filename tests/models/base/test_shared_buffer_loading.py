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

"""HF materialization must preserve shared-buffer values and registrations."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, distribute_tensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import _materialize_and_load_weights, _move_model_buffers_to_device
from veomni.models.checkpoint import weights
from veomni.models.checkpoint.convert import ConvertedCheckpointTensor


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_PERSISTENCE = [(True, False), (False, True), (False, False), (True, True)]


class _SharedBuffers(nn.Module):
    def __init__(self, root_persistent, child_persistent, layout="direct"):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.layout = layout
        self.weight = nn.Parameter(torch.zeros(2))
        self.child = nn.Module()
        shared = torch.tensor([7.0, 11.0])
        self.register_buffer("root_buffer", shared, persistent=root_persistent)
        self.child.register_buffer("child_buffer", shared, persistent=child_persistent)
        self.register_buffer("optional", None)

    def get_parallel_plan(self):
        # Buffers are read whole even by the EP streaming loader.
        return ParallelPlan(extra_parallel_plan={"ep": {}})

    @staticmethod
    def _create_checkpoint_tensor_converter(model):
        return None if model.layout == "direct" else _BufferAliasConverter(model.layout)


class _BufferAliasConverter:
    def __init__(self, layout):
        self.layout = layout
        self.pending = []

    def can_handle(self, name):
        return name.startswith("legacy.")

    def convert(self, name, tensor):
        converted = ConvertedCheckpointTensor(name=name.removeprefix("legacy."), tensor=tensor)
        if self.layout == "finalize":
            self.pending.append(converted)
            return None
        return converted

    def finalize(self):
        pending, self.pending = self.pending, []
        return pending

    def fused_expert_target(self, name):
        return None

    def for_expert_range(self, start, num_local):
        return _BufferAliasConverter(self.layout)


class _RandomInitSharedBuffers(_SharedBuffers):
    def __init__(self, root_persistent, child_persistent, overwrite_buffer=False):
        super().__init__(root_persistent, child_persistent)
        self.register_buffer("standalone", torch.tensor([5.0, 8.0]))
        self.overwrite_buffer = overwrite_buffer
        self.init_calls = 0

    def init_weights(self):
        self.init_calls += 1
        with torch.no_grad():
            self.weight.copy_(torch.tensor([1.0, 2.0]))
            if self.overwrite_buffer:
                self.child.child_buffer.copy_(torch.tensor([31.0, 53.0]))


@pytest.mark.parametrize("root_persistent,child_persistent", _PERSISTENCE)
@pytest.mark.parametrize("overwrite_buffer", [False, True])
def test_random_init_preserves_real_buffers(root_persistent, child_persistent, overwrite_buffer):
    with weights.init_empty_weights():
        model = _RandomInitSharedBuffers(root_persistent, child_persistent, overwrite_buffer)
    _materialize_and_load_weights(
        model,
        None,
        "cpu",
        should_skip_hf_weight_load=False,
        is_peft_model=False,
        adapter_path=None,
        broadcast_from_rank0=False,
    )
    expected = torch.tensor([31.0, 53.0] if overwrite_buffer else [7.0, 11.0])
    assert model.init_calls == 1
    assert model.root_buffer is model.child.child_buffer
    assert torch.equal(model.root_buffer, expected)
    assert torch.equal(model.standalone, torch.tensor([5.0, 8.0]))
    assert torch.equal(model.weight, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("root_persistent,child_persistent", _PERSISTENCE)
def test_resume_keeps_checkpoint_authority(monkeypatch, root_persistent, child_persistent):
    with weights.init_empty_weights():
        model = _RandomInitSharedBuffers(root_persistent, child_persistent)
    to_empty = model.to_empty

    def sentinel_to_empty(*, device):
        result = to_empty(device=device)
        for buffer in model.buffers():
            buffer.fill_(-1234)
        return result

    monkeypatch.setattr(model, "to_empty", sentinel_to_empty)
    _materialize_and_load_weights(
        model,
        "must-not-read-hf",
        "cpu",
        should_skip_hf_weight_load=True,
        is_peft_model=False,
        adapter_path=None,
        broadcast_from_rank0=False,
    )
    expected = torch.full((2,), -1234.0) if root_persistent or child_persistent else torch.tensor([7.0, 11.0])
    assert model.init_calls == 0
    assert model.root_buffer is model.child.child_buffer
    assert torch.equal(model.root_buffer, expected)
    assert torch.equal(model.standalone, torch.full((2,), -1234.0))
    checkpoint = {"weight": torch.tensor([1.0, 2.0]), "standalone": torch.tensor([42.0, 43.0])}
    if root_persistent:
        checkpoint["root_buffer"] = torch.tensor([21.0, 34.0])
    if child_persistent:
        checkpoint["child.child_buffer"] = torch.tensor([21.0, 34.0])
    model.load_state_dict(checkpoint, strict=True)
    expected = torch.tensor([21.0, 34.0] if root_persistent or child_persistent else [7.0, 11.0])
    assert model.root_buffer is model.child.child_buffer
    assert torch.equal(model.root_buffer, expected)
    assert torch.equal(model.standalone, torch.tensor([42.0, 43.0]))


def _write_checkpoint(directory, root_persistent, child_persistent, layout="direct"):
    directory.mkdir(exist_ok=True)
    payload = {"weight": torch.tensor([1.0, 2.0])}
    for name, persistent in (("root_buffer", root_persistent), ("child.child_buffer", child_persistent)):
        if persistent:
            prefix = "" if layout == "direct" else "legacy."
            payload[prefix + name] = torch.tensor([21.0, 34.0])
    save_file(payload, str(directory / "model.safetensors"))


def _assert_shared_values(model, root_persistent, child_persistent):
    root, child = model.root_buffer, model.child.child_buffer
    expected = torch.tensor([21.0, 34.0] if root_persistent or child_persistent else [7.0, 11.0])
    local = root.to_local() if hasattr(root, "to_local") else root
    assert torch.equal(local, expected)
    assert root.dtype == expected.dtype
    assert root is child
    assert model.optional is None
    expected_keys = {"weight"}
    if root_persistent:
        expected_keys.add("root_buffer")
    if child_persistent:
        expected_keys.add("child.child_buffer")
    assert set(model.state_dict()) == expected_keys
    assert torch.equal(model.weight, torch.tensor([1.0, 2.0]))
    _move_model_buffers_to_device(model, torch.device("cpu"))
    assert model.root_buffer is model.child.child_buffer
    model.root_buffer.fill_(123)
    child = model.child.child_buffer
    local = child.to_local() if hasattr(child, "to_local") else child
    assert torch.equal(local, torch.full((2,), 123.0))


@pytest.fixture
def cpu_group(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    clear_parallel_state()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    _init_parallel_state(device_type="cpu")
    monkeypatch.setattr(weights, "_get_communication_device", lambda _: torch.device("cpu"))
    try:
        yield
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("root_persistent,child_persistent", _PERSISTENCE)
@pytest.mark.parametrize("loader", ["plain", "rank0", "ep_sharded"])
@pytest.mark.parametrize("layout", ["direct", "rename", "finalize"])
def test_hf_shared_buffer_values_and_identity(tmp_path, cpu_group, root_persistent, child_persistent, loader, layout):
    _write_checkpoint(tmp_path, root_persistent, child_persistent, layout)
    with weights.init_empty_weights():
        model = _SharedBuffers(root_persistent, child_persistent, layout)
    loaders = {
        "plain": weights.load_model_weights,
        "rank0": weights.rank0_load_and_broadcast_weights,
        "ep_sharded": weights.load_model_weights_ep_sharded,
    }
    loaders[loader](model, str(tmp_path), init_device="cpu")
    _assert_shared_values(model, root_persistent, child_persistent)


@pytest.mark.parametrize("root_persistent,child_persistent", _PERSISTENCE)
def test_hf_shared_dtensor_buffers_rebind_after_dispatch(tmp_path, cpu_group, root_persistent, child_persistent):
    _write_checkpoint(tmp_path, root_persistent, child_persistent)
    with weights.init_empty_weights():
        model = _SharedBuffers(root_persistent, child_persistent)
    shared = distribute_tensor(model.root_buffer, init_device_mesh("cpu", (1,)), [Replicate()])
    model.root_buffer = model.child.child_buffer = shared
    weights.load_model_weights(model, str(tmp_path), init_device="cpu", dtensor_factory=distribute_tensor)
    _assert_shared_values(model, root_persistent, child_persistent)


@pytest.mark.parametrize("first_persistent,second_persistent", _PERSISTENCE)
@pytest.mark.parametrize("reverse", [False, True])
def test_mapping_load_preserves_cross_child_buffers(tmp_path, first_persistent, second_persistent, reverse):
    parent = nn.Module()
    shared = torch.tensor([7.0, 11.0])
    parent.register_buffer("root_alias", shared, persistent=False)
    paths = {}
    for name, persistent in (("first", first_persistent), ("second", second_persistent)):
        with weights.init_empty_weights():
            child = nn.Module()
            child.weight = nn.Parameter(torch.zeros(2))
        child.config = SimpleNamespace(tie_word_embeddings=False)
        child.register_buffer("buffer", shared, persistent=persistent)
        parent.add_module(name, child)
        directory = tmp_path / name
        directory.mkdir()
        payload = {"weight": torch.tensor([1.0, 2.0])}
        if persistent:
            payload["buffer"] = torch.tensor([21.0, 34.0])
        save_file(payload, str(directory / "model.safetensors"))
        paths[name] = str(directory)
    if reverse:
        paths = dict(reversed(paths.items()))
    _materialize_and_load_weights(
        parent,
        paths,
        "cpu",
        should_skip_hf_weight_load=False,
        is_peft_model=False,
        adapter_path=None,
        broadcast_from_rank0=False,
    )
    expected = torch.tensor([21.0, 34.0] if first_persistent or second_persistent else [7.0, 11.0])
    assert torch.equal(parent.root_alias, expected)
    assert parent.root_alias is parent.first.buffer is parent.second.buffer
    expected_keys = {"first.weight", "second.weight"}
    if first_persistent:
        expected_keys.add("first.buffer")
    if second_persistent:
        expected_keys.add("second.buffer")
    assert set(parent.state_dict()) == expected_keys
    for name in ("first", "second"):
        assert torch.equal(getattr(parent, name).weight, torch.tensor([1.0, 2.0]))
    parent.first.buffer.fill_(123)
    assert torch.equal(parent.second.buffer, torch.full((2,), 123.0))


def _rank0_alias_worker(rank, directory, endpoint):
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2)
    _init_parallel_state(dp_size=2, device_type="cpu")
    # This is a CPU/Gloo test on both CPU and accelerator CI runners.
    weights._get_communication_device = lambda _: torch.device("cpu")
    try:
        for layout in ("direct", "rename", "finalize"):
            for index, (root_persistent, child_persistent) in enumerate(_PERSISTENCE):
                with weights.init_empty_weights():
                    model = _SharedBuffers(root_persistent, child_persistent, layout)
                weights.rank0_load_and_broadcast_weights(
                    model, str(directory / f"{layout}-{index}"), init_device="cpu"
                )
                _assert_shared_values(model, root_persistent, child_persistent)
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_rank0_shared_buffers_on_receiving_rank(tmp_path):
    for layout in ("direct", "rename", "finalize"):
        for index, persistence in enumerate(_PERSISTENCE):
            _write_checkpoint(tmp_path / f"{layout}-{index}", *persistence, layout)
    spawn_with_rendezvous(_rank0_alias_worker, (tmp_path,), nprocs=2)
