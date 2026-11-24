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

"""Which modules ``ParallelPlan`` hands the parallelizer to wrap on each para mesh,
and how FSDP2 manual prefetching wires those wrapped stacks."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous, torchrun
from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import _configure_fsdp2_manual_prefetch, parallelize_model_fsdp2
from veomni.distributed.utils import sort_fqn_by_submodule_first
from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan as qwen3_5_moe_plan
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CI_ACCELERATOR_4 = LaunchSpec(nproc=4, device="accelerator")


def test_every_planned_parameter_owner_is_a_para_module():
    """An owner left out would be re-sharded over the regular FSDP mesh, mixing ranks' slices."""
    assert qwen3_5_moe_plan().extra_parallel_fsdp_no_shard_module == {
        "ep": {"model.language_model.layers.*.mlp.experts", "mtp.layers.*.mlp.experts"}
    }


def test_owners_of_one_module_collapse_to_one_entry():
    plan = ParallelPlan(
        extra_parallel_plan={"ep": {"layers.*.experts.gate_up_proj": Shard(0), "layers.*.experts.down_proj": Shard(0)}}
    )
    assert plan.extra_parallel_fsdp_no_shard_module == {"ep": {"layers.*.experts"}}


def test_every_owner_is_found_in_the_model():
    model = nn.Module()
    model.embed_tokens = nn.Embedding(8, 4)
    model.decoder = nn.Module()
    model.decoder.ngram = nn.Embedding(8, 4)
    model.decoder.ngram_extra = nn.Embedding(8, 4)
    plan = ParallelPlan(
        extra_parallel_plan={"emb": {"embed_tokens.weight": Shard(0), "decoder.ngram.weight": Shard(0)}}
    )

    found = plan.get_extra_parallel_fsdp_no_shard_info(model, "emb")
    assert found == {"embed_tokens": model.embed_tokens, "decoder.ngram": model.decoder.ngram}


def test_nested_owners_are_rejected():
    with pytest.raises(ValueError, match="'decoder' and 'decoder.ngram' are nested"):
        ParallelPlan(extra_parallel_plan={"emb": {"decoder.weight": Shard(0), "decoder.ngram.weight": Shard(0)}})


def test_owners_that_only_share_a_name_prefix_are_not_nested():
    plan = ParallelPlan(extra_parallel_plan={"emb": {"ngram.weight": Shard(0), "ngram_extra.weight": Shard(0)}})
    assert plan.extra_parallel_fsdp_no_shard_module == {"emb": {"ngram", "ngram_extra"}}


def test_sort_fqn_by_submodule_first_preserves_siblings_and_orders_descendants():
    layers = [f"model.layers.{i}" for i in range(15)]
    assert sort_fqn_by_submodule_first(layers) == layers
    assert sort_fqn_by_submodule_first(["decoder", "embed_tokens", "decoder.embed_tokens"]) == [
        "embed_tokens",
        "decoder.embed_tokens",
        "decoder",
    ]


class _RecordingFsdpStub(nn.Module):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.forward_prefetch: list[nn.Module] | None = None
        self.backward_prefetch: list[nn.Module] | None = None

    def set_modules_to_forward_prefetch(self, modules: list[nn.Module]) -> None:
        self.forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules: list[nn.Module]) -> None:
        self.backward_prefetch = list(modules)


def test_manual_prefetch_groups_by_parent_prefix_and_triggers_from_inner_ep():
    l0 = _RecordingFsdpStub("layers.0")
    l0_ep = _RecordingFsdpStub("layers.0.experts")
    l0._fsdp_modules = [l0_ep, l0]
    l0._extra_parallel_fsdp_modules = [l0_ep]

    l1 = _RecordingFsdpStub("layers.1")
    l1_ep = _RecordingFsdpStub("layers.1.experts")
    l1._fsdp_modules = [l1_ep, l1]
    l1._extra_parallel_fsdp_modules = [l1_ep]

    m0 = _RecordingFsdpStub("mtp.0")
    m0_ep = _RecordingFsdpStub("mtp.0.experts")
    m0._fsdp_modules = [m0_ep, m0]
    m0._extra_parallel_fsdp_modules = [m0_ep]

    m1 = _RecordingFsdpStub("mtp.1")
    m1_ep = _RecordingFsdpStub("mtp.1.experts")
    m1._fsdp_modules = [m1_ep, m1]
    m1._extra_parallel_fsdp_modules = [m1_ep]

    singleton = _RecordingFsdpStub("sidecar.0")
    singleton._fsdp_modules = [singleton]
    singleton._extra_parallel_fsdp_modules = []

    _configure_fsdp2_manual_prefetch(
        [
            ("model.layers.0", (l0, {})),
            ("model.layers.1", (l1, {})),
            ("model.mtp.0", (m0, {})),
            ("model.mtp.1", (m1, {})),
            ("model.sidecar.0", (singleton, {})),
        ]
    )

    # Forward chains stay strictly inside each parent container prefix.
    assert l0.forward_prefetch == [l1, l1_ep]
    assert l1.forward_prefetch is None
    assert m0.forward_prefetch == [m1, m1_ep]
    assert m1.forward_prefetch is None
    assert singleton.forward_prefetch is None

    # Backward prefetch attaches to the inner EP submodule and does not cross stacks.
    assert l1_ep.backward_prefetch == [l0, l0_ep]
    assert l1.backward_prefetch is None
    assert l0_ep.backward_prefetch is None
    assert m1_ep.backward_prefetch == [m0, m0_ep]
    assert m1.backward_prefetch is None
    assert m0_ep.backward_prefetch is None
    assert m0.backward_prefetch is None
    assert singleton.backward_prefetch is None


def test_manual_prefetch_backward_trigger_falls_back_for_dense_and_mp_ignored_only():
    dense0 = _RecordingFsdpStub("layers.0")
    dense0._fsdp_modules = [dense0]
    dense0._extra_parallel_fsdp_modules = []

    dense1 = _RecordingFsdpStub("layers.1")
    fp32_norm = _RecordingFsdpStub("layers.1.input_layernorm")
    dense1._fsdp_modules = [fp32_norm, dense1]
    dense1._extra_parallel_fsdp_modules = []

    moe2 = _RecordingFsdpStub("layers.2")
    ep_first = _RecordingFsdpStub("layers.2.ep_first")
    ep_last = _RecordingFsdpStub("layers.2.ep_last")
    moe2._fsdp_modules = [ep_first, ep_last, moe2]
    moe2._extra_parallel_fsdp_modules = [ep_first, ep_last]

    _configure_fsdp2_manual_prefetch(
        [
            ("model.layers.0", (dense0, {})),
            ("model.layers.1", (dense1, {})),
            ("model.layers.2", (moe2, {})),
        ]
    )

    # Pure mp_ignored_classes layer keeps backward prefetch on the outer block, not the early norm.
    assert dense1.backward_prefetch == [dense0]
    assert fp32_norm.backward_prefetch is None
    # Multi-EP-submodule layer triggers from the last inner EP submodule in forward order.
    assert ep_last.backward_prefetch == [dense1, fp32_norm]
    assert ep_first.backward_prefetch is None
    assert moe2.backward_prefetch is None


class _ToyExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.mean(dim=0))


class _ToyBlock(nn.Module):
    def __init__(self, num_experts: int, hidden: int, use_checkpoint: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.experts = _ToyExperts(num_experts, hidden)
        self.post_proj = nn.Linear(hidden, hidden, bias=False)
        self.use_checkpoint = use_checkpoint

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.post_proj(self.experts(self.proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._run, x, use_reentrant=False)
        return self._run(x)


class _ToyMultiStackMoe(nn.Module):
    _no_split_modules = ["_ToyBlock"]

    def __init__(self, use_checkpoint: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock(4, 8, use_checkpoint) for _ in range(2)])
        self.mtp = nn.ModuleList([_ToyBlock(4, 8, use_checkpoint) for _ in range(2)])

    def get_parallel_plan(self) -> ParallelPlan:
        return ParallelPlan(
            extra_parallel_plan={
                "ep": {
                    "layers.*.experts.weight": Shard(0),
                    "mtp.*.experts.weight": Shard(0),
                }
            }
        )

    def init_weights(self) -> None:
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(0.05)

    def forward(self, x: torch.Tensor, *, run_mtp: bool) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if run_mtp:
            for stage in self.mtp:
                x = stage(x)
        return x


def _run_and_verify_fsdp2_manual_prefetch(*, world_size: int, device_type: str, use_checkpoint: bool) -> None:
    _init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=1,
        dp_shard_size=world_size,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        extra_parallel_sizes=(2,),
        extra_parallel_names=("ep",),
        dp_mode="fsdp2",
        device_type=device_type,
    )
    with torch.device("meta"):
        model = _ToyMultiStackMoe(use_checkpoint=use_checkpoint)

    model = parallelize_model_fsdp2(
        model,
        weights_path=None,
        mixed_precision=MixedPrecisionConfig(enable=False),
        enable_forward_prefetch=True,
        init_device="meta",
        broadcast_model_weights_from_rank0=False,
    )

    # Confirm the expert FSDP group has the expected shard width (world_size // ep_size).
    assert model.layers[0].experts.weight.device_mesh.size() == world_size // 2

    # Verify actual FSDP2 prefetch wiring across the two stacks.
    def _fwd_targets(mod):
        return list(mod._get_fsdp_state()._states_to_forward_prefetch)

    def _bwd_targets(mod):
        return list(mod._get_fsdp_state()._states_to_backward_prefetch)

    assert _fwd_targets(model.layers[0]) == [
        model.layers[1]._get_fsdp_state(),
        model.layers[1].experts._get_fsdp_state(),
    ]
    assert _fwd_targets(model.layers[1]) == []
    assert _fwd_targets(model.mtp[0]) == [
        model.mtp[1]._get_fsdp_state(),
        model.mtp[1].experts._get_fsdp_state(),
    ]
    assert _bwd_targets(model.layers[1].experts) == [
        model.layers[0]._get_fsdp_state(),
        model.layers[0].experts._get_fsdp_state(),
    ]
    assert _bwd_targets(model.layers[1]) == []
    assert _bwd_targets(model.mtp[0].experts) == []
    assert _bwd_targets(model.mtp[1].experts) == [
        model.mtp[0]._get_fsdp_state(),
        model.mtp[0].experts._get_fsdp_state(),
    ]

    # Instrument real FSDP2 unshard launches and pre-backward hooks to validate runtime prefetch timing.
    pg_to_name = {}
    state_to_name = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "_get_fsdp_state"):
            state = mod._get_fsdp_state()
            state_to_name[state] = name or "root"
            if state._fsdp_param_group is not None:
                pg_to_name[state._fsdp_param_group] = name or "root"

    events: list[tuple[str, str]] = []
    orig_unshard = FSDPParamGroup.unshard
    orig_pre_bwd = FSDPState._pre_backward

    def _recording_unshard(self, async_op: bool = False):
        if self._all_gather_result is None and not self.is_unsharded:
            events.append(("unshard", pg_to_name.get(self, "?")))
        return orig_unshard(self, async_op=async_op)

    def _recording_pre_bwd(self, grad):
        events.append(("pre_bwd", state_to_name.get(self, "?")))
        return orig_pre_bwd(self, grad)

    device = torch.device(device_type)
    FSDPParamGroup.unshard = _recording_unshard
    FSDPState._pre_backward = _recording_pre_bwd
    try:
        # Exercise both a skipped-MTP step and an active-MTP step end to end.
        for run_mtp in (False, True):
            events.clear()
            model.zero_grad(set_to_none=True)
            x = torch.randn(2, 4, 8, device=device, requires_grad=True)
            loss = model(x, run_mtp=run_mtp).square().mean()
            bwd_start = len(events)
            loss.backward()
            assert torch.isfinite(loss)

            fwd_events = events[:bwd_start]
            bwd_events = events[bwd_start:]

            if not run_mtp:
                # Skipped MTP stack must never be unsharded in forward or backward.
                assert all(not target.startswith("mtp.") for _, target in fwd_events)
                assert all(not target.startswith("mtp.") for _, target in bwd_events)
            else:
                # Forward of layers.1 must not prefetch mtp.0; mtp.1 backward prefetches mtp.0 from mtp.1.experts.
                idx_mtp1_pre_bwd = bwd_events.index(("pre_bwd", "mtp.1"))
                idx_mtp1_exp_unshard = bwd_events.index(("unshard", "mtp.1.experts"))
                idx_mtp1_exp_pre_bwd = bwd_events.index(("pre_bwd", "mtp.1.experts"))
                idx_mtp0_unshard = bwd_events.index(("unshard", "mtp.0"))
                idx_mtp0_exp_unshard = bwd_events.index(("unshard", "mtp.0.experts"))
                assert idx_mtp1_pre_bwd < idx_mtp1_exp_unshard < idx_mtp1_exp_pre_bwd < idx_mtp0_unshard
                assert idx_mtp1_exp_pre_bwd < idx_mtp0_exp_unshard

            # In backward, layers.1 prefetches its own experts first, and layers.0 (+ experts) are
            # triggered only when layers.1.experts enters pre-backward.
            idx_l1_pre_bwd = bwd_events.index(("pre_bwd", "layers.1"))
            idx_l1_exp_unshard = bwd_events.index(("unshard", "layers.1.experts"))
            idx_l1_exp_pre_bwd = bwd_events.index(("pre_bwd", "layers.1.experts"))
            idx_l0_unshard = bwd_events.index(("unshard", "layers.0"))
            idx_l0_exp_unshard = bwd_events.index(("unshard", "layers.0.experts"))
            assert idx_l1_pre_bwd < idx_l1_exp_unshard < idx_l1_exp_pre_bwd < idx_l0_unshard
            assert idx_l1_exp_pre_bwd < idx_l0_exp_unshard

            for param in model.parameters():
                if param.grad is not None:
                    local_grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                    assert torch.isfinite(local_grad).all()
    finally:
        FSDPParamGroup.unshard = orig_unshard
        FSDPState._pre_backward = orig_pre_bwd


def _fsdp2_prefetch_cpu_worker(rank: int, use_checkpoint: bool, endpoint) -> None:
    import veomni.distributed.torch_parallelize as torch_parallelize

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2)
    orig_get_device_type = torch_parallelize.get_device_type
    orig_ckpt_device_type = DefaultDeviceType._default_device_type
    torch_parallelize.get_device_type = lambda: "cpu"
    # Keep non-reentrant checkpointing on CPU even when torch_npu/CUDA sets a non-CPU default
    # device type while accelerator devices are masked off for CPU unit tests.
    DefaultDeviceType.set_device_type("cpu")
    try:
        _run_and_verify_fsdp2_manual_prefetch(world_size=2, device_type="cpu", use_checkpoint=use_checkpoint)
    finally:
        torch_parallelize.get_device_type = orig_get_device_type
        DefaultDeviceType._default_device_type = orig_ckpt_device_type
        dist.destroy_process_group()
        clear_parallel_state()


def _fsdp2_prefetch_accelerator_worker(use_checkpoint: bool) -> None:
    try:
        _run_and_verify_fsdp2_manual_prefetch(
            world_size=dist.get_world_size(),
            device_type=get_device_type(),
            use_checkpoint=use_checkpoint,
        )
        dist.barrier()
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the CPU Gloo backend")
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_parallelize_model_fsdp2_manual_prefetch_e2e_gloo(use_checkpoint: bool) -> None:
    spawn_with_rendezvous(_fsdp2_prefetch_cpu_worker, args=(use_checkpoint,), nprocs=2)


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_4)
@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE),
    reason="requires 4 CUDA or NPU devices",
)
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_parallelize_model_fsdp2_manual_prefetch_e2e_accelerator(use_checkpoint: bool) -> None:
    torchrun(_fsdp2_prefetch_accelerator_worker, _CI_ACCELERATOR_4.nproc, use_checkpoint)
