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

"""Shared same-NK row contracts and schedule reuse across expert variants."""

from collections import Counter
from functools import wraps

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ops.utils import require_nvidia_cuda
from veomni.utils.device import get_device_type, get_torch_device


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("total", [None, -1, 3, 5, torch.tensor(4)])
def test_tile_total_contract_precedes_allocation(total, monkeypatch):
    """Reject bad host metadata before output allocation or kernel dispatch."""
    pytest.importorskip("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm

    a = torch.ones(4, 8, dtype=torch.bfloat16)
    b = torch.ones(2, 8, 4, dtype=torch.bfloat16)
    cumsum = torch.tensor([2, 4], dtype=torch.int64)
    prefix = torch.tensor([1, 2], dtype=torch.int64)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("The invalid prefix contract must fail before output allocation")

    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    error = TypeError if type(total) is not int else ValueError
    with pytest.raises(error, match="[Tt]ile.*total"):
        gemm.group_gemm_same_nk(a, b, cumsum, 4, tile_cumsum=prefix, tile_total_M=total)


def test_device_scalar_total_is_rejected_without_synchronizing():
    """A tensor cannot turn the host-only contract into a hidden scalar read."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm

    device = get_device_type()
    a = torch.ones(4, 8, device=device, dtype=torch.bfloat16)
    b = torch.ones(2, 8, 4, device=device, dtype=torch.bfloat16)
    cumsum = torch.tensor([2, 4], device=device, dtype=torch.int64)
    prefix = torch.tensor([1, 2], device=device, dtype=torch.int64)
    total = torch.tensor(4, device=device)
    api = get_torch_device()
    previous = api.get_sync_debug_mode()
    api.set_sync_debug_mode("error")
    try:
        with pytest.raises(TypeError, match="Python int"):
            gemm.same_nk_tile_schedule(cumsum, total)
        with pytest.raises(TypeError, match="Python int"):
            gemm.group_gemm_same_nk(a, b, cumsum, 4, tile_cumsum=prefix, tile_total_M=total)
    finally:
        api.set_sync_debug_mode(previous)


def _observe_expert_schedules(monkeypatch):
    """Observe native calls while checking each expert forward/backward for sync."""
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm
    from veomni.ops.kernels.moe_experts.shared.ep import EPGroupGemm, EPMergedFc1GroupGemm
    from veomni.ops.kernels.moe_experts.standard import triton as standard
    from veomni.ops.kernels.moe_experts_lora.independent import triton as independent
    from veomni.ops.kernels.moe_experts_lora.shared import triton as shared

    classes = [
        standard.TritonFusedMoeExpertFunction,
        standard.MergedFc1TritonFusedMoeExpertFunction,
        EPGroupGemm,
        EPMergedFc1GroupGemm,
        shared.MergedFc1TritonFusedLoRAMoeExpertFunction,
        independent.MergedFc1IndependentTritonFusedLoRAMoeExpertFunction,
        shared.EPMergedFc1SharedLoRAGroupGemm,
        independent.EPMergedFc1IndependentLoRAGroupGemm,
    ]
    owner = [None]
    prefixes, calls = {}, []
    original_prefix, original_nk = gemm.merged_same_nk_tile_prefix, gemm.group_gemm_same_nk
    api = get_torch_device()

    def checked_method(function, name, phase):
        @wraps(function)
        def checked(*args, **kwargs):
            previous_owner, previous_mode = owner[0], api.get_sync_debug_mode()
            owner[0] = (name, phase)
            api.set_sync_debug_mode("error")
            try:
                return function(*args, **kwargs)
            finally:
                api.set_sync_debug_mode(previous_mode)
                owner[0] = previous_owner

        return checked

    for cls in classes:
        for method in ["forward", "backward"]:
            monkeypatch.setattr(cls, method, staticmethod(checked_method(getattr(cls, method), cls.__name__, method)))

    def build(cumsum, total):
        prefix = original_prefix(cumsum, total)
        assert prefix is not None
        assert prefix.data_ptr() not in prefixes
        assert owner[0][1] == "forward"
        prefixes[prefix.data_ptr()] = (owner[0][0], cumsum.data_ptr(), total, prefix)
        return prefix

    def consume(*args, **kwargs):
        prefix = kwargs.get("tile_cumsum")
        total = kwargs.get("tile_total_M")
        if prefix is not None:
            built_owner, boundary_pointer, built_total, _ = prefixes[prefix.data_ptr()]
            assert built_owner == owner[0][0]
            assert boundary_pointer == kwargs["cumsum_M"].data_ptr()
            assert type(total) is int and total == built_total == kwargs["a"].shape[0]
        else:
            assert total is None
        calls.append((owner[0][0], prefix.data_ptr() if prefix is not None else None, owner[0][1]))
        return original_nk(*args, **kwargs)

    monkeypatch.setattr(gemm, "merged_same_nk_tile_prefix", build)
    monkeypatch.setattr(gemm, "group_gemm_same_nk", consume)
    for module in [shared, independent]:
        monkeypatch.setattr(module, "group_gemm_same_nk", consume)
    return prefixes, calls


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("variant", ["ep_split", "ep_merged", "shared", "independent", "ep_shared", "ep_independent"])
def test_expert_variants_reuse_one_schedule_and_match_native_oracles(variant, fallback, monkeypatch):
    """Exercise EP and LoRA numerics with their actual compact or fallback consumers."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import scheduling

    if fallback:
        monkeypatch.setattr(scheduling, "must_use_shape_derived_bound", lambda: True)
    prefixes, calls = _observe_expert_schedules(monkeypatch)
    if variant in ["ep_split", "ep_merged"]:
        from tests.ops.moe_experts import test_moe_experts_ep as native

        case = native.test_ep_vs_non_ep if variant == "ep_split" else native.test_ep_merged_vs_non_ep
        case(128, 4, 512, 256, 2, 1, None)
        expected_owner = "EPGroupGemm" if variant == "ep_split" else "EPMergedFc1GroupGemm"
    else:
        from tests.ops.moe_experts_lora import test_moe_experts_lora as native

        mode = variant.removeprefix("ep_")
        if variant.startswith("ep_"):
            native.test_triton_ep_class_matches_nonep_single_rank(mode)
            expected_owner = (
                "EPMergedFc1SharedLoRAGroupGemm" if mode == "shared" else "EPMergedFc1IndependentLoRAGroupGemm"
            )
        else:
            native._run_fused_vs_eager("fused_triton", mode, shape="small", base_trainable=True)
            expected_owner = (
                "MergedFc1TritonFusedLoRAMoeExpertFunction"
                if mode == "shared"
                else "MergedFc1IndependentTritonFusedLoRAMoeExpertFunction"
            )

    assert any(name == expected_owner for name, _, _ in calls)
    if fallback:
        assert not prefixes
        assert all(pointer is None for _, pointer, _ in calls)
    else:
        counts = Counter((pointer, phase) for _, pointer, phase in calls)
        owner_counts = Counter(info[0] for info in prefixes.values())
        assert owner_counts[expected_owner] == (2 if variant in ["ep_split", "ep_merged"] else 1)
        completed = False
        for pointer, (name, _, _, _) in prefixes.items():
            forward_calls = 8 if "Independent" in name else 3 if name == "EPGroupGemm" else 2
            assert counts[pointer, "forward"] == forward_calls
            assert counts[pointer, "backward"] in [0, forward_calls]
            completed |= name == expected_owner and counts[pointer, "backward"] == forward_calls
        assert completed


def test_same_total_different_boundaries_have_independent_prefixes():
    """Equal row totals cannot justify sharing a prefix between different routes."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared.group_gemm import same_nk_tile_schedule

    device = get_device_type()
    first = torch.tensor([0, 128, 256, 256], device=device, dtype=torch.int64)
    second = torch.tensor([127, 128, 129, 256], device=device, dtype=torch.int64)
    prefix_a, total_a = same_nk_tile_schedule(first, 256)
    prefix_b, total_b = same_nk_tile_schedule(second, 256)
    assert total_a == total_b == 256
    assert prefix_a.data_ptr() != prefix_b.data_ptr()
    torch.testing.assert_close(prefix_a, torch.tensor([0, 1, 2, 2], device=device), rtol=0, atol=0)
    torch.testing.assert_close(prefix_b, torch.tensor([1, 2, 3, 4], device=device), rtol=0, atol=0)
