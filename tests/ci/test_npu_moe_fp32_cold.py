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

"""FSDP2 + expert-parallel optimizer/DCP cold-recovery regression on Ascend."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_capabilities
from tests.ci._npu_moe_fp32_cold_worker import run, run_selective
from veomni.checkpoint.layout import weights_dir
from veomni.utils.device import IS_NPU_AVAILABLE


def _assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    else:
        assert left == right


def _assert_close_across_devices(left, right, path="root"):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_close_across_devices(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype
        if not left.is_floating_point():
            assert torch.equal(left, right)
            return
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        difference = left.double() - right.double()
        denominator = max(
            torch.linalg.vector_norm(left.double()).item(), torch.linalg.vector_norm(right.double()).item()
        )
        relative_l2 = torch.linalg.vector_norm(difference).item() / denominator if denominator else 0.0
        max_abs = difference.abs().max().item() if difference.numel() else 0.0
        print(f"{path}: max_abs={max_abs:.9g}, relative_l2={relative_l2:.9g}", flush=True)
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-12)
        assert relative_l2 <= 1e-5
    else:
        assert left == right


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_cross_device_comparison_rejects_precision_or_dtype_regressions():
    reference = {"tensor": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    _assert_close_across_devices(reference, {"tensor": reference["tensor"].clone()})
    with pytest.raises(AssertionError):
        _assert_close_across_devices(reference, {"tensor": torch.tensor([1.0, 2.001])})
    with pytest.raises(AssertionError):
        _assert_close_across_devices(reference, {"tensor": reference["tensor"].double()})


def _assert_ep2_checkpoint_layout(checkpoint_dir):
    weights = FileSystemReader(weights_dir(str(checkpoint_dir))).read_metadata().state_dict_metadata
    for name, shape in (
        ("layer.experts.gate_up_proj", (4, 64, 256)),
        ("layer.experts.down_proj", (4, 256, 32)),
    ):
        chunks = {((0, 0, 0), (2, *shape[1:])), ((2, 0, 0), (2, *shape[1:]))}
        metadata = weights[f"model.{name}"]
        assert tuple(metadata.size) == shape
        assert metadata.properties.dtype == torch.float32
        assert len(metadata.chunks) == len(chunks)
        assert {(tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in metadata.chunks} == chunks


def _require_two_native_npus():
    """Fail an assigned or explicitly required NPU test when its lease is unusable."""
    capabilities = runner_capabilities() if os.environ.get("VEOMNI_CI_CAPABILITIES") else None
    required = os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1"
    if capabilities is not None and capabilities.backend != "npu":
        if required:
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires an NPU CI assignment")
        pytest.skip("requires an NPU CI assignment")
    npu = getattr(torch, "npu", None)
    if not IS_NPU_AVAILABLE or npu is None or not npu.is_available() or npu.device_count() < 2:
        if required or (capabilities is not None and capabilities.backend == "npu"):
            pytest.fail("Assigned or required NPU runtime must provide two usable devices")
        pytest.skip("requires two native Ascend devices")


def _native_npu_launch(context):
    """Keep an unavailable required test runnable so its guard reports failure."""
    caps = context.capabilities
    if caps.backend == "npu" and len(caps.devices) >= 2:
        return LaunchSpec(nproc=2, device="accelerator")
    return LaunchSpec(device="cpu")


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "backend,required,available,count,outcome",
    [
        (None, False, False, 0, "skip"),
        (None, True, False, 0, "fail"),
        ("npu", False, False, 0, "fail"),
        ("npu", False, True, 1, "fail"),
        ("npu", False, True, 2, "pass"),
        ("cuda", False, True, 2, "skip"),
        ("cuda", True, True, 2, "fail"),
    ],
)
def test_native_npu_guard_respects_ci_assignment(monkeypatch, backend, required, available, count, outcome):
    monkeypatch.setitem(globals(), "IS_NPU_AVAILABLE", available)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: available, device_count=lambda: count), raising=False
    )
    if backend is None:
        monkeypatch.delenv("VEOMNI_CI_CAPABILITIES", raising=False)
    else:
        monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", "test")
        monkeypatch.setitem(globals(), "runner_capabilities", lambda: SimpleNamespace(backend=backend))
    monkeypatch.setenv("VEOMNI_REQUIRE_NPU_TESTS", "1" if required else "0")
    if outcome == "fail":
        with pytest.raises(pytest.fail.Exception):
            _require_two_native_npus()
    elif outcome == "skip":
        with pytest.raises(pytest.skip.Exception):
            _require_two_native_npus()
    else:
        _require_two_native_npus()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_native_npu_launch_keeps_missing_devices_runnable():
    assert _native_npu_launch(SimpleNamespace(capabilities=SimpleNamespace(backend="cpu", devices=()))) == LaunchSpec(
        device="cpu"
    )
    assert _native_npu_launch(
        SimpleNamespace(capabilities=SimpleNamespace(backend="npu", devices=("0",)))
    ) == LaunchSpec(device="cpu")
    assert _native_npu_launch(
        SimpleNamespace(capabilities=SimpleNamespace(backend="npu", devices=("0", "1")))
    ) == LaunchSpec(nproc=2, device="accelerator")


@pytest.mark.launch_plan(provider=_native_npu_launch)
@pytest.mark.parametrize("ep", [1, 2])
def test_native_fsdp_expert_fp32_dcp_cold_resume(tmp_path, ep):
    _require_two_native_npus()
    from tests.tools.launch_utils import spawn_with_rendezvous

    for phase in ("fresh", "resume"):
        spawn_with_rendezvous(run, args=(phase, str(tmp_path), ep), nprocs=2)
    fresh_by_rank = []
    for rank in range(2):
        fresh = torch.load(tmp_path / f"ep{ep}-fresh-rank{rank}.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(tmp_path / f"ep{ep}-resume-rank{rank}.pt", map_location="cpu", weights_only=True)
        _assert_equal(fresh, resumed)
        fresh_by_rank.append(fresh)
    if ep == 2:
        for name in fresh_by_rank[0]["before"]:
            for row in (0, 1):
                left = fresh_by_rank[0]["before"][name]["param"][row]
                right = fresh_by_rank[1]["before"][name]["param"][row]
                assert not torch.equal(left, right), f"symmetric EP fixture for {name}: param row {row}"
            for field in ("m", "v"):
                left = fresh_by_rank[0]["before"][name][field][1]
                right = fresh_by_rank[1]["before"][name][field][1]
                assert not torch.equal(left, right), f"symmetric EP fixture for {name}: {field}"
        _assert_ep2_checkpoint_layout(tmp_path / "ep2-dcp")


@pytest.mark.launch_plan(provider=_native_npu_launch)
def test_native_fp32_expert_selective_offload_cold_resume(tmp_path):
    _require_two_native_npus()
    from tests.tools.launch_utils import spawn_with_rendezvous

    results = {}
    for resident in (False, True):
        output = tmp_path / ("resident" if resident else "offloaded")
        output.mkdir()
        for phase in ("fresh", "resume"):
            spawn_with_rendezvous(run_selective, args=(phase, str(output), resident), nprocs=2)
        results[resident] = []
        for rank in range(2):
            fresh = torch.load(output / f"ep2-fresh-rank{rank}.pt", map_location="cpu", weights_only=True)
            resumed = torch.load(output / f"ep2-resume-rank{rank}.pt", map_location="cpu", weights_only=True)
            _assert_equal(fresh, resumed)
            results[resident].append(fresh)
        _assert_ep2_checkpoint_layout(output / "ep2-dcp")
    for offloaded, resident in zip(results[False], results[True], strict=True):
        # AdamW executes on CPU for offloaded shards and on NPU for resident
        # shards. Their floating-point arithmetic need not be bitwise identical.
        _assert_close_across_devices(offloaded, resident)
