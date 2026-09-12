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

"""NPU parity coverage for transformers-v5 generated modeling and loading.

The CUDA suite is a bitwise baseline across GPU-only and shared model families.
This module intentionally selects eager-fp32 cases that have an independently
generated NPU modeling file. Ascend matmul kernels are compared numerically,
while model/input construction and the disk-backed loader path are reused from
the CUDA suite so both accelerators exercise the same semantic contract.
"""

from __future__ import annotations

import copy
import gc
import tempfile

import pytest
import torch
import torch.distributed as dist

from veomni.utils.device import (
    IS_NPU_AVAILABLE,
    empty_cache,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)

from . import test_models_logits_equal_v5 as _base


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="transformers-v5 NPU parity requires torch_npu")

_NPU_RTOL = 1e-3
_NPU_ATOL = 1e-3


def _case(case_id: str) -> _base.Case:
    return next(case for case in _base.CASES if case.case_id == case_id)


def _loader_case(case_id: str) -> _base.Case:
    return next(case for case in _base._LOADER_CASES if case.case_id == case_id)


# Only include eager-fp32 cases with a patchgen-generated ``*_npu.py`` module.
# GPU-only generated families (Qwen2/Qwen2-VL/Qwen2.5-VL/GPT-OSS/Qwen2.5-Omni)
# stay in the CUDA suite. NPU-generated VLM/MoE/Omni cases whose existing base
# case requires FA2 or SDPA+bf16 are deferred until those paths are validated in
# the image environment.
_NPU_FORWARD_CASES = [
    _case("qwen3-eager"),
    _case("qwen3_moe-eager"),
    _case("deepseek_v3-eager"),
    _case("glm_moe_dsa-eager"),
    _case("qwen3_5-text-eager"),
    _case("qwen3_5_moe-text-eager"),
    _case("qwen3_vl-eager"),
]

_NPU_LOADER_CASES = [
    _loader_case("qwen3_moe-eager-loader"),
    _loader_case("deepseek_v3-eager-loader"),
    _loader_case("glm_moe_dsa-eager-loader"),
]


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """Provide the one-rank HCCL group required by SP-aware wrappers."""
    if not IS_NPU_AVAILABLE:
        yield
        return

    initialized_here = False
    if not dist.is_initialized():
        get_torch_device().set_device(0)
        dist.init_process_group(backend=get_dist_comm_backend(), rank=0, world_size=1)
        initialized_here = True
    try:
        yield
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def _release_npu() -> None:
    synchronize()
    gc.collect()
    empty_cache()


def _apply_npu_determinism() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor, case_id: str) -> None:
    assert actual.shape == expected.shape, (
        f"[{case_id}] shape mismatch: hf={tuple(expected.shape)} veomni={tuple(actual.shape)}"
    )
    assert torch.isfinite(actual).all(), f"[{case_id}] VeOmni logits contain non-finite values"
    assert torch.isfinite(expected).all(), f"[{case_id}] HF logits contain non-finite values"
    torch.testing.assert_close(
        actual,
        expected,
        rtol=_NPU_RTOL,
        atol=_NPU_ATOL,
        msg=lambda msg: f"[{case_id}] NPU generated-model logits differ from pristine HF: {msg}",
    )


def _assert_npu_generated_model(model: torch.nn.Module, case_id: str) -> None:
    module_name = type(model).__module__
    assert ".generated." in module_name and module_name.endswith("_npu"), (
        f"[{case_id}] expected NPU generated modeling, got {module_name}"
    )


@pytest.mark.parametrize("case", _NPU_FORWARD_CASES, ids=[case.case_id for case in _NPU_FORWARD_CASES])
def test_logits_equal_v5_npu(case: _base.Case) -> None:
    """Pristine HF and VeOmni NPU generated modeling produce matching logits."""
    _apply_npu_determinism()
    config = _base._make_config(case)
    dtype = _base._DTYPE_MAP[case.dtype]
    input_ids, forward_kwargs = _base._make_inputs(case, config, get_device_type(), dtype)

    model_hf = _base._build_hf_model(case, config, dtype)
    with torch.no_grad():
        logits_hf = (
            _base._forward_target(model_hf, case)(input_ids=input_ids.clone(), use_cache=False, **forward_kwargs)
            .logits.detach()
            .clone()
        )
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    del model_hf
    _release_npu()

    model_veomni = _base._build_veomni_model(case, config, hf_state_dict)
    _assert_npu_generated_model(model_veomni, case.case_id)
    with torch.no_grad():
        logits_veomni = (
            _base._forward_target(model_veomni, case)(input_ids=input_ids.clone(), use_cache=False, **forward_kwargs)
            .logits.detach()
            .clone()
        )

    _assert_logits_close(logits_veomni, logits_hf, case.case_id)
    del model_veomni, hf_state_dict, logits_hf, logits_veomni
    _release_npu()


@pytest.mark.parametrize("case", _NPU_LOADER_CASES, ids=[case.case_id for case in _NPU_LOADER_CASES])
def test_logits_equal_v5_via_loader_npu(case: _base.Case) -> None:
    """The NPU disk-backed loader preserves the pristine HF logits contract."""
    _apply_npu_determinism()
    config = _base._make_config(case)
    dtype = _base._DTYPE_MAP[case.dtype]
    input_ids, forward_kwargs = _base._make_inputs(case, config, get_device_type(), dtype)

    model_hf = _base._build_hf_model(case, config, dtype)
    with torch.no_grad():
        logits_hf = (
            _base._forward_target(model_hf, case)(input_ids=input_ids.clone(), use_cache=False, **forward_kwargs)
            .logits.detach()
            .clone()
        )
    hf_state_dict = copy.deepcopy(model_hf.state_dict())
    hf_buffers = {name: buffer.detach().clone() for name, buffer in model_hf.named_buffers()}
    del model_hf
    _release_npu()

    with tempfile.TemporaryDirectory(prefix="veomni_v5_npu_loader_test_") as tmp_dir:
        model_veomni = _base._build_veomni_model_from_disk(
            case,
            config,
            hf_state_dict,
            hf_buffers,
            tmp_dir,
        )
        _assert_npu_generated_model(model_veomni, case.case_id)
        with torch.no_grad():
            logits_veomni = (
                _base._forward_target(model_veomni, case)(
                    input_ids=input_ids.clone(), use_cache=False, **forward_kwargs
                )
                .logits.detach()
                .clone()
            )

    _assert_logits_close(logits_veomni, logits_hf, case.case_id)
    del model_veomni, hf_state_dict, hf_buffers, logits_hf, logits_veomni
    _release_npu()
