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

"""CPU/Gloo coverage for the job-wide NPU expert precision handshake."""

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.npu_moe_precision import (
    require_expert_precision_consensus,
    validate_expert_precision_consensus,
    validate_qwen4_experts,
)


def _accelerator(enabled, *, reduce_dtype="float32", low_comm=False):
    mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=enabled, reduce_dtype=reduce_dtype)
    fsdp = SimpleNamespace(mixed_precision=mixed, fsdp_mode="fsdp2", low_precision_reduce_scatter_comm=low_comm)
    return SimpleNamespace(fsdp_config=fsdp, tp_size=1, pp_size=1, torch_compile=SimpleNamespace(enable=False))


def _worker(rank, directory, case):
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        enabled = case != "disabled" and (case != "mismatch_flag" or rank == 0)
        dtype = "bfloat16" if case in ("disabled", "mismatch_dtype") and rank == 1 else "float32"
        accelerator = _accelerator(enabled, reduce_dtype=dtype, low_comm=case == "low_comm")
        if case == "mismatch_output_dtype" and rank == 1:
            accelerator.fsdp_config.mixed_precision.output_dtype = "float32"
        try:
            backend = "eager" if case == "mismatch_backend" and rank == 1 else "fused_npu"
            validate_expert_precision_consensus(accelerator, backend)
            status = "ok"
            if enabled:
                mixed = accelerator.fsdp_config.mixed_precision
                require_expert_precision_consensus(mixed, "fsdp2", False, 1, 1, False)
                altered = copy.deepcopy(mixed)
                altered.enable = False
                with pytest.raises(ValueError, match="unchanged job-wide"):
                    require_expert_precision_consensus(altered, "fsdp2", False, 1, 1, False)
                altered = copy.deepcopy(mixed)
                altered.output_dtype = "float32"
                with pytest.raises(ValueError, match="unchanged job-wide"):
                    require_expert_precision_consensus(altered, "fsdp2", False, 1, 1, False)
                if case == "mismatch_module":
                    import torch.nn as nn

                    class Qwen4ExpTextExperts(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.veomni_moe = SimpleNamespace(impl="eager" if rank == 1 else "fused_npu")
                            self.use_swiglu_mlp = True

                    model = nn.Module()
                    model.config = SimpleNamespace(model_type="qwen4_exp")
                    model.experts = Qwen4ExpTextExperts()
                    validate_qwen4_experts(model)
        except ValueError as exc:
            status = str(exc)
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps({"status": status}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("disabled", "ok"),
        ("enabled", "ok"),
        ("mismatch_flag", "identical precision configuration"),
        ("mismatch_dtype", "identical precision configuration"),
        ("mismatch_output_dtype", "identical precision configuration"),
        ("mismatch_backend", "identical precision configuration"),
        ("mismatch_module", "identical Qwen4-Exp expert modules"),
        ("low_comm", "native communication"),
    ],
)
def test_all_ranks_agree_before_mesh(tmp_path, case, expected):
    mp.spawn(_worker, args=(str(tmp_path), case), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert expected in result["status"]


def test_caller_without_consensus_cannot_opt_in():
    mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=True)
    with pytest.raises(ValueError, match="job-wide precision consensus"):
        require_expert_precision_consensus(mixed, "fsdp2", False, 1, 1, False)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_expert_precision_option_requires_real_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        MixedPrecisionConfig(npu_expert_fp32_wgrad=value)


def test_direct_fsdp2_cannot_bypass_consensus(monkeypatch):
    from veomni.distributed import torch_parallelize

    state = SimpleNamespace(dp_mode="fsdp2", tp_size=1, pp_size=1, fsdp_enabled=True)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: state)
    with pytest.raises(ValueError, match="job-wide precision consensus"):
        torch_parallelize.parallelize_model_fsdp2(
            object(), mixed_precision=MixedPrecisionConfig(npu_expert_fp32_wgrad=True)
        )


def test_qwen4_vlm_decoder_layers_are_fsdp_targets():
    import torch
    from transformers import Qwen4ExpConfig

    from tests.models.compare import eager_ops_config, ops_config_scope
    from veomni.distributed.torch_parallelize import _fsdp_target_classes, _is_fsdp_wrap_target
    from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_npu import (
        Qwen4ExpForConditionalGeneration,
    )

    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json")
    config._attn_implementation = "eager"
    with ops_config_scope(eager_ops_config()), torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration(config)
    targets = _fsdp_target_classes(model, [], True)
    blocks = {
        name for name, module in model.named_modules() if _is_fsdp_wrap_target(name, type(module).__name__, targets)
    }
    experts = [name for name, module in model.named_modules() if type(module).__name__ == "Qwen4ExpTextExperts"]
    assert experts and all(any(name.startswith(block + ".") for block in blocks) for name in experts)
    # Also cover a root that has not inherited the child model's no-split list.
    assert "Qwen4ExpTextDecoderLayer" in _fsdp_target_classes(SimpleNamespace(_no_split_modules=None), [], True)
    assert not _fsdp_target_classes(SimpleNamespace(_no_split_modules=None), [], False)


@pytest.mark.parametrize("for_inference", [False, True])
def test_seed_omni_module_does_not_silently_ignore_opt_in(for_inference):
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime

    accelerator = _accelerator(True)
    args = SimpleNamespace(accelerator=accelerator)
    with pytest.raises(ValueError, match="unsupported by SeedOmni"):
        ModuleRuntime(args, "test", module_config=None, global_accelerator=accelerator, for_inference=for_inference)
