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

"""Adapter binding contracts and two-rank training tests for FSDPTurbo."""

from functools import partial
from importlib import import_module
from importlib.util import find_spec
from math import isfinite

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools import DummyDataset, ParallelConfig, materialize_weights, run_training_config
from veomni.arguments import AcceleratorConfig, FSDPConfig, FSDPTurboConfig, ModelArguments
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


TOY_CONFIG = "./tests/toy_config/qwen3_toy"
_TURBO_LAUNCH = LaunchSpec(nproc=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_fsdp_turbo_patch_bindings(monkeypatch):
    if find_spec("fsdp_turbo") is None:
        pytest.skip("Requires the optional FSDPTurbo dependency.")
    adapter = import_module("fsdp_turbo.integration.veomni")
    dependencies = {
        "veomni.distributed.torch_parallelize": ("sort_fqn_by_submodule_first",),
        "veomni.distributed.parallel_state": ("get_parallel_state",),
    }
    for module, names in dependencies.items():
        for name in names:
            assert callable(getattr(import_module(module), name)), f"{module}.{name} is unavailable"

    tp_mod = import_module("veomni.distributed.torch_parallelize")
    moe_layer_mod = import_module("veomni.distributed.moe.moe_layer")
    orig_shard = tp_mod.fully_shard
    orig_parallelize = tp_mod.parallelize_model_fsdp2
    orig_all_to_all = moe_layer_mod.all_to_all

    monkeypatch.setenv("WORLD_SIZE", "2")
    args = ModelArguments(
        model_path="unused",
        accelerator=AcceleratorConfig(
            ep_size=2,
            fsdp_config=FSDPConfig(
                fsdp_mode="turbo_fsdp2",
                fsdp_turbo=FSDPTurboConfig(prefetch_wait_a2a=True),
            ),
        ),
    )
    try:
        adapter.apply_fsdp_turbo_patch(args)
        assert tp_mod.fully_shard is not orig_shard
        assert tp_mod.parallelize_model_fsdp2 is not orig_parallelize
        assert moe_layer_mod.all_to_all is not orig_all_to_all
    finally:
        adapter.remove_fsdp_turbo_patch()
    assert tp_mod.fully_shard is orig_shard
    assert tp_mod.parallelize_model_fsdp2 is orig_parallelize
    assert moe_layer_mod.all_to_all is orig_all_to_all


@pytest.mark.launch_plan(provider=_TURBO_LAUNCH)
def test_fsdp_turbo_training(tmp_path):
    if find_spec("fsdp_turbo") is None:
        pytest.skip("Requires the optional FSDPTurbo dependency.")
    if not (IS_NPU_AVAILABLE or IS_CUDA_AVAILABLE) or runner_device_count() < _TURBO_LAUNCH.nproc:
        pytest.skip("Requires two accelerator devices.")

    model_path = tmp_path / "model"
    materialize_weights(TOY_CONFIG, str(model_path))
    dataset = DummyDataset(
        num_samples=24,
        seq_len=64,
        dataset_type="text",
        cache_name=f"turbo_adamw_{tmp_path.name}",
    )
    try:
        run = partial(
            run_training_config,
            script="tests/train_scripts/train_text_test.py",
            config_path=TOY_CONFIG,
            model_path=str(model_path),
            train_path=dataset.save_path,
            output_dir=str(tmp_path),
            parallel_config=ParallelConfig(fsdp_mode="turbo_fsdp2"),
            nproc=_TURBO_LAUNCH.nproc,
            model_name="qwen3",
        )
        extra_args = [
            "--model.optimizer.type=adamw",
            "--data.max_seq_len=64",
            "--train.max_steps=3",
            "--data.dataloader.shuffle=False",
            "--train.checkpoint.save_steps=1",
            "--model.accelerator.fsdp_config.fsdp_turbo.forward_prefetch_depth=2",
            "--model.accelerator.fsdp_config.fsdp_turbo.backward_prefetch_depth=2",
            "--model.accelerator.fsdp_config.fsdp_turbo.max_inflight_all_gathers=3",
            "--model.accelerator.fsdp_config.fsdp_turbo.max_inflight_reductions=3",
        ]
        result = run(task_name="turbo_adamw", extra_args=extra_args)
        for metric in ("loss", "grad_norm"):
            assert len(result[metric]) == 3
            assert all(isfinite(value) for value in result[metric])

        native_args = [arg for arg in extra_args if not arg.startswith("--model.accelerator.fsdp_config.fsdp_turbo.")]
        native = run(
            task_name="native_adamw",
            parallel_config=ParallelConfig(fsdp_mode="fsdp2"),
            extra_args=[*native_args, "--train.checkpoint.save_steps=0"],
        )
        for metric in ("loss", "grad_norm"):
            assert len(native[metric]) == len(result[metric])
            assert result[metric] == pytest.approx(native[metric], rel=1e-5, abs=1e-6)

        checkpoint = tmp_path / "turbo_adamw" / "checkpoints" / "global_step_1"
        resumed = run(
            task_name="turbo_adamw_resumed",
            extra_args=[
                *extra_args,
                f"--train.checkpoint.load_path={checkpoint}",
                "--train.checkpoint.save_steps=0",
            ],
        )
        for metric in ("loss", "grad_norm"):
            assert len(resumed[metric]) == 2
            assert resumed[metric] == pytest.approx(result[metric][1:], rel=1e-5, abs=1e-6)
    finally:
        dataset.clean_cache()
