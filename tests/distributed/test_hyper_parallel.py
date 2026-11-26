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

"""Adapter binding contracts and two-NPU training tests for HyperParallel."""

from functools import partial
from importlib import import_module
from importlib.util import find_spec
from math import isfinite

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools import DummyDataset, ParallelConfig, materialize_weights, run_training_config
from veomni.arguments import AcceleratorConfig, FSDPConfig, ModelArguments, OptimizerConfig
from veomni.utils.device import get_device_type


TOY_CONFIG = "./tests/toy_config/qwen3_toy"
_HYPER_LAUNCH = LaunchSpec(nproc=2)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_hyper_patch_bindings():
    if find_spec("hyper_parallel") is None:
        pytest.skip("Requires the optional HyperParallel dependency.")
    adapter = import_module("hyper_parallel.integration.veomni")
    dependencies = {
        "veomni.trainer.vlm_trainer": ("_get_vlm_visual_module",),
        "veomni.optim.optimizer": (
            "_collect_muon_kwargs",
            "_make_param_groups_for_subset",
            "_should_build_extra_parallel_aware",
            "_is_extra_parallel_param",
        ),
        "veomni.distributed.torch_parallelize": ("sort_fqn_by_submodule_first",),
        "veomni.lora.state_dict": ("strip_adapter_name",),
        "veomni.utils.dtensor_utils": ("rewrap_dtensor_local",),
    }
    for module, names in dependencies.items():
        for name in names:
            assert callable(getattr(import_module(module), name)), f"{module}.{name} is unavailable"
    targets = {
        "veomni.models.checkpoint.weights": ("_dispatch_parameter", "_find_submodule"),
        "veomni.distributed.fsdp2.clip_grad_norm": ("clip_grad_norm",),
        "veomni.optim.optimizer": ("build_optimizer",),
        "veomni.checkpoint.dcp_checkpointer": ("OptimizerState", "get_model_state_dict", "set_model_state_dict"),
        "veomni.distributed.torch_parallelize": ("fully_shard", "FSDPModule", "parallelize_model_fsdp2"),
        "veomni.lora.state_dict": ("get_lora_state_dict",),
        "veomni.models.checkpoint.convert": ("export_weights",),
        "veomni.distributed.emb_parallel.sharded_embedding": ("register_fsdp_forward_method",),
    }
    originals = [
        (import_module(module), name, getattr(import_module(module), name))
        for module, names in targets.items()
        for name in names
    ]
    args = ModelArguments(
        model_path="unused",
        accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="hyper_fsdp2")),
        optimizer=OptimizerConfig(type="hyper_muon"),
    )
    try:
        adapter.apply_hyper_parallel_patch(args)
        for module, name, original in originals:
            assert getattr(module, name) is not original, f"{module.__name__}.{name} was not patched"
    finally:
        adapter.remove_hyper_parallel_patch()
    for module, name, original in originals:
        assert getattr(module, name) is original, f"{module.__name__}.{name} was not restored"


def _run_hyper_training(tmp_path, optimizer_type, *, check_resume=False):
    model_path = tmp_path / "model"
    materialize_weights(TOY_CONFIG, str(model_path))
    dataset = DummyDataset(
        num_samples=24 if check_resume else 16,
        seq_len=64,
        dataset_type="text",
        cache_name=f"hyper_{optimizer_type}_{tmp_path.name}",
    )
    try:
        run = partial(
            run_training_config,
            script="tests/train_scripts/train_text_test.py",
            config_path=TOY_CONFIG,
            model_path=str(model_path),
            train_path=dataset.save_path,
            output_dir=str(tmp_path),
            parallel_config=ParallelConfig(fsdp_mode="hyper_fsdp2"),
            nproc=_HYPER_LAUNCH.nproc,
            model_name="qwen3",
        )
        extra_args = [
            f"--model.optimizer.type={optimizer_type}",
            "--model.optimizer.muon_ns_implementation=std",
            "--data.max_seq_len=64",
        ]
        if check_resume:
            extra_args.extend(
                [
                    "--train.max_steps=3",
                    "--data.dataloader.shuffle=False",
                    "--train.checkpoint.save_steps=1",
                ]
            )
        result = run(task_name=optimizer_type, extra_args=extra_args)
        for metric in ("loss", "grad_norm"):
            assert len(result[metric]) == (3 if check_resume else 2)
            assert all(isfinite(value) for value in result[metric])
        if check_resume:
            native = run(
                task_name="torch_adamw",
                parallel_config=ParallelConfig(fsdp_mode="fsdp2"),
                extra_args=[*extra_args, "--train.checkpoint.save_steps=0"],
            )
            for metric in ("loss", "grad_norm"):
                assert len(native[metric]) == len(result[metric])
                assert result[metric] == pytest.approx(native[metric], rel=1e-5, abs=1e-6)
            checkpoint = tmp_path / optimizer_type / "checkpoints" / "global_step_1"
            resumed = run(
                task_name=f"{optimizer_type}_resumed",
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


@pytest.mark.launch_plan(provider=_HYPER_LAUNCH)
def test_hyper_fsdp_training(tmp_path):
    if get_device_type() != "npu" or runner_device_count() < _HYPER_LAUNCH.nproc:
        pytest.skip("Requires two Ascend NPU devices.")
    _run_hyper_training(tmp_path, "adamw", check_resume=True)


@pytest.mark.launch_plan(provider=_HYPER_LAUNCH)
def test_hyper_muon_training(tmp_path):
    if get_device_type() != "npu" or runner_device_count() < _HYPER_LAUNCH.nproc:
        pytest.skip("Requires two Ascend NPU devices.")
    _run_hyper_training(tmp_path, "hyper_muon")
