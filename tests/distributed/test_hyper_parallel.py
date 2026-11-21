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

"""Two-NPU model training smoke tests for HyperParallel."""

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools import DummyDataset, ParallelConfig, materialize_weights, run_training_config
from veomni.utils.device import get_device_type


TOY_CONFIG = "./tests/toy_config/qwen3_toy"
_HYPER_LAUNCH = LaunchSpec(nproc=2)


def _run_hyper_training(tmp_path, optimizer_type):
    model_path = tmp_path / "model"
    materialize_weights(TOY_CONFIG, str(model_path))
    dataset = DummyDataset(
        num_samples=16,
        seq_len=64,
        dataset_type="text",
        cache_name=f"hyper_{optimizer_type}_{tmp_path.name}",
    )
    try:
        result = run_training_config(
            script="tests/train_scripts/train_text_test.py",
            config_path=TOY_CONFIG,
            model_path=str(model_path),
            train_path=dataset.save_path,
            output_dir=str(tmp_path),
            task_name=optimizer_type,
            parallel_config=ParallelConfig(fsdp_mode="hyper_fsdp2"),
            nproc=_HYPER_LAUNCH.nproc,
            model_name="qwen3",
            extra_args=[
                f"--model.optimizer.type={optimizer_type}",
                "--model.optimizer.muon_ns_implementation=std",
                "--data.max_seq_len=64",
            ],
        )
        assert result["loss"]
    finally:
        dataset.clean_cache()


@pytest.mark.launch_plan(provider=_HYPER_LAUNCH)
def test_hyper_fsdp_training(tmp_path):
    if get_device_type() != "npu" or runner_device_count() < _HYPER_LAUNCH.nproc:
        pytest.skip("Requires two Ascend NPU devices.")
    _run_hyper_training(tmp_path, "adamw")


@pytest.mark.launch_plan(provider=_HYPER_LAUNCH)
def test_hyper_muon_training(tmp_path):
    if get_device_type() != "npu" or runner_device_count() < _HYPER_LAUNCH.nproc:
        pytest.skip("Requires two Ascend NPU devices.")
    _run_hyper_training(tmp_path, "hyper_muon")
