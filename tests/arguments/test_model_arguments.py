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

"""Every unit answers ``config_path``, so the loader never asks what it holds."""

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import AcceleratorConfig, FSDPConfig, ModelArguments, OptimizerConfig


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class TestWhereTheLoaderReadsTheConfigFrom:
    def test_a_module_falls_back_to_its_weights_folder(self):
        # A module inside a composed checkpoint is addressed by its own subfolder,
        # so it never configures a config path separately.
        assert ModelArguments(model_path="somewhere").config_path == "somewhere"

    def test_a_whole_model_honours_a_separate_config_path(self):
        # Toy-config runs rely on this: architecture from a local json, weights
        # (and tokenizer) from somewhere else entirely.
        assert ModelArguments(model_path="weights", config_path="cfg").config_path == "cfg"

    def test_a_whole_model_falls_back_to_its_weights_path(self):
        assert ModelArguments(model_path="weights").config_path == "weights"

    def test_the_tokenizer_follows_the_config_the_base_settled(self):
        assert ModelArguments(model_path="weights", config_path="cfg").tokenizer_path == "cfg"


def test_swap_fraction_is_coerced_and_ranged():
    # YAML resolves an exponent literal without a decimal point to a string; it
    # must be coerced before the range check instead of raising TypeError.
    assert OptimizerConfig(type="adamw_swap", swap_mem_fraction_static="8e-1").swap_mem_fraction_static == 0.8
    with pytest.raises(ValueError, match="swap_mem_fraction_static"):
        OptimizerConfig(type="adamw_swap", swap_mem_fraction_static=1.5)


def test_swap_rejects_fsdp_cpu_offload():
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelArguments(
            model_path="weights",
            optimizer=OptimizerConfig(type="adamw_swap"),
            accelerator=AcceleratorConfig(fsdp_config=FSDPConfig(offload=True)),
        )
