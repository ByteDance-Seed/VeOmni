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
from veomni.arguments import ModelArguments


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fsdp_mode": "ddp"},
        {"fsdp_mode": "eager"},
        {"offload": True},
        {"mixed_precision": {"enable": False}},
        {"mixed_precision": {"param_dtype": "float16"}},
    ],
)
def test_fp8_all_gather_rejects_unsupported_policies(kwargs):
    from veomni.arguments.arguments_types import FSDPConfig
    from veomni.arguments.parser import _instantiate_recursive

    with pytest.raises(ValueError, match="fp8_all_gather"):
        _instantiate_recursive(FSDPConfig, {"fp8_all_gather": True, **kwargs})


@pytest.mark.parametrize("value", ["true", 1, None])
def test_fp8_all_gather_requires_boolean(value):
    from veomni.arguments.arguments_types import FSDPConfig

    with pytest.raises(ValueError, match="boolean"):
        FSDPConfig(fp8_all_gather=value)


def test_fp8_all_gather_is_opt_in():
    from veomni.arguments.arguments_types import FSDPConfig

    assert FSDPConfig().fp8_all_gather is False
    assert FSDPConfig(fp8_all_gather=True).fp8_all_gather is True


@pytest.mark.parametrize("enabled", [False, True])
def test_fp8_all_gather_reaches_parallel_builder(monkeypatch, enabled):
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed import torch_parallelize

    calls = []
    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda **kw: calls.append(kw) or kw["model"])
    model = nn.Linear(2, 2)
    torch_parallelize.build_parallelize_model(
        model, fp8_all_gather=enabled, expert_fsdp_shard_dim=0, enable_gradient_checkpointing=False
    )
    assert calls[0]["fp8_all_gather"] is enabled
    assert calls[0]["expert_fsdp_shard_dim"] == 0


def test_fp8_all_gather_rejects_compile_before_wrapping(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed import torch_parallelize
    from veomni.distributed.torch_compile import CompileConfig

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: SimpleNamespace(dp_mode="fsdp2"))
    with pytest.raises(ValueError, match="does not support torch.compile"):
        torch_parallelize.build_parallelize_model(
            nn.Linear(2, 2), fp8_all_gather=True, compile_config=CompileConfig(enable=True)
        )


def test_fp8_all_gather_rejects_non_swiglu_before_collectives():
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed.fsdp2.fp8_all_gather import register_deepgemm_fp8_all_gather

    module = nn.Module()
    module.veomni_moe = SimpleNamespace(impl="deepgemm_fp8")
    module.use_swiglu_mlp = False
    with pytest.raises(ValueError, match="SwiGLU"):
        register_deepgemm_fp8_all_gather(module)


@pytest.mark.parametrize("value", [-1, 2, True, False, "0", 0.0])
def test_expert_fsdp_shard_dim_rejects_invalid_values(value):
    from veomni.arguments.arguments_types import FSDPConfig

    with pytest.raises(ValueError, match="expert_fsdp_shard_dim"):
        FSDPConfig(expert_fsdp_shard_dim=value)


@pytest.mark.parametrize("value", [None, 0, 1])
def test_expert_fsdp_shard_dim_valid_values(value):
    from veomni.arguments.arguments_types import FSDPConfig

    assert FSDPConfig(expert_fsdp_shard_dim=value).expert_fsdp_shard_dim == value


def test_expert_fsdp_shard_dim_rejects_ddp():
    from veomni.arguments.arguments_types import FSDPConfig

    with pytest.raises(ValueError, match="expert_fsdp_shard_dim"):
        FSDPConfig(expert_fsdp_shard_dim=0, fsdp_mode="ddp")
