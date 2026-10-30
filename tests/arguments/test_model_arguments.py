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
@pytest.mark.parametrize("has_experts", [False, True])
def test_fp8_all_gather_validates_policies_only_for_matching_experts(monkeypatch, kwargs, has_experts):
    from types import SimpleNamespace

    from torch import nn

    from veomni.arguments.arguments_types import FSDPConfig
    from veomni.arguments.parser import _instantiate_recursive
    from veomni.distributed import torch_parallelize

    config = _instantiate_recursive(FSDPConfig, {"fp8_all_gather": True, **kwargs})
    assert config.fp8_all_gather is True
    model = nn.Sequential(nn.Linear(2, 2))
    if has_experts:
        model[0].veomni_moe = SimpleNamespace(impl="deepgemm_fp8")
    calls = []
    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=config.fsdp_mode != "eager", tp_enabled=False, dp_mode=config.fsdp_mode),
    )
    for builder in ("parallelize_model_fsdp2", "parallelize_model_ddp"):
        monkeypatch.setattr(torch_parallelize, builder, lambda **kw: calls.append(kw) or kw["model"])
    options = dict(
        fp8_all_gather=True,
        mixed_precision=config.mixed_precision,
        enable_fsdp_offload=config.offload,
        init_device="cuda",
        enable_gradient_checkpointing=False,
    )
    if has_experts:
        with pytest.raises(ValueError, match="fp8_all_gather requires"):
            torch_parallelize.build_parallelize_model(model, **options)
        assert not calls
    else:
        assert torch_parallelize.build_parallelize_model(model, **options) is model
        if config.fsdp_mode == "fsdp2":
            assert calls[0]["fp8_all_gather"] is False
            assert calls[0]["mixed_precision"] is config.mixed_precision
            assert calls[0]["enable_fsdp_offload"] is config.offload


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
@pytest.mark.parametrize("impl", [None, "fused_triton", "deepgemm_fp8"])
def test_fp8_all_gather_reaches_parallel_builder(monkeypatch, enabled, impl):
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
    model = nn.Sequential(nn.Linear(2, 2))
    if impl is not None:
        model[0].veomni_moe = SimpleNamespace(impl=impl)
    torch_parallelize.build_parallelize_model(
        model,
        fp8_all_gather=enabled,
        expert_fsdp_shard_dim=0,
        enable_gradient_checkpointing=False,
    )
    assert calls[0]["fp8_all_gather"] is (enabled and impl == "deepgemm_fp8")
    assert calls[0]["expert_fsdp_shard_dim"] == 0


@pytest.mark.parametrize("has_experts", [False, True])
def test_fp8_all_gather_rejects_compile_only_for_matching_experts(monkeypatch, has_experts):
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed import torch_parallelize
    from veomni.distributed.torch_compile import CompileConfig

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_mode="fsdp2", fsdp_enabled=True, tp_enabled=False),
    )
    calls = []
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda **kw: calls.append(kw) or kw["model"])
    model = nn.Linear(2, 2)
    if has_experts:
        model.veomni_moe = SimpleNamespace(impl="deepgemm_fp8")
    options = dict(fp8_all_gather=True, compile_config=CompileConfig(enable=True))
    if has_experts:
        with pytest.raises(ValueError, match="does not support torch.compile"):
            torch_parallelize.build_parallelize_model(model, **options)
        assert not calls
    else:
        assert torch_parallelize.build_parallelize_model(model, **options) is model
        assert calls[0]["fp8_all_gather"] is False
        assert calls[0]["compile_config"].enable


@pytest.mark.parametrize("builder", ["build_parallelize_model", "parallelize_model_fsdp2"])
@pytest.mark.parametrize("value", ["true", 1, None, 0])
def test_fp8_all_gather_builders_reject_non_boolean_without_experts(monkeypatch, builder, value):
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed import torch_parallelize

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: SimpleNamespace(dp_mode="fsdp2"))
    with pytest.raises(ValueError, match="boolean"):
        getattr(torch_parallelize, builder)(nn.Linear(2, 2), fp8_all_gather=value)


def test_fp8_all_gather_rejects_non_swiglu_before_collectives():
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed.fsdp2.fp8_all_gather import register_deepgemm_fp8_all_gather

    module = nn.Module()
    module.veomni_moe = SimpleNamespace(impl="deepgemm_fp8")
    module.use_swiglu_mlp = False
    with pytest.raises(ValueError, match="SwiGLU"):
        register_deepgemm_fp8_all_gather(module)


def test_fp8_all_gather_allows_empty_selection():
    from torch import nn

    from veomni.distributed.fsdp2.fp8_all_gather import register_deepgemm_fp8_all_gather

    module = nn.Linear(2, 2)
    assert register_deepgemm_fp8_all_gather(module) == 0


def test_fp8_all_gather_still_validates_selected_experts():
    from types import SimpleNamespace

    from torch import nn

    from veomni.distributed.fsdp2.fp8_all_gather import register_deepgemm_fp8_all_gather

    module = nn.Linear(2, 2)
    module.veomni_moe = SimpleNamespace(impl="deepgemm_fp8")
    with pytest.raises(ValueError, match="two merged expert weights"):
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
