# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 transport follows the bound rows and validates every shard before installation."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_llama_config
from veomni.distributed import torch_parallelize
from veomni.distributed.fsdp2.mxfp8_all_gather import has_mxfp8_all_gather_targets, register_mxfp8_all_gather
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.ops import OpsSelection, VeomniOp


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def mesh(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("CPU FSDP test requires Gloo.")
    dist.init_process_group("gloo", init_method=(tmp_path / "rdzv").as_uri(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_installs_only_the_resolved_llama_projections(mesh):
    pytest.importorskip("torchao")
    from veomni.models.transformers.llama.modeling_llama import LlamaDecoderLayer

    config = tiny_llama_config()
    ops = eager_ops_config()
    mxfp8 = {"name": "torchao", "kwargs": {"dtype": "mxfp8", "emulated": True}}
    ops.linear_implementation = {"name": "eager", "overrides": [{"site": r"layers\.0\.k_proj", **mxfp8}]}
    ops.gated_mlp_implementation = {"name": "eager", "overrides": [{"site": r"layers\.1\.mlp", **mxfp8}]}
    config._set_ops_implementation(ops)
    model = nn.Sequential(*(LlamaDecoderLayer(config, idx) for idx in range(2)))
    assert has_mxfp8_all_gather_targets(model)
    assert not has_mxfp8_all_gather_targets(model[0].mlp)
    fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

    assert register_mxfp8_all_gather(model) == 4
    assert {name for name, p in model.named_parameters() if hasattr(p._local_tensor, "fsdp_pre_all_gather")} == {
        "0.self_attn.k_proj.weight",
        "1.mlp.gate_proj.weight",
        "1.mlp.up_proj.weight",
        "1.mlp.down_proj.weight",
    }


def _linear_target(name, *, emulated=True):
    owner = nn.Module()
    setattr(owner, name, nn.Linear(64, 64, bias=False))
    row = OpsSelection(name="torchao", kwargs={"dtype": "mxfp8", "emulated": emulated})
    setattr(owner, f"veomni_{name}", VeomniOp("linear", "standard", row))
    return owner


@pytest.mark.parametrize("shared", ["same_row", "native_row", "unselected"])
def test_shared_weight_consumers(mesh, shared):
    pytest.importorskip("torchao")
    # DeepSeek V4's indexer_q_b_proj site consumes its owner's q_b_proj.
    model = nn.Sequential(_linear_target("q_b_proj"), _linear_target("kv_proj", emulated=shared != "native_row"))
    model[1].kv_proj.weight = model[0].q_b_proj.weight
    if shared == "unselected":
        model[1].veomni_kv_proj = VeomniOp("linear", "standard", "eager")
    fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

    if shared == "same_row":
        assert register_mxfp8_all_gather(model) == 1
    else:
        with pytest.raises(ValueError, match="conflicting emulated modes|also used by"):
            register_mxfp8_all_gather(model)
        assert not hasattr(model[0].q_b_proj.weight._local_tensor, "fsdp_pre_all_gather")


@pytest.mark.parametrize("both_selected, error", [(False, "also used by"), (True, "same FSDP unit")])
def test_cross_block_shared_projection_weights(mesh, monkeypatch, both_selected, error):
    pytest.importorskip("torchao")
    from veomni.models.transformers.llama.modeling_llama import LlamaForCausalLM

    config = tiny_llama_config()
    ops = eager_ops_config()
    ops.linear_implementation = {
        "name": "eager",
        "overrides": [
            {
                "site": r"layers\.\d+\.q_proj" if both_selected else r"layers\.0\.q_proj",
                "name": "torchao",
                "kwargs": {"dtype": "mxfp8", "emulated": True},
            }
        ],
    }
    config._set_ops_implementation(ops)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    first, second = model.model.layers
    second.self_attn.q_proj.weight = first.self_attn.q_proj.weight
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")
    _init_parallel_state(device_type="cpu")
    try:
        with pytest.raises(ValueError, match=error):
            torch_parallelize.parallelize_model_fsdp2(model, init_device="meta", mxfp8_all_gather=True)
    finally:
        clear_parallel_state()


def test_invalid_shard_leaves_all_hooks_uninstalled(mesh):
    pytest.importorskip("torchao")
    model = nn.Sequential(_linear_target("q_proj"), _linear_target("k_proj"))
    model[1].k_proj = nn.Linear(64, 48, bias=False)
    fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"32-element blocks: 1\.k_proj\.weight"):
        register_mxfp8_all_gather(model)
    assert all(not hasattr(p._local_tensor, "fsdp_pre_all_gather") for p in model.parameters())
