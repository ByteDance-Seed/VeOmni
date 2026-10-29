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

"""Two-rank native FSDP2/EP expert-gradient and fresh-process DCP probe."""

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_npu import Qwen4ExpTextExperts
from veomni.ops.config import set_ops_config


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = SimpleNamespace(num_experts=4, hidden_size=256, moe_intermediate_size=32, hidden_act="silu")
        self.experts = Qwen4ExpTextExperts(config)

    def forward(self, x, indices, weights):
        return self.experts(x, indices, weights)


class ToyQwen4(nn.Module):
    _no_split_modules = ["ToyDecoder"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen4_exp")
        self.layer = ToyDecoder()

    def forward(self, x, indices, weights):
        return self.layer(x, indices, weights)

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.fill_(0.01 if "gate" in name else 0.02)

    def get_parallel_plan(self):
        plan = ParallelPlan(
            extra_parallel_plan={"ep": {"layer.experts.gate_up_proj": Shard(0), "layer.experts.down_proj": Shard(0)}}
        )
        plan.extra_parallel_fsdp_no_shard_module = {"ep": {"layer.experts"}}
        return plan


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _snapshot(model, optimizer):
    result = {}
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        result[name] = {
            "param": _local(param.detach()).cpu().clone(),
            "m": _local(state["exp_avg"]).detach().cpu().clone(),
            "v": _local(state["exp_avg_sq"]).detach().cpu().clone(),
            "step": _local(state["step"]).detach().cpu().clone(),
        }
        assert all(result[name][part].dtype == torch.float32 for part in ("param", "m", "v"))
    return result


def _step(model, optimizer, rank, ep, step):
    generator = torch.Generator().manual_seed(9900 + rank * 10 + step)
    x = torch.randn(8, 256, generator=generator).to(device="npu", dtype=torch.bfloat16)
    # Each EP rank owns a zero-token expert while experts 0 and 2 receive tokens.
    indices = torch.tensor([[0, 2]] * 8, device="npu", dtype=torch.int64)
    weights = torch.full((8, 2), 0.5, device="npu", dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, indices, weights).float().square().mean()
    loss.backward()
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None and param.grad.dtype == torch.float32
        grads[name] = _local(param.grad).detach().cpu().clone()
        assert torch.isfinite(grads[name]).all()
        assert grads[name].shape[0] == 2
        assert torch.count_nonzero(grads[name][0]).item() > 0
        assert torch.count_nonzero(grads[name][1]).item() == 0
        assert torch.any(grads[name][0] != grads[name][0].bfloat16().float()).item()
    optimizer.step()
    return {
        "input": x.cpu(),
        "indices": indices.cpu(),
        "weights": weights.cpu(),
        "loss": loss.detach().cpu(),
        "grads": grads,
        "state": _snapshot(model, optimizer),
    }


def run(rank, phase, output_dir, ep, endpoint=None):
    import torch_npu  # noqa: F401

    torch.npu.set_device(rank)
    kwargs = {"backend": "hccl", "rank": rank, "world_size": 2, "timeout": timedelta(seconds=180)}
    if endpoint is not None:
        kwargs["store"] = endpoint.connect()
    dist.init_process_group(**kwargs)
    try:
        mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=True)
        _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            device_type="npu",
            name="npu-fp32-experts",
            extra_parallel_sizes=(ep,),
            extra_parallel_names=("ep",),
            extra_parallel_placement_innermost=(False,),
        )
        set_ops_config(SimpleNamespace(moe_implementation="fused_npu"))
        with torch.device("meta"):
            model = ToyQwen4()
        model = build_parallelize_model(
            model,
            mixed_precision=mixed,
            enable_gradient_checkpointing=False,
            init_device="meta",
            enable_forward_prefetch=False,
        )
        assert model.layer.experts._veomni_fp32_expert_wgrad
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        ckpt = str(Path(output_dir) / f"ep{ep}-dcp")
        if phase == "fresh":
            _step(model, optimizer, rank, ep, 1)
            model_state, optim_state = get_state_dict(model, optimizer)
            dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=ckpt)
        else:
            model_state, optim_state = get_state_dict(model, optimizer)
            dcp.load({"model": model_state, "optimizer": optim_state}, checkpoint_id=ckpt)
            set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optim_state)
        before = _snapshot(model, optimizer)
        after = _step(model, optimizer, rank, ep, 2)
        for name in before:
            assert not torch.equal(before[name]["param"], after["state"][name]["param"])
        torch.save({"before": before, "after": after}, Path(output_dir) / f"ep{ep}-{phase}-rank{rank}.pt")
        dist.barrier()
    finally:
        clear_parallel_state()
        dist.destroy_process_group()


if __name__ == "__main__":
    run(
        int(os.environ["RANK"]),
        os.environ["FP32_EXPERT_PHASE"],
        os.environ["FP32_EXPERT_OUTPUT"],
        int(os.environ["FP32_EXPERT_EP"]),
    )
