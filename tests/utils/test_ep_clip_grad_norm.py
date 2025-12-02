import math
import subprocess
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard

from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.utils import helper
from veomni.utils.arguments import TrainingArguments, parse_args
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


@dataclass
class Argument:
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


class ToyMoeModel(torch.nn.Module):
    _no_split_modules = ["ToyMoeDecoderLayer"]

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)
        self.decoder = ToyMoeDecoderLayer()

    def set_grad(self):
        for _, p in self.named_parameters(recurse=True):
            p.grad = torch.ones_like(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        return loss

    def init_weights(self):
        self.bias.data.fill_(1.0)
        self.decoder.regular_mlp.data.fill_(1.0)
        self.decoder.ep_layer.data.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {"decoder.ep_layer": Shard(0)}
        parallel_plan = ParallelPlan(
            ep_plan=ep_plan,
        )
        return parallel_plan


class ToyMoeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.ep_layer = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)


def main():
    dist.init_process_group(backend=get_dist_comm_backend())
    args = parse_args(Argument)

    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    model = ToyMoeModel()
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=None,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=[],
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    max_grad_norm = 1.0
    model.set_grad()
    grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)
    # Run the clipper again to measure the norm after the first pass.
    grad_norm_post_clip = veomni_clip_grad_norm(model, max_grad_norm)
    logger.info_rank0(f"grad_norm_pre_clip: {grad_norm_pre_clip}, grad_norm_post_clip: {grad_norm_post_clip}")

    expected_grad_norm_pre_clip = math.sqrt(16 + 64 * 16 + 64 * 16 * 32)
    torch.testing.assert_close(grad_norm_pre_clip, expected_grad_norm_pre_clip, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        grad_norm_post_clip, min(expected_grad_norm_pre_clip, max_grad_norm), atol=1e-6, rtol=1e-6
    )

    expected_clipped_grad = 1 / expected_grad_norm_pre_clip
    for _, p in model.named_parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, expected_clipped_grad), atol=1e-6, rtol=1e-6)

    dist.barrier()
    dist.destroy_process_group()


def test_clip_grad_norm_fsdp2_ep4():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port=4321",
        "tests/utils/test_ep_clip_grad_norm.py",
        "--train.expert_parallel_size=2",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.output_dir='debug'",
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def test_clip_grad_norm_fsdp2_ep8():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port=4321",
        "tests/utils/test_ep_clip_grad_norm.py",
        "--train.expert_parallel_size=4",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.output_dir='debug'",
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
