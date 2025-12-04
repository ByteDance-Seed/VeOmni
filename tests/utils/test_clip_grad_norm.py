import subprocess
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.utils import helper
from veomni.utils.arguments import TrainingArguments, parse_args
from veomni.utils.device import get_device_id, get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


@dataclass
class Argument:
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        return loss

    def init_weights(self):
        self.bias.data.fill_(1.0)


def main():
    dist.init_process_group(backend=get_dist_comm_backend())
    args = parse_args(Argument)

    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    device = get_device_id()
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

    model = Model()
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

    max_grad_norm = args.train.max_grad_norm
    for step in range(10):
        inputs = torch.randn(1, 16).to(device)
        loss = model(inputs)
        loss.backward()
        grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)
        # Run the clipper again to measure the norm after the first pass.
        grad_norm_post_clip = veomni_clip_grad_norm(model, max_grad_norm)
        logger.info_rank0(
            f"step: {step}, loss: {loss.item()}, grad_norm_pre_clip: {grad_norm_pre_clip}, "
            f"grad_norm_post_clip: {grad_norm_post_clip}"
        )
        model.zero_grad()

    torch.testing.assert_close(grad_norm_pre_clip, 4.0, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_norm_post_clip, min(4.0, max_grad_norm), atol=1e-6, rtol=1e-6)
    clip_coeff = min(max_grad_norm / 4.0, 1.0)
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
        logger.info_rank0(f"After clipping, the local grad for {name}: {grad_local}")
        expected = clip_coeff
        torch.testing.assert_close(
            grad_local,
            torch.full_like(grad_local, expected),
            atol=1e-6,
            rtol=1e-6,
            msg=f"Gradient mismatch for {name}",
        )

    dist.barrier()
    dist.destroy_process_group()


def test_clip_grad_norm_fsdp2():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
        "--master_port=4321",
        "tests/utils/test_clip_grad_norm.py",
        "--train.data_parallel_shard_size=8",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.output_dir='debug'",
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
