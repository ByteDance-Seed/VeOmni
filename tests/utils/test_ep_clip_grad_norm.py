import math
import subprocess
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard

from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import TrainingArguments, parse_args
from veomni.utils.device import get_device_id, get_device_type, get_dist_comm_backend, get_torch_device


# from veomni.optim.optimizer import build_optimizer

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        loss = loss + self.decoder()
        return loss

    def init_weights(self):
        self.bias.data.fill_(1.0)
        self.decoder.regular_mlp.data.fill_(1.0)
        self.decoder.moe.experts.data.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {"decoder.moe.experts": Shard(0)}
        parallel_plan = ParallelPlan(
            ep_plan=ep_plan,
        )
        return parallel_plan


class ToyMoeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.moe = ToyMoeExperts()

    def forward(self) -> torch.Tensor:
        return self.regular_mlp.sum() + self.moe()


class ToyMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum()


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
    # building a optimizer is necessary to register ep/non-ep params group
    # _ = build_optimizer(model, fused=True)

    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
    ep_group = ps.ep_group if ps.ep_enabled else None
    ep_fsdp_group = None
    if ps.ep_enabled and ps.ep_fsdp_device_mesh is not None:
        ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group()
    # build optimizer to register ep param groups when ep is enabled
    _ = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
    )
    logger.info_rank0(
        "group sizes - fsdp: %s, ep: %s, ep_fsdp: %s",
        dist.get_world_size(group=fsdp_group) if fsdp_group is not None else None,
        dist.get_world_size(group=ep_group) if ep_group is not None else None,
        dist.get_world_size(group=ep_fsdp_group) if ep_fsdp_group is not None else None,
    )
    device_type = get_device_type()
    tensor_device = torch.device(f"{device_type}:{get_device_id()}")
    max_grad_norm = args.train.max_grad_norm

    # this is wrong for ep because model is not aware of ep
    # for ep=4, model's ep layer here becomes [16, 16, 32] instead of [64, 16, 32]
    def compute_global_grad_norm_pure_fsdp2(apply_ep_scale: bool = False) -> float:
        """Reconstruct the optimizer-facing L2 norm from the current gradient shards."""
        # local_sq accumulates the sum of squares for this rank only; we keep it on the same device
        # as the gradients to avoid device synchronization.
        local_sq = None
        for name, p in model.named_parameters():
            g = p.grad
            if g is None:
                continue
            # FSDP2 + EP stores gradients as DTensors, so convert them to the local view before
            # measuring. Non-DTensor grads (e.g., bias) are already local tensors.
            if isinstance(g, DTensor):
                g_local = g.to_local()
            else:
                g_local = g
            g_fp32 = g_local.detach().float()
            logger.info_rank0(f"param name {name} has grad {g_fp32}")
            # contrib is the squared L2 contribution of this parameter shard on this rank.
            contrib = g_fp32.pow(2).sum()
            if local_sq is None:
                # Initialize the accumulator lazily so we only allocate once we know the device.
                local_sq = torch.zeros(1, device=g_fp32.device, dtype=torch.float32)
            # Accumulate the per-parameter contribution into this rank's running total.
            local_sq = local_sq + contrib
        if local_sq is None:
            # If every parameter had `grad=None`, fall back to a zero tensor on the current device.
            local_sq = torch.zeros(1, device=torch.device(device_type), dtype=torch.float32)
        # Combine the per-rank totals so every rank sees the global sum of squares.
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        # Final L2 norm is the square root of the summed squares.
        return torch.sqrt(local_sq).item()

    total_grad_norm_pre_clip = None
    grad_norm_post_clip = None
    for step in range(3):
        inputs = torch.ones(1, 16, device=tensor_device)
        loss = model(inputs)
        loss.backward()

        logger.info_rank0("manually checking the initial param grads before any clipping")
        # check them one-by-one
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
            logger.info_rank0(f"Before clipping, the local grad for {name}: {grad_local}")
            expected = 1.0
            torch.testing.assert_close(
                grad_local,
                torch.full_like(grad_local, expected),
                atol=1e-6,
                rtol=1e-6,
                msg=f"Gradient mismatch for {name}, which has local shape {grad_local.shape}",
            )
        expected_total_grad_norm = math.sqrt(16 + 64 * 16 + 64 * 16 * 32)
        total_grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)
        # check whether total grad norm meets our expectation
        torch.testing.assert_close(total_grad_norm_pre_clip, expected=expected_total_grad_norm, atol=1e-6, rtol=1e-6)

        # go through each param grad one-by-one after clipping to check whether their value meets our expectation
        clip_coeff = min(max_grad_norm / expected_total_grad_norm, 1.0)
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

        logger.info_rank0(
            f"step: {step}, loss: {loss.item()}, grad_norm_pre_clip: {total_grad_norm_pre_clip}, "
            f"grad_norm_post_clip: {grad_norm_post_clip}"
        )
        model.zero_grad()

    dist.barrier()
    dist.destroy_process_group()


def test_clip_grad_norm_fsdp2_no_ep():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port=4321",
        "tests/utils/test_ep_clip_grad_norm.py",
        "--train.expert_parallel_size=1",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.output_dir='debug'",
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


def test_clip_grad_norm_fsdp2_ep4():
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


def test_clip_grad_norm_fsdp2_ep8():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--master_port=4321",
        "tests/utils/test_ep_clip_grad_norm.py",
        "--train.expert_parallel_size=8",
        "--train.data_parallel_mode=fsdp2",
        "--train.init_device=meta",
        "--train.output_dir='debug'",
    ]
    result = subprocess.run(command, check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
