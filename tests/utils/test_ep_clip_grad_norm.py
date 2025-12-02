import subprocess
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard

from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
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

    def forward(self) -> torch.Tensor:
        return self.regular_mlp.sum() + self.ep_layer.sum()


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

    ep_params = []
    non_ep_params = []
    for n, p in model.named_parameters():
        entry = f"{n} (shape={tuple(p.shape)}, type={type(p)}, is_dtensor={isinstance(p, DTensor)})"
        if "ep_layer" in n:
            ep_params.append(p)
            logger.info_rank0(f"Register EP param: {entry}")
        else:
            non_ep_params.append(p)
            logger.info_rank0(f"Register non-EP param: {entry}")
    model._ep_param_groups = {"ep": ep_params, "non_ep": non_ep_params}
    logger.info_rank0(
        "group sizes - fsdp: %s, ep: %s, ep_fsdp: %s",
        dist.get_world_size(group=fsdp_group) if fsdp_group is not None else None,
        dist.get_world_size(group=ep_group) if ep_group is not None else None,
        dist.get_world_size(group=ep_fsdp_group) if ep_fsdp_group is not None else None,
    )
    tensor_device = torch.device(f"{get_device_type()}:{get_device_id()}")
    max_grad_norm = args.train.max_grad_norm

    def compute_global_grad_norm() -> float:
        """Reconstruct the optimizer-facing L2 norm from the current gradient shards."""
        # local_sq accumulates the sum of squares for this rank only; we keep it on the same device
        # as the gradients to avoid device synchronization.
        local_sq = None
        for p in model.parameters():
            g = p.grad
            if g is None:
                continue
            # FSDP2 + EP stores gradients as DTensors, so convert them to the local view before
            # measuring. Non-DTensor grads (e.g., bias) are already local tensors.
            if isinstance(g, DTensor):
                g_local = g.to_local()
            else:
                g_local = g
            # contrib is the squared L2 contribution of this parameter shard on this rank.
            contrib = g_local.detach().float().pow(2).sum()
            if local_sq is None:
                # Initialize the accumulator lazily so we only allocate once we know the device.
                local_sq = torch.zeros(1, device=g_local.device, dtype=torch.float32)
            # Accumulate the per-parameter contribution into this rank's running total.
            local_sq = local_sq + contrib
        if local_sq is None:
            # If every parameter had `grad=None`, fall back to a zero tensor on the current device.
            local_sq = torch.zeros(1, device=torch.device(get_device_type()), dtype=torch.float32)
        # Combine the per-rank totals so every rank sees the global sum of squares.
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        # Final L2 norm is the square root of the summed squares.
        return torch.sqrt(local_sq).item()

    grad_norm_pre_clip = None
    grad_norm_post_clip = None
    for step in range(3):
        inputs = torch.ones(1, 16, device=tensor_device)
        loss = model(inputs)
        loss.backward()

        manual_pre = compute_global_grad_norm()
        grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)
        # Clip API reports the norm before scaling, so compare against the pre-clip measurement.
        torch.testing.assert_close(grad_norm_pre_clip, manual_pre, atol=1e-6, rtol=1e-6)

        manual_post = compute_global_grad_norm()
        grad_norm_post_clip = veomni_clip_grad_norm(model, max_grad_norm)
        torch.testing.assert_close(grad_norm_post_clip, manual_post, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad_norm_post_clip, min(manual_pre, max_grad_norm), atol=1e-6, rtol=1e-6)

        # The clipper scales every gradient by the same coefficient max_norm/manual_pre (capped at 1).
        clip_coeff = min(max_grad_norm / manual_pre, 1.0)
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
            torch.testing.assert_close(
                grad_local,
                torch.full_like(grad_local, clip_coeff),
                atol=1e-6,
                rtol=1e-6,
                msg=f"Gradient mismatch for {name}",
            )

        logger.info_rank0(
            f"step: {step}, loss: {loss.item()}, grad_norm_pre_clip: {grad_norm_pre_clip}, "
            f"grad_norm_post_clip: {grad_norm_post_clip}"
        )
        model.zero_grad()

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
