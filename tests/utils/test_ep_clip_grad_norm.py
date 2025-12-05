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
from veomni.utils.device import get_device_id, get_device_type, get_dist_comm_backend, get_torch_device, IS_NPU_AVAILABLE


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

    def check_model_param_grad_one_by_one(expected_grad, msg):
        # check them one-by-one
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_local = grad.to_local() if isinstance(grad, DTensor) else grad
            logger.info_rank0(f"{msg}: the local grad for {name}: {grad_local}")
            torch.testing.assert_close(
                grad_local,
                torch.full_like(grad_local, expected_grad),
                atol=1e-6,
                rtol=1e-6,
                msg=f"Gradient mismatch for {name}, which has local shape {grad_local.shape}",
            )

    total_grad_norm_pre_clip = None
    for step in range(3):
        inputs = torch.ones(1, 16, device=tensor_device)
        loss = model(inputs)
        loss.backward()

        logger.info_rank0("manually checking the initial param grads before any clipping")
        # On GPU, the local gard of each param after local backward is 1.0
        # At loss.backward(), reduce scatter is triggered to **average** grad for the same param against different data input on each fsdp rank
        # By default, this is achieved by dividing sum of param grad on each rank by fsdp size
        # * For example, for pure FSDP2 on 8 GPUs,
        #   the local grad of each param after backward is  1.0 x 8 (every rank every param local grad is 1.0) / 8 (fsdp size)
        # * When ep is enabled, the divide factor for ep params should be ep_fsdp size (which is fsdp size / ep size)
        #   * by applying set_gradient_divide_factor(ep_fsdp_size) for EP modules in torch_parallelize
        # * In general, the divide factor for each param should be its num of different input data, which is its dp size
        #   EP params has different num of input data from fsdp-only params
        # On NPU, we are missing PreSumMul ReduceOp for set_gradient_divide_factor, so the expected param grad here should have not been divided by ep_fsdp_size yet
        if IS_NPU_AVAILABLE and ps.ep_enabled:
            expected = float(ps.ep_fsdp_size)
        else:
            expected = 1.0
        check_model_param_grad_one_by_one(expected_grad=expected, msg="Before clipping")

        # Every local param grad is 1.0, model total norm should be sqrt(1^2 * total_param_num) which is sqrt(total_param_num)
        expected_total_grad_norm = math.sqrt(16 + 64 * 16 + 64 * 16 * 32)
        total_grad_norm_pre_clip = veomni_clip_grad_norm(model, max_grad_norm)
        # check whether total grad norm meets our expectation
        if device_type != "npu":
            torch.testing.assert_close(
                total_grad_norm_pre_clip, expected=expected_total_grad_norm, atol=1e-6, rtol=1e-6
            )
        else:
            logger.info_rank0("checking npu expected total grad norm")
            # on npu we manually divide grads by ep_fsdp size in place before reduce scatter
            npu_expected_total_grad_norm = (
                math.sqrt(16 + 64 * 16 + (64 * 16 * 32) / ps.ep_fsdp_size)
                if ps.ep_enabled
                else expected_total_grad_norm
            )
            torch.testing.assert_close(
                total_grad_norm_pre_clip, expected=npu_expected_total_grad_norm, atol=1e-6, rtol=1e-6
            )

        # go through each param grad one-by-one after clipping to check whether their value meets our expectation
        clip_coeff = min(max_grad_norm / expected_total_grad_norm, 1.0)
        logger.info_rank0("Checking model param grad one-by-one after clipping")
        check_model_param_grad_one_by_one(clip_coeff, msg="After clipping")

        logger.info_rank0(
            f"step: {step}, loss: {loss.item()}, grad_norm_pre_clip: {total_grad_norm_pre_clip}, "
        )
        model.zero_grad()

    dist.barrier()
    dist.destroy_process_group()


def test_clip_grad_norm_fsdp2_no_ep():
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=8",
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
        "--nproc_per_node=8",
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
        "--nproc_per_node=8",
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
