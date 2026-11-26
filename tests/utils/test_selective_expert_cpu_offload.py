import argparse
import gc
import os
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import ModelArguments
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.distributed.parallel_state import (
    _init_parallel_state,
    clear_parallel_state,
    use_parallel_state,
)
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim import build_optimizer
from veomni.utils import helper
from veomni.utils.device import (
    get_device_id,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)


_CI_LAUNCH_2 = LaunchSpec(nproc=2)
logger = helper.create_logger(__name__)


class ToyMoeAndEmbedModel(torch.nn.Module):
    """
    This toy model with MoE+Embedding module has all param value set to 1
    and all its submodules' forward only returns the sum of all its param
    so whatever the input is, the grad of each param is always 1 after its local backward
    As a result, the MoE forward in this model does not have all2all,
    so EP param grad accumulation across ranks is not real,
    where it only accumulates the ep_fsdp ranks, missing accumulation between ep ranks
    """

    _no_split_modules = ["ToyMoeAndEmbedDecoderLayer", "ToyEmbed"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.bias = torch.nn.Parameter(torch.ones(16), requires_grad=True)
        self.decoder = ToyMoeAndEmbedDecoderLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss = (x + self.bias).sum()
        loss = loss + self.embed_tokens() + self.decoder()
        return loss

    def init_weights(self):
        self.embed_tokens.weight.data.fill_(1.0)
        self.decoder.embed_tokens.weight.data.fill_(1.0)
        self.bias.data.fill_(1.0)
        self.decoder.regular_mlp.data.fill_(1.0)
        self.decoder.moe.experts.data.fill_(1.0)

    def get_parallel_plan(self):
        ep_plan = {"decoder.moe.experts": Shard(0)}
        emb_plan = {"embed_tokens.weight": Shard(0), "decoder.embed_tokens.weight": Shard(0)}
        parallel_plan = ParallelPlan(
            extra_parallel_plan={
                "ep": ep_plan,
                "emb": emb_plan,
            }
        )
        parallel_plan.extra_parallel_fsdp_no_shard_module = {
            "ep": {"decoder.moe"},
            "emb": {"embed_tokens", "decoder.embed_tokens"},
        }

        return parallel_plan


class ToyEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.weight.sum()


class ToyMoeAndEmbedDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = ToyEmbed()
        self.regular_mlp = torch.nn.Parameter(torch.ones(64, 16), requires_grad=True)
        self.moe = ToyMoeExperts()

    def forward(self) -> torch.Tensor:
        return self.embed_tokens() + self.regular_mlp.sum() + self.moe()


class ToyMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.ones(64, 16, 32), requires_grad=True)

    def forward(self) -> torch.Tensor:
        return self.experts.sum()


class SelectiveOffloadModel(ToyMoeAndEmbedModel):
    """Two expert groups let one remain resident while the other is offloaded."""

    def __init__(self):
        super().__init__()
        self.decoder2 = ToyMoeAndEmbedDecoderLayer()

    def forward(self, x):
        return super().forward(x) + self.decoder2()

    def init_weights(self):
        super().init_weights()
        for parameter in self.decoder2.parameters():
            parameter.data.fill_(1.0)

    def get_parallel_plan(self):
        plan = super().get_parallel_plan()
        plan.extra_parallel_plan["ep"]["decoder2.moe.experts"] = Shard(0)
        plan.extra_parallel_plan["emb"]["decoder2.embed_tokens.weight"] = Shard(0)
        plan.extra_parallel_fsdp_no_shard_module["ep"].add("decoder2.moe")
        plan.extra_parallel_fsdp_no_shard_module["emb"].add("decoder2.embed_tokens")
        return plan


def _run_selective_expert_offload(fused, checkpoint_dir):
    from torch.utils._pytree import tree_map

    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.optim.lr_scheduler import build_lr_scheduler

    world = dist.get_world_size()
    ps = _init_parallel_state(
        dp_size=world,
        dp_shard_size=world,
        extra_parallel_names=("ep", "emb"),
        extra_parallel_sizes=(world, 1),
        extra_parallel_placement_innermost=(False, False),
        dp_mode="fsdp2",
        name="selective_expert_offload",
    )
    device = torch.device(get_device_type(), get_device_id())
    config = ModelArguments(config_path="test").accelerator.fsdp_config

    def build(selected, resident):
        return build_parallelize_model(
            SelectiveOffloadModel(),
            init_device="meta",
            weights_path=None,
            basic_modules=[],
            mixed_precision=config.mixed_precision,
            enable_gradient_checkpointing=False,
            enable_fsdp_offload=False,
            fsdp_offload_pin_memory=False,
            cpu_offload_extra_parallel_names=["ep"] if selected else [],
            cpu_offload_exclude_module_patterns=resident,
            broadcast_model_weights_from_rank0=False,
        )

    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    def optimizer_snapshot(optimizer):
        return tree_map(
            lambda tensor: local(tensor).detach().cpu().clone() if torch.is_tensor(tensor) else tensor,
            optimizer.state_dict(),
        )

    def optimizer_lrs(optimizer):
        return {name: [group["lr"] for group in leaf.param_groups] for name, leaf in optimizer.optimizers_dict.items()}

    with use_parallel_state(ps):
        for resident in ([], ["decoder.moe"]):
            reference, candidate = build(False, []), build(True, resident)
            assert candidate._cpu_offload_resident_module_fqns == tuple(resident)
            for name, parameter in candidate.named_parameters():
                offloaded = (
                    name in ("decoder.moe.experts", "decoder2.moe.experts")
                    and name.removesuffix(".experts") not in resident
                )
                assert local(parameter).device.type == ("cpu" if offloaded else device.type)
            # Register the real EP/non-EP ownership buckets used by clipping.
            optimizers = [
                build_optimizer(model, lr=0.125, weight_decay=0.0, fused=fused, optimizer_type="adamw")
                for model in (reference, candidate)
            ]
            schedulers = [
                build_lr_scheduler(optimizer, train_steps=20, lr=0.125, lr_decay_style="linear")
                for optimizer in optimizers
            ]
            for _step in range(2):
                for scale in (1.0, 2.0, 0.5):
                    x = torch.ones(16, device=device)
                    left, right = reference(x) * scale, candidate(x) * scale
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-12)
                    left.backward()
                    right.backward()
                get_torch_device().synchronize()
                for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
                    torch.testing.assert_close(local(left.grad).cpu(), local(right.grad).cpu(), rtol=1e-5, atol=1e-12)
                before_clip = [local(parameter.grad).detach().cpu().clone() for parameter in candidate.parameters()]
                norms = [veomni_clip_grad_norm(model, 0.5) for model in (reference, candidate)]
                torch.testing.assert_close(norms[0], norms[1], rtol=1e-6, atol=1e-6)
                assert norms[1].item() > 0.5
                coefficient = torch.clamp(0.5 / (norms[1].detach().cpu() + 1e-6), max=1.0)
                for left, right, before in zip(
                    reference.parameters(), candidate.parameters(), before_clip, strict=True
                ):
                    clipped = local(right.grad).cpu()
                    torch.testing.assert_close(local(left.grad).cpu(), clipped, rtol=1e-5, atol=1e-12)
                    torch.testing.assert_close(clipped, before * coefficient, rtol=1e-5, atol=1e-12)
                for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
                    torch.testing.assert_close(local(left).cpu(), local(right).cpu(), rtol=1e-6, atol=1e-7)
            checkpoint = str(Path(checkpoint_dir) / f"resident-{len(resident)}")
            assert any(lr < 0.125 for rates in optimizer_lrs(optimizers[1]).values() for lr in rates)
            for leaf in optimizers[1].optimizers_dict.values():
                for parameter, state in leaf.state.items():
                    if local(parameter).device.type == "cpu":
                        for value in (parameter, state["exp_avg"], state["exp_avg_sq"]):
                            assert local(value).device.type == "cpu"
                            assert value.dtype == torch.float32
            saved = {
                "model": candidate,
                "optimizer": optimizers[1],
                "extra_state": {"lr_scheduler": schedulers[1].state_dict()},
                "_veomni_scheduler_object": schedulers[1],
            }
            DistributedCheckpointer.save(checkpoint, saved, parallel_state=ps)
            resumed = build(True, resident)
            resumed_optimizer = build_optimizer(
                resumed, lr=0.125, weight_decay=0.0, fused=fused, optimizer_type="adamw"
            )
            resumed_scheduler = build_lr_scheduler(
                resumed_optimizer, train_steps=20, lr=0.125, lr_decay_style="linear"
            )
            restored = {
                "model": resumed,
                "optimizer": resumed_optimizer,
                "extra_state": {},
                "_veomni_scheduler_object": resumed_scheduler,
            }
            DistributedCheckpointer.load(checkpoint, restored, parallel_state=ps)
            resumed_scheduler.load_state_dict(restored["extra_state"]["lr_scheduler"])

            for left, right in zip(candidate.parameters(), resumed.parameters(), strict=True):
                assert local(left).device == local(right).device
                torch.testing.assert_close(local(left).cpu(), local(right).cpu(), rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer_snapshot(optimizers[1]), optimizer_snapshot(resumed_optimizer), rtol=0, atol=0
            )
            assert schedulers[1].state_dict() == resumed_scheduler.state_dict()
            assert optimizer_lrs(optimizers[1]) == optimizer_lrs(resumed_optimizer)
            for model, optimizer, scheduler in (
                (candidate, optimizers[1], schedulers[1]),
                (resumed, resumed_optimizer, resumed_scheduler),
            ):
                model(torch.ones(16, device=device)).backward()
                veomni_clip_grad_norm(model, 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            for left, right in zip(candidate.parameters(), resumed.parameters(), strict=True):
                torch.testing.assert_close(local(left).cpu(), local(right).cpu(), rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer_snapshot(optimizers[1]), optimizer_snapshot(resumed_optimizer), rtol=0, atol=0
            )
            assert schedulers[1].state_dict() == resumed_scheduler.state_dict()
            assert optimizer_lrs(optimizers[1]) == optimizer_lrs(resumed_optimizer)
            del reference, candidate, resumed, resumed_optimizer, optimizers
            gc.collect()
        with pytest.raises(ValueError, match="did not match selected CPU-offload modules"):
            build(True, ["missing.experts"])


@pytest.mark.launch_plan(provider=_CI_LAUNCH_2)
@pytest.mark.parametrize("fused", [False, True])
def test_selective_expert_cpu_offload_preserves_accumulation_and_update(fused, tmp_path):
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--standalone",
        str(Path(__file__).resolve()),
        "--checkpoint-dir",
        str(tmp_path),
    ]
    if fused:
        command.append("--fused")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    get_torch_device().set_device(f"{get_device_type()}:{local_rank}")
    dist.init_process_group(get_dist_comm_backend())
    try:
        _run_selective_expert_offload(args.fused, args.checkpoint_dir)
    finally:
        try:
            dist.destroy_process_group()
        finally:
            clear_parallel_state()
