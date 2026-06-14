"""Default SeedOmni V2 FSDP2 training smoke worker for the parity suite."""

from __future__ import annotations

import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from tests.seed_omni.parity_suite.core import configure_torch_determinism, to_device  # noqa: E402
from veomni.arguments import OmniArguments, parse_omni_args  # noqa: E402
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm  # noqa: E402
from veomni.distributed.parallel_state import init_parallel_state  # noqa: E402
from veomni.models.seed_omni.modeling_omni import OmniModel  # noqa: E402
from veomni.trainer.omni_trainer import MultiLRScheduler, MultiOptimizer, OmniModuleTrainer  # noqa: E402
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device  # noqa: E402


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set for parity-suite FSDP2 smoke.")
    return Path(value)


def _init_distributed() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_comm_backend())

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device_type = get_device_type()
    get_torch_device().set_device(f"{device_type}:{local_rank}")
    return rank, world_size, torch.device(f"{device_type}:{local_rank}")


def _init_parallel_from_args(args: OmniArguments) -> None:
    acc = args.train.accelerator
    init_parallel_state(
        dp_size=acc.dp_size,
        dp_replicate_size=acc.dp_replicate_size,
        dp_shard_size=acc.dp_shard_size,
        tp_size=acc.tp_size,
        pp_size=acc.pp_size,
        cp_size=acc.cp_size,
        ulysses_size=acc.ulysses_size,
        extra_parallel_sizes=acc.extra_parallel_sizes,
        extra_parallel_placement_innermost=acc.extra_parallel_placement_innermost,
        extra_parallel_names=acc.extra_parallel_names,
        dp_mode=acc.fsdp_config.fsdp_mode,
        async_enabled=acc.enable_async,
    )


def _build_fsdp_omni_model(args: OmniArguments) -> OmniModel:
    omni_config = args.load_omni_config()
    modules: dict[str, torch.nn.Module] = {}
    executors: dict[str, Any] = {}

    for name in omni_config.module_names:
        module_trainer = OmniModuleTrainer(omni_config.module_config(name), subfolder_name=name)
        modules[name] = module_trainer.base.model
        executors[name] = module_trainer.forward

    model = OmniModel(omni_config, modules).train()
    model.set_node_executors(executors)
    return model


def _build_sgd_optimizer(model: OmniModel, *, lr: float) -> MultiOptimizer:
    optimizers: dict[str, torch.optim.Optimizer] = {}
    for name, module in model.modules_dict.items():
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            optimizers[name] = torch.optim.SGD(params, lr=lr)
    return MultiOptimizer(optimizers)


def _build_scheduler(optimizer: MultiOptimizer) -> MultiLRScheduler:
    return MultiLRScheduler(
        {
            name: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5**step)
            for name, opt in optimizer.optimizers.items()
        }
    )


def _count_fsdp_modules(model: torch.nn.Module) -> int:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return 0
    return sum(1 for module in model.modules() if isinstance(module, FSDPModule))


def _grad_stats(model: torch.nn.Module) -> dict[str, Any]:
    total_norm_sq = 0.0
    tensors_with_grad = 0
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        total_norm_sq += float(local_grad.detach().float().norm().item() ** 2)
        tensors_with_grad += 1
    return {
        "local_grad_norm": math.sqrt(total_norm_sq),
        "tensors_with_grad": tensors_with_grad,
    }


def main() -> None:
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.modules",))
    rank, world_size, device = _init_distributed()
    _init_parallel_from_args(args)

    payload = torch.load(_env_path("VEOMNI_PARITY_FSDP2_PAYLOAD"), map_location="cpu", weights_only=False)
    output_dir = _env_path("VEOMNI_PARITY_FSDP2_OUTPUT_DIR")
    dtype = torch.bfloat16 if str(payload.get("dtype", "bf16")) == "bf16" else torch.float32

    configure_torch_determinism(int(payload.get("seed", 1234)))
    batch_kwargs = to_device(payload["batch_kwargs"], device)
    model = _build_fsdp_omni_model(args)
    optimizer = _build_sgd_optimizer(model, lr=float(payload.get("lr", 1.0e-4)))
    scheduler = _build_scheduler(optimizer)

    optimizer.zero_grad(set_to_none=True)
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with autocast_context:
        outputs = model(**batch_kwargs)
        loss = outputs["loss"]
    if loss is None:
        raise RuntimeError("SeedOmni V2 FSDP2 smoke produced no loss.")
    loss.backward()

    grad_norm = veomni_clip_grad_norm(model, max_norm=float(payload.get("max_grad_norm", 1.0)))
    grad_stats = _grad_stats(model)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    rank_metrics = {
        "rank": rank,
        "world_size": world_size,
        "loss": float(loss.detach().float().item()),
        "grad_norm": float(grad_norm),
        "local_grad_norm": grad_stats["local_grad_norm"],
        "tensors_with_grad": grad_stats["tensors_with_grad"],
        "fsdp_module_count": _count_fsdp_modules(model),
        "optimizer_count": len(optimizer.optimizers),
        "scheduler_lrs": scheduler.get_last_lr(),
    }
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_metrics)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_pass = all(
            item is not None
            and math.isfinite(item["loss"])
            and math.isfinite(item["grad_norm"])
            and item["grad_norm"] > 0
            and item["tensors_with_grad"] > 0
            and item["fsdp_module_count"] > 0
            and item["optimizer_count"] > 0
            for item in gathered
        )
        report = {
            "all_pass": all_pass,
            "dp_mode": args.train.accelerator.fsdp_config.fsdp_mode,
            "global_batch_size": args.train.global_batch_size,
            "micro_batch_size": args.train.micro_batch_size,
            "ranks": gathered,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        if not all_pass:
            raise AssertionError(report)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
