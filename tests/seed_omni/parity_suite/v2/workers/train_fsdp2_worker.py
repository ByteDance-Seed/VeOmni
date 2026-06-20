"""Default SeedOmni V2 FSDP2 training smoke worker for the parity suite."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from tests.seed_omni.parity_suite.core import configure_torch_determinism, to_device  # noqa: E402
from tests.seed_omni.parity_suite.driver.observations import shifted_label_rows_from_conversation  # noqa: E402
from veomni.arguments import OmniArguments, parse_omni_args  # noqa: E402
from veomni.trainer.omni_trainer import MultiLRScheduler, MultiOptimizer, OmniTrainer  # noqa: E402


try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - older torch builds do not expose DTensor.
    DTensor = None  # type: ignore[assignment]


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set for parity-suite FSDP2 smoke.")
    return Path(value)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))


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


def _all_grads_are_cleared(model: torch.nn.Module) -> bool:
    return all(param.grad is None for param in model.parameters())


def _install_lightweight_optimizer(trainer: OmniTrainer, *, lr: float) -> None:
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR] = {}
    for name, module in trainer.base.model.modules_dict.items():
        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            continue
        optimizer = torch.optim.SGD(params, lr=lr)
        optimizers[name] = optimizer
        schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5**step)
    trainer.optimizers = optimizers
    trainer.lr_schedulers = schedulers
    trainer.base.optimizer = MultiOptimizer(optimizers)
    trainer.base.lr_scheduler = MultiLRScheduler(schedulers)


def _apply_module_config_overrides(model: Any, module_overrides: Mapping[str, Any]) -> None:
    for module_name, override in module_overrides.items():
        if not isinstance(override, Mapping):
            continue
        config_values = override.get("config")
        if not isinstance(config_values, Mapping):
            continue
        module = model.get_module(str(module_name))
        config = getattr(module, "config", None)
        if config is None:
            raise ValueError(f"Cannot apply config override to {module_name!r}: module has no config.")
        for key, value in config_values.items():
            setattr(config, str(key), value)


def _tensor_to_full_cpu(value: torch.Tensor) -> torch.Tensor:
    if DTensor is not None and isinstance(value, DTensor):
        value = value.full_tensor()
    elif hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return value.detach().cpu()


def _sample_param(module: torch.nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    value = _tensor_to_full_cpu(dict(module.named_parameters())[name])
    if rows is not None:
        return value[rows.to(device=value.device)]
    if value.dim() >= 2:
        return value[:4, :4]
    return value[:16]


def _sample_bagel_parameters(model: Any, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Sample BAGEL parameters used by the parent FSDP2 numeric comparison."""

    labels = batch.get("_bagel_train_label_ids")
    label_rows = (
        torch.unique(labels.detach().cpu()).to(dtype=torch.long)
        if torch.is_tensor(labels)
        else shifted_label_rows_from_conversation(batch.get("conversation_list"))
    )
    return {
        "qwen_early_q_proj": _sample_param(
            model.get_module("bagel_qwen2_mot"),
            "model.layers.0.self_attn.q_proj.weight",
        ),
        "lm_head_rows": _sample_param(
            model.get_module("bagel_text_encoder"),
            "lm_head.weight",
            rows=label_rows,
        ),
        "flow_llm2vae": _sample_param(
            model.get_module("bagel_flow_connector"),
            "llm2vae.weight",
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stdout_report(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(report)
    parameters = summary.get("parameters_after_step")
    if isinstance(parameters, Mapping):
        summary["parameters_after_step"] = sorted(parameters)
    return summary


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(_stdout_report(report), indent=2))


def _fresh_micro_batch(batch_kwargs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    # SeedOmni graph nodes mutate ConversationItem values in place. Gradient
    # accumulation therefore needs independent batch objects for each micro-step.
    batch = deepcopy(dict(batch_kwargs))
    # Synthetic parity batches bypass the dataloader, so add ds_idx for multisource meter accounting.
    if "ds_idx" not in batch:
        conversation_list = batch.get("conversation_list")
        batch_size = len(conversation_list) if conversation_list else 1
        batch["ds_idx"] = [0] * batch_size
    return to_device(batch, device)


def main() -> None:
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.modules",))
    payload = torch.load(_env_path("VEOMNI_PARITY_FSDP2_PAYLOAD"), map_location="cpu", weights_only=False)
    output_dir = _env_path("VEOMNI_PARITY_FSDP2_OUTPUT_DIR")
    seed = int(payload.get("seed", args.train.seed))
    num_micro_steps = int(payload.get("num_micro_steps", 1))
    args.train.seed = seed
    args.train.optimizer.lr = float(payload.get("lr", args.train.optimizer.lr))
    args.train.optimizer.max_grad_norm = float(payload.get("max_grad_norm", args.train.optimizer.max_grad_norm))

    trainer = OmniTrainer(args)
    trainer.base.LOG_SAMPLE = False
    _apply_module_config_overrides(trainer.base.model, payload.get("module_overrides", {}) or {})
    _install_lightweight_optimizer(trainer, lr=float(payload.get("lr", args.train.optimizer.lr)))

    events: dict[str, Any] = {}
    original_on_step_end = trainer.on_step_end

    def _record_step(
        loss: float | None = None,
        loss_dict: dict[str, float] | None = None,
        grad_norm: float | None = None,
    ) -> None:
        events.update({"loss": loss, "loss_dict": loss_dict, "grad_norm": grad_norm})
        original_on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    trainer.on_step_end = _record_step  # type: ignore[method-assign]

    base = trainer.base
    device = base.device
    configure_torch_determinism(seed)
    micro_batches = [_fresh_micro_batch(payload["batch_kwargs"], device) for _ in range(num_micro_steps)]
    require_fsdp_modules = bool(payload.get("require_fsdp_modules", True))

    try:
        trainer.on_train_begin()
        base.state.epoch = 0
        trainer.on_epoch_begin()
        trainer.train_step(iter([micro_batches]))
        trainer.on_epoch_end()
        trainer.on_train_end()

        model = base.model
        grad_stats = _grad_stats(model)
        zero_grad_passes = _all_grads_are_cleared(model)
        parameters_after_step = (
            _sample_bagel_parameters(model, micro_batches[0])
            if bool(payload.get("collect_parameter_samples", False))
            else {}
        )
        optimizer_count = len(getattr(base.optimizer, "optimizers", {}))
        scheduler_lrs = base.lr_scheduler.get_last_lr()
        scheduler_epochs = {
            name: int(module_scheduler.last_epoch)
            for name, module_scheduler in getattr(base.lr_scheduler, "schedulers", {}).items()
        }

        rank_metrics = {
            "rank": _rank(),
            "world_size": _world_size(),
            "loss": float(events["loss"]),
            "loss_dict": _jsonable(events.get("loss_dict") or {}),
            "grad_norm": float(events["grad_norm"]),
            "local_grad_norm": grad_stats["local_grad_norm"],
            "tensors_with_grad": grad_stats["tensors_with_grad"],
            "fsdp_module_count": _count_fsdp_modules(model),
            "optimizer_count": optimizer_count,
            "scheduler_lrs": scheduler_lrs,
            "scheduler_epochs": scheduler_epochs,
            "zero_grad_passes": zero_grad_passes,
            "parameters_after_step": _jsonable(parameters_after_step),
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(_world_size())]
        if dist.is_initialized():
            dist.all_gather_object(gathered, rank_metrics)
        else:
            gathered[0] = rank_metrics

        if _rank() == 0:
            reference_rank = gathered[0]
            all_pass = all(
                item is not None
                and reference_rank is not None
                and math.isfinite(item["loss"])
                and math.isfinite(item["grad_norm"])
                and item["grad_norm"] > 0
                and (item["fsdp_module_count"] > 0 or not require_fsdp_modules)
                and item["optimizer_count"] > 0
                and item["zero_grad_passes"]
                and item["loss"] == reference_rank["loss"]
                and item["loss_dict"] == reference_rank["loss_dict"]
                and item["grad_norm"] == reference_rank["grad_norm"]
                and item["scheduler_lrs"] == reference_rank["scheduler_lrs"]
                and item["scheduler_epochs"] == reference_rank["scheduler_epochs"]
                and item["fsdp_module_count"] == reference_rank["fsdp_module_count"]
                and item["optimizer_count"] == reference_rank["optimizer_count"]
                and item["parameters_after_step"] == reference_rank["parameters_after_step"]
                for item in gathered
            )
            report_ranks = [
                {key: value for key, value in item.items() if key != "parameters_after_step"}
                if item is not None
                else None
                for item in gathered
            ]
            report = {
                "all_pass": all_pass,
                "dp_mode": args.train.accelerator.fsdp_config.fsdp_mode,
                "dp_size": args.train.accelerator.dp_size,
                "dp_replicate_size": args.train.accelerator.dp_replicate_size,
                "dp_shard_size": args.train.accelerator.dp_shard_size,
                "global_batch_size": args.train.global_batch_size,
                "micro_batch_size": args.train.micro_batch_size,
                "num_micro_steps": num_micro_steps,
                "loss": rank_metrics["loss"],
                "loss_dict": rank_metrics["loss_dict"],
                "grad_norm": rank_metrics["grad_norm"],
                "scheduler_lrs": rank_metrics["scheduler_lrs"],
                "scheduler_epochs": rank_metrics["scheduler_epochs"],
                "zero_grad_passes": rank_metrics["zero_grad_passes"],
                "require_fsdp_modules": require_fsdp_modules,
                "parameters_after_step": rank_metrics["parameters_after_step"],
                "ranks": report_ranks,
            }
            _write_report(output_dir, report)
            if not all_pass:
                raise AssertionError(report)
    finally:
        if dist.is_initialized():
            base.destroy_distributed()


if __name__ == "__main__":
    main()
