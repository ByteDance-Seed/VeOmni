"""SeedOmni V2 production-data loss smoke worker for the parity suite."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch.distributed as dist


_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from veomni.arguments import OmniArguments, parse_omni_args  # noqa: E402
from veomni.trainer.omni_trainer import OmniTrainer  # noqa: E402
from veomni.utils.device import get_torch_device  # noqa: E402


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set for parity-suite data loss smoke.")
    return Path(value)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))


def _count_fsdp_modules(model: Any) -> int:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return 0
    return sum(1 for module in model.modules() if isinstance(module, FSDPModule))


def _loss_decreased(losses: list[float], *, window: int, expect: bool) -> bool:
    """Compare the first and last loss windows for a coarse downward trend."""

    if not expect:
        return True
    if len(losses) < max(2, window * 2):
        return False
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    return math.isfinite(first) and math.isfinite(last) and last < first


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main() -> None:
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_omni_args(OmniArguments, preload_path_fields=("model.modules",))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if get_torch_device().is_available():
        get_torch_device().set_device(local_rank)

    output_dir = _env_path("VEOMNI_PARITY_DATA_LOSS_OUTPUT_DIR")
    loss_window = int(os.environ.get("VEOMNI_PARITY_DATA_LOSS_WINDOW", "10"))
    expect_decrease = os.environ.get("VEOMNI_PARITY_DATA_LOSS_EXPECT_DECREASE", "true").lower() == "true"

    trainer = OmniTrainer(args)
    trainer.base.LOG_SAMPLE = False
    losses: list[float] = []
    grad_norms: list[float] = []
    original_on_step_end = trainer.on_step_end
    total_steps = int(args.train_steps)

    def _should_log_step(step: int) -> bool:
        return step <= 3 or step % 10 == 0 or step == total_steps

    # Reuse the production trainer loop and observe only step-end metrics needed
    # for a coarse data/loss health check.
    def _record_step(
        loss: float | None = None, loss_dict: dict[str, float] | None = None, grad_norm: float | None = None
    ) -> None:
        if loss is not None:
            losses.append(float(loss))
        if grad_norm is not None:
            grad_norms.append(float(grad_norm))
        if _rank() == 0:
            step = len(losses)
            if _should_log_step(step):
                print(
                    "[data_loss_smoke] "
                    f"step={step} loss={loss if loss is not None else 'NA'} "
                    f"grad_norm={grad_norm if grad_norm is not None else 'NA'}",
                    flush=True,
                )
        original_on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    trainer.on_step_end = _record_step  # type: ignore[method-assign]

    base = trainer.base
    args = base.args
    trainer.on_train_begin()
    try:
        for epoch in range(base.start_epoch, args.train.num_train_epochs):
            if hasattr(base.train_dataloader, "set_epoch"):
                base.train_dataloader.set_epoch(epoch)
            base.state.epoch = epoch
            trainer.on_epoch_begin()
            data_iterator = iter(base.train_dataloader)
            for _ in range(base.start_step, args.train_steps):
                try:
                    trainer.train_step(data_iterator)
                except StopIteration:
                    break
            trainer.on_epoch_end()
            base.start_step = 0
        trainer.on_train_end()

        finite_losses = bool(losses) and all(math.isfinite(loss) for loss in losses)
        finite_grad_norms = bool(grad_norms) and all(math.isfinite(norm) for norm in grad_norms)
        rank_metrics = {
            "rank": _rank(),
            "world_size": _world_size(),
            "steps": len(losses),
            "losses": losses,
            "grad_norms": grad_norms,
            "finite_losses": finite_losses,
            "finite_grad_norms": finite_grad_norms,
            "fsdp_module_count": _count_fsdp_modules(base.model),
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(_world_size())]
        if dist.is_initialized():
            dist.all_gather_object(gathered, rank_metrics)
        else:
            gathered[0] = rank_metrics

        if _rank() == 0:
            main_losses = losses
            decreased = _loss_decreased(main_losses, window=loss_window, expect=expect_decrease)
            all_pass = (
                all(
                    item is not None
                    and item["steps"] == args.train_steps
                    and item["finite_losses"]
                    and item["finite_grad_norms"]
                    and item["fsdp_module_count"] > 0
                    for item in gathered
                )
                and decreased
            )
            report = {
                "all_pass": all_pass,
                "dp_mode": args.train.accelerator.fsdp_config.fsdp_mode,
                "dp_size": args.train.accelerator.dp_size,
                "dp_replicate_size": args.train.accelerator.dp_replicate_size,
                "dp_shard_size": args.train.accelerator.dp_shard_size,
                "global_batch_size": args.train.global_batch_size,
                "micro_batch_size": args.train.micro_batch_size,
                "steps": len(main_losses),
                "loss_window": loss_window,
                "loss_decreased": decreased,
                "gradient_checkpointing": args.train.gradient_checkpointing.enable,
                "losses": main_losses,
                "ranks": gathered,
            }
            _write_report(output_dir, report)
            if not all_pass:
                raise AssertionError(report)
    finally:
        if dist.is_initialized():
            dist.barrier()
        base.destroy_distributed()


if __name__ == "__main__":
    main()
