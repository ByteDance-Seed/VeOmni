"""SeedOmni V2 production-data loss smoke worker for the parity suite."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist


_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from veomni.arguments import OmniArguments, parse_omni_args  # noqa: E402
from veomni.ops.batch_invariant_ops import set_batch_invariant_mode  # noqa: E402
from veomni.trainer.omni.omni_trainer import OmniTrainer  # noqa: E402
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


def _debug_enabled() -> bool:
    raw = os.environ.get("VEOMNI_PARITY_WORKER_DEBUG_LOG", os.environ.get("VEOMNI_PARITY_DATA_LOSS_DEBUG", "false"))
    return raw.lower() == "true"


def _conversation_summary(batch: Mapping[str, Any] | None) -> str:
    if not isinstance(batch, Mapping):
        return "batch=unknown"
    conversation_list = batch.get("conversation_list")
    if not isinstance(conversation_list, list):
        return "conversation=missing"
    counts: dict[str, int] = {}
    for sample in conversation_list:
        if not isinstance(sample, list):
            continue
        for item in sample:
            item_type = str(getattr(item, "type", type(item).__name__))
            role = str(getattr(item, "role", "unknown"))
            meta = getattr(item, "meta", {}) or {}
            source = meta.get("source", "raw") if isinstance(meta, Mapping) else "raw"
            key = f"{item_type}:{role}:{source}"
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "conversation=empty"
    return ",".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _loss_keys(result: Mapping[str, Any]) -> str:
    losses = result.get("losses")
    if not isinstance(losses, Mapping):
        return "losses=missing"
    return "losses=" + ",".join(str(key) for key in sorted(losses))


def _debug_log(message: str, *, trainer: OmniTrainer | None = None, batch: Mapping[str, Any] | None = None) -> None:
    if not _debug_enabled():
        return
    global_step = "NA"
    if trainer is not None:
        global_step = str(getattr(trainer.base.state, "global_step", "NA"))
    suffix = f" {_conversation_summary(batch)}" if batch is not None else ""
    print(f"[data_loss_debug][rank={_rank()}][step={global_step}] {message}{suffix}", flush=True)


def _install_debug_hooks(trainer: OmniTrainer) -> None:
    if not _debug_enabled():
        return

    base = trainer.base
    model = base.model
    executors = getattr(model, "_node_executors", None)
    if isinstance(executors, Mapping):
        wrapped_executors: dict[str, Callable[..., dict[str, Any]]] = {}

        for module_name, executor in executors.items():

            def _wrap_executor(
                name: str,
                fn: Callable[..., dict[str, Any]],
            ) -> Callable[..., dict[str, Any]]:
                def _wrapped(method: str, **kwargs: Any) -> dict[str, Any]:
                    _debug_log(f"node_enter {name}.{method}", trainer=trainer, batch=kwargs)
                    out = fn(method, **kwargs)
                    _debug_log(f"node_exit {name}.{method} keys={','.join(sorted(out))}", trainer=trainer)
                    return out

                return _wrapped

            wrapped_executors[str(module_name)] = _wrap_executor(str(module_name), executor)

        model.set_node_executors(wrapped_executors)

    def _debug_forward_backward_step(self: OmniTrainer, micro_batch: dict[str, Any]):
        del self
        _debug_log("preforward_enter", trainer=trainer, batch=micro_batch)
        prepared_batch = base.preforward(micro_batch)
        _debug_log("preforward_exit", trainer=trainer, batch=prepared_batch)

        with base.model_fwd_context, set_batch_invariant_mode(base.args.train.enable_batch_invariant_mode):
            _debug_log("model_forward_enter", trainer=trainer, batch=prepared_batch)
            result: dict[str, Any] = base.model(**prepared_batch)
            _debug_log(f"model_forward_exit {_loss_keys(result)}", trainer=trainer)

        total_loss: torch.Tensor = result["loss"]
        if total_loss is None:
            raise RuntimeError(
                "OmniModel.forward produced no loss — no training node emitted a `_loss`. "
                "Check that the training data + per-module training forwards are wired (D4/D5)."
            )
        loss_dict: dict[str, torch.Tensor] = result.get("losses", {})

        with base.model_bwd_context, set_batch_invariant_mode(base.args.train.enable_batch_invariant_mode):
            _debug_log("backward_enter", trainer=trainer)
            total_loss.backward()
            _debug_log("backward_exit", trainer=trainer)

        del prepared_batch
        return total_loss, loss_dict

    trainer.forward_backward_step = MethodType(_debug_forward_backward_step, trainer)  # type: ignore[method-assign]


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
    _install_debug_hooks(trainer)
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
            for step_index in range(base.start_step, args.train_steps):
                try:
                    _debug_log(f"train_step_enter loop_step={step_index + 1}", trainer=trainer)
                    trainer.train_step(data_iterator)
                    _debug_log(f"train_step_exit loop_step={step_index + 1}", trainer=trainer)
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
