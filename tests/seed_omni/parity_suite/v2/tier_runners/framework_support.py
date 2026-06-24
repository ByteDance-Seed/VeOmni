"""Local trainer and reporting helpers for framework-tier parity policies."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ProbeReport,
    autocast_for_dtype,
    compare_values,
    configure_torch_determinism,
    tolerance_from_policy,
    zero_module_grads,
)
from veomni.distributed import parallel_state as parallel_state_module
from veomni.distributed.clip_grad_norm import veomni_omni_module_clip_grad_norm
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.trainer.base import BaseTrainer
from veomni.trainer.omni.omni_module_trainer import OmniModuleTrainer
from veomni.trainer.omni.omni_trainer import MultiLRScheduler, MultiOptimizer, OmniTrainer


@dataclass(frozen=True)
class TrainerStepOptions:
    lr: float
    max_grad_norm: float
    require_active_clip: bool

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> TrainerStepOptions:
        checks = tuple(str(item) for item in options.get("checks", ()) or ())
        return cls(
            lr=float(options.get("lr", 1.0e-4)),
            max_grad_norm=float(options.get("max_grad_norm", 1.0e9)),
            require_active_clip="active_clipping" in checks,
        )


def single_rank_ddp_parallel_state() -> parallel_state_module.ParallelState:
    return parallel_state_module.ParallelState(dp_mode="ddp")


def build_stub_module_trainer(module: nn.Module) -> OmniModuleTrainer:
    """Minimal module-trainer stub for local framework policies."""

    module_trainer = OmniModuleTrainer.__new__(OmniModuleTrainer)
    module_trainer.base = BaseTrainer.__new__(BaseTrainer)
    module_trainer.base.model = module
    module_trainer.parallel_state = single_rank_ddp_parallel_state()
    return module_trainer


def build_module_trainers(model: OmniModel) -> dict[str, OmniModuleTrainer]:
    return {name: build_stub_module_trainer(module) for name, module in model.modules_dict.items()}


def build_minimal_omni_trainer(model: OmniModel, *, device: torch.device, dtype: torch.dtype) -> OmniTrainer:
    """Create only the trainer state needed by ``forward_backward_step`` / ``train_step``."""

    trainer = OmniTrainer.__new__(OmniTrainer)
    base = BaseTrainer.__new__(BaseTrainer)
    base.model = model
    base.device = device
    base.LOG_SAMPLE = False
    base.args = SimpleNamespace(
        train=SimpleNamespace(
            enable_batch_invariant_mode=False,
            local_rank=0,
            accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="ddp")),
        )
    )
    base.preforward = BaseTrainer.preforward.__get__(base, BaseTrainer)
    base.model_fwd_context = autocast_for_dtype(device, dtype)
    base.model_bwd_context = nullcontext()
    trainer.base = base
    trainer.module_trainers = build_module_trainers(model)
    return trainer


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def run_direct_train_step(
    model: OmniModel,
    batch: Mapping[str, Any],
    *,
    seed: int,
    dtype: torch.dtype,
    lr: float,
    max_grad_norm: float,
    num_micro_steps: int = 1,
) -> dict[str, Any]:
    zero_module_grads(model.modules_dict.values())
    optimizer = build_multi_optimizer(model, lr=lr)
    scheduler = build_multi_scheduler(optimizer)
    configure_torch_determinism(seed)
    total_loss = 0.0
    pristine_batch = _clone_batch(batch)
    for _micro_step in range(num_micro_steps):
        micro_batch = _clone_batch(pristine_batch)
        device = next(model.parameters()).device
        with torch.enable_grad(), autocast_for_dtype(device, dtype):
            outputs = model(**micro_batch)
            loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("Direct framework policy produced no loss.")
        loss.backward()
        total_loss += float(loss.detach().cpu().item()) / num_micro_steps
    with single_rank_ddp_clip_state():
        module_grad_norms = [
            veomni_omni_module_clip_grad_norm(module, single_rank_ddp_parallel_state(), max_grad_norm)
            for module in model.modules_dict.values()
        ]
        grad_norm = math.sqrt(sum(norm * norm for norm in module_grad_norms)) if module_grad_norms else 0.0
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return {
        "loss": torch.tensor(total_loss),
        "grad_norm": grad_norm,
        "scheduler_lrs": scheduler.get_last_lr(),
        "scheduler_epochs": {name: int(scheduler.last_epoch) for name, scheduler in scheduler.schedulers.items()},
    }


def run_trainer_train_step(
    driver: Any,
    model: OmniModel,
    batch: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    trainer = build_minimal_omni_trainer(model, device=device, dtype=dtype)
    trainer.base.args.train.optimizer = SimpleNamespace(max_grad_norm=max_grad_norm)
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.optimizer = build_multi_optimizer(model, lr=lr)
    trainer.base.lr_scheduler = build_multi_scheduler(trainer.base.optimizer)
    events: dict[str, Any] = {}
    trainer.on_step_begin = lambda micro_batches=None: events.setdefault("micro_batches", micro_batches)
    trainer.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None: events.update(
        {"loss": loss, "loss_dict": loss_dict, "grad_norm": grad_norm}
    )
    configure_torch_determinism(seed)
    with single_rank_ddp_clip_state():
        trainer.train_step(iter([[dict(batch)]]))
    events["global_step"] = int(trainer.base.state.global_step)
    events["scheduler_lrs"] = trainer.base.lr_scheduler.get_last_lr()
    events["scheduler_epochs"] = {
        name: int(scheduler.last_epoch) for name, scheduler in trainer.base.lr_scheduler.schedulers.items()
    }
    return events


def trainer_step_reports(
    driver: Any,
    *,
    trainer_result: Mapping[str, Any],
    direct_result: Mapping[str, Any],
    trainer_parameters: Mapping[str, torch.Tensor],
    direct_parameters: Mapping[str, torch.Tensor],
    zero_grad_passes: bool,
    options: TrainerStepOptions,
) -> list[ProbeReport]:
    reports = [
        framework_report(
            driver, "framework.loss", torch.tensor([trainer_result["loss"]]), direct_result["loss"].reshape(1), "exact"
        ),
        framework_report(
            driver,
            "framework.grad_norm",
            torch.tensor([float(trainer_result["grad_norm"])]),
            torch.tensor([float(direct_result["grad_norm"])]),
            "gradient",
        ),
        framework_report(
            driver,
            "framework.scheduler_lrs",
            torch.tensor(trainer_result["scheduler_lrs"]),
            torch.tensor(direct_result["scheduler_lrs"]),
            "exact",
        ),
        framework_report(
            driver,
            "framework.scheduler_epochs",
            trainer_result["scheduler_epochs"],
            direct_result["scheduler_epochs"],
            "exact",
        ),
        framework_report(driver, "framework.global_step", trainer_result["global_step"], 1, "exact"),
        framework_report(driver, "framework.zero_grad", zero_grad_passes, True, "exact"),
    ]
    if trainer_parameters or direct_parameters:
        reports.append(
            framework_report(
                driver,
                "framework.parameters_after_step",
                trainer_parameters,
                direct_parameters,
                "gradient",
            )
        )
    if options.require_active_clip:
        reports.append(
            framework_report(
                driver,
                "framework.active_clip",
                float(trainer_result["grad_norm"]) > options.max_grad_norm
                and float(direct_result["grad_norm"]) > options.max_grad_norm,
                True,
                "exact",
            )
        )
    return reports


def fsdp2_numeric_reports(
    driver: Any,
    *,
    report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> list[ProbeReport]:
    reports = [
        framework_report(
            driver,
            "framework.fsdp2_baseline_exit_code",
            baseline_report["exit_code"],
            0,
            "exact",
        ),
        framework_report(
            driver,
            "framework.fsdp2_baseline_all_pass",
            bool(baseline_report.get("all_pass", False)),
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.fsdp2_loss",
            torch.tensor([float(report.get("loss", float("nan")))]),
            torch.tensor([float(baseline_report.get("loss", float("nan")))]),
            "exact",
        ),
        framework_report(
            driver,
            "framework.fsdp2_scheduler_lrs",
            torch.tensor(report.get("scheduler_lrs", [])),
            torch.tensor(baseline_report.get("scheduler_lrs", [])),
            "exact",
        ),
        framework_report(
            driver,
            "framework.fsdp2_scheduler_epochs",
            report.get("scheduler_epochs", {}),
            baseline_report.get("scheduler_epochs", {}),
            "exact",
        ),
        framework_report(
            driver, "framework.fsdp2_zero_grad", bool(report.get("zero_grad_passes", False)), True, "exact"
        ),
    ]
    baseline_parameters = baseline_report.get("parameters_after_step", {})
    if baseline_parameters:
        reports.append(
            framework_report(
                driver,
                "framework.fsdp2_parameters_after_step",
                report.get("parameters_after_step", {}),
                _floating_tensors_to_float32(baseline_parameters),
                "gradient",
            )
        )
    return reports


def _floating_tensors_to_float32(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.float() if torch.is_floating_point(value) else value
    if isinstance(value, Mapping):
        return {key: _floating_tensors_to_float32(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_floating_tensors_to_float32(item) for item in value)
    return value


def _clone_batch(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    return value


def build_multi_optimizer(model: OmniModel, *, lr: float) -> MultiOptimizer:
    return MultiOptimizer(
        {
            name: torch.optim.SGD(module.parameters(), lr=lr)
            for name, module in model.modules_dict.items()
            if any(param.requires_grad for param in module.parameters())
        }
    )


def build_multi_scheduler(optimizer: MultiOptimizer) -> MultiLRScheduler:
    return MultiLRScheduler(
        {
            name: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5**step)
            for name, opt in optimizer.optimizers.items()
        }
    )


def all_grads_are_cleared(model: OmniModel) -> bool:
    return all(param.grad is None for module in model.modules_dict.values() for param in module.parameters())


def framework_report(driver: Any, probe: str, actual: Any, expected: Any, tol: str) -> ProbeReport:
    metric = compare_values(
        actual,
        expected,
        tolerance=tolerance_from_policy(tol, driver.case.model.tolerance),
        path=probe,
    )
    return ProbeReport(node="framework", probe=probe, passed=metric.passed, metric=metric)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def policy_output_dir(driver: Any, name: str) -> Path:
    safe_id = driver.case.node_id.replace(".", "_").replace("/", "_")
    return Path("outputs") / "parity_suite" / name / safe_id


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


@contextmanager
def single_rank_ddp_clip_state():
    """Temporarily fake DDP state for shared grad clipping in local policies."""

    previous = parallel_state_module._PARALLEL_STATE
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(dp_mode="ddp")
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = previous


__all__ = [
    "TrainerStepOptions",
    "all_grads_are_cleared",
    "build_multi_optimizer",
    "build_multi_scheduler",
    "find_free_port",
    "framework_report",
    "fsdp2_numeric_reports",
    "optional_int",
    "policy_output_dir",
    "release_cuda_memory",
    "run_direct_train_step",
    "run_trainer_train_step",
    "single_rank_ddp_clip_state",
    "single_rank_ddp_parallel_state",
    "trainer_step_reports",
    "build_minimal_omni_trainer",
]
