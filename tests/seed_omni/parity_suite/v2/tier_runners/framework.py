"""Default framework-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

import gc
import json
import os
import runpy
import select
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ParityReport,
    ProbeReport,
    autocast_for_dtype,
    compare_values,
    tolerance_from_policy,
    zero_module_grads,
)
from tests.seed_omni.parity_suite.v2.model import load_graph_active_omni_config
from veomni.distributed import parallel_state as parallel_state_module
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks import TrainerState
from veomni.trainer.omni_trainer import (
    MultiLRScheduler,
    MultiOptimizer,
    OmniGlobalStateCallback,
    OmniModuleDcpCallback,
    OmniModuleTrainer,
    OmniTrainer,
)
from veomni.utils.device import get_dist_comm_backend, get_torch_device


@dataclass(frozen=True)
class _TrainerStepOptions:
    lr: float
    max_grad_norm: float
    require_active_clip: bool

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> _TrainerStepOptions:
        checks = tuple(str(item) for item in options.get("checks", ()) or ())
        return cls(
            lr=float(options.get("lr", 1.0e-4)),
            max_grad_norm=float(options.get("max_grad_norm", 1.0e9)),
            require_active_clip="active_clipping" in checks,
        )


def run_v2_train_framework(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | ParityReport:
    """Run training framework tier through mapping comparison or policy reports."""

    run = driver.case.run
    if run.kind == "forward_backward":
        return run_v2_train_framework_batch(
            driver,
            driver.v2_request_kwargs(reference_output, device=device),
            whitelist,
            device=device,
            dtype=dtype,
        )

    _release_cuda_memory()
    try:
        if run.kind == "train_step":
            return _run_train_step_policy(driver, reference_output, device=device, dtype=dtype)
        if run.kind == "checkpoint_resume":
            return _run_checkpoint_resume_policy(driver, reference_output, device=device, dtype=dtype)
        if run.kind == "launcher":
            return _run_launcher_policy(driver, reference_output, device=device, dtype=dtype)
        if run.kind == "optimizer_trajectory":
            return _run_optimizer_trajectory_policy(driver, reference_output, device=device, dtype=dtype)
        if run.kind == "distributed_train":
            return _run_distributed_train_policy(driver, reference_output, device=device, dtype=dtype)
        if run.kind == "data_loss_smoke":
            return _run_data_loss_smoke_policy(driver, device=device, dtype=dtype)
        raise NotImplementedError(f"Unsupported training framework kind {run.kind!r}.")
    finally:
        _release_cuda_memory()


def run_v2_infer_framework(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | ParityReport:
    """Run inference framework tier through distributed policy reports."""

    run = driver.case.run
    _release_cuda_memory()
    try:
        if run.kind == "distributed_infer":
            return _run_distributed_infer_policy(driver, reference_output, device=device, dtype=dtype)
        raise NotImplementedError(f"Unsupported inference framework kind {run.kind!r}.")
    finally:
        _release_cuda_memory()


def run_v2_train_framework_batch(
    driver: Any,
    batch_kwargs: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = driver.load_v2_model(device=device, dtype=dtype).train()
    model.set_node_executors(build_trainer_node_executors(model))
    zero_module_grads(model.modules_dict.values())
    batch = dict(batch_kwargs)
    trainer = build_minimal_omni_trainer(model, device=device, dtype=dtype)
    loss, loss_dict = trainer.forward_backward_step(batch)
    if loss is None:
        raise RuntimeError(f"{type(driver).__name__} V2 train framework tier produced no loss.")
    observations = driver.collect_v2_train_framework_observations(model, loss_dict, whitelist, batch=batch)
    return {"observations": observations, "ctx": {"loss": loss, "losses": loss_dict}, "trace": ["train:framework"]}


def _single_rank_ddp_parallel_state() -> parallel_state_module.ParallelState:
    return parallel_state_module.ParallelState(dp_mode="ddp")


def build_stub_module_trainer(module: nn.Module) -> OmniModuleTrainer:
    """Minimal module-trainer stub for local framework policies."""

    module_trainer = OmniModuleTrainer.__new__(OmniModuleTrainer)
    module_trainer.base = BaseTrainer.__new__(BaseTrainer)
    module_trainer.base.model = module
    module_trainer.parallel_state = _single_rank_ddp_parallel_state()
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


def build_trainer_node_executors(model: OmniModel) -> dict[str, Any]:
    """Bind graph nodes to ``OmniModuleTrainer.forward`` without full trainer setup."""

    return {name: build_stub_module_trainer(module).forward for name, module in model.modules_dict.items()}


def _run_train_step_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = _TrainerStepOptions.from_options(driver.case.run.options)
    direct_batch = driver.v2_request_kwargs(reference_output, device=device)
    trainer_batch = driver.v2_request_kwargs(reference_output, device=device)

    # Compare the production trainer step against a direct optimizer step using
    # fresh models with identical seeds, batches, optimizer, and scheduler.
    driver.configure_determinism(driver.case.model.seed)
    direct_model = driver.load_v2_model(device=device, dtype=dtype).train()
    direct_result = _run_direct_train_step(
        direct_model,
        direct_batch,
        seed=driver.case.model.seed,
        dtype=dtype,
        lr=options.lr,
        max_grad_norm=options.max_grad_norm,
    )
    direct_parameters = driver.sample_v2_framework_parameters(direct_model, direct_batch)
    del direct_model, direct_batch
    _release_cuda_memory()

    driver.configure_determinism(driver.case.model.seed)
    trainer_model = driver.load_v2_model(device=device, dtype=dtype).train()
    trainer_model.set_node_executors(build_trainer_node_executors(trainer_model))
    trainer_result = _run_trainer_train_step(
        driver,
        trainer_model,
        trainer_batch,
        seed=driver.case.model.seed,
        device=device,
        dtype=dtype,
        lr=options.lr,
        max_grad_norm=options.max_grad_norm,
    )
    trainer_parameters = driver.sample_v2_framework_parameters(trainer_model, trainer_batch)
    zero_grad_passes = _all_grads_are_cleared(trainer_model)
    reports = _trainer_step_reports(
        driver,
        trainer_result=trainer_result,
        direct_result=direct_result,
        trainer_parameters=trainer_parameters,
        direct_parameters=direct_parameters,
        zero_grad_passes=zero_grad_passes,
        options=options,
    )
    del trainer_model, trainer_batch
    _release_cuda_memory()
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_checkpoint_resume_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = _TrainerStepOptions.from_options(driver.case.run.options)
    output_dir = _policy_output_dir(driver, "checkpoint_resume")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _single_rank_process_group():
        driver.configure_determinism(driver.case.model.seed)
        save_model = driver.load_v2_model(device=device, dtype=dtype).train()
        save_model.set_node_executors(build_trainer_node_executors(save_model))
        save_batch = driver.v2_request_kwargs(reference_output, device=device)
        save_trainer = build_minimal_omni_trainer(save_model, device=device, dtype=dtype)
        _attach_checkpoint_state(
            save_trainer,
            output_dir=output_dir,
            load_path=None,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        _run_checkpoint_train_step(save_trainer, save_batch, seed=driver.case.model.seed)
        saved_parameters = driver.sample_v2_framework_parameters(save_model, save_batch)
        # Reuse the post-step sampling context so resume compares the same rows
        # without keeping graph-bearing batch tensors alive across model reload.
        sample_context = getattr(driver, "framework_parameter_sample_context", lambda batch: batch)(save_batch)
        saved_lrs = save_trainer.base.lr_scheduler.get_last_lr()
        checkpoint_root = output_dir / "checkpoints" / "global_step_1"
        saved_layout = _checkpoint_layout(save_model, checkpoint_root)
        del save_trainer, save_model, save_batch
        _release_cuda_memory()

        driver.configure_determinism(driver.case.model.seed)
        resume_model = driver.load_v2_model(device=device, dtype=dtype).train()
        resume_model.set_node_executors(build_trainer_node_executors(resume_model))
        resume_batch = driver.v2_request_kwargs(reference_output, device=device)
        resume_trainer = build_minimal_omni_trainer(resume_model, device=device, dtype=dtype)
        _attach_checkpoint_state(
            resume_trainer,
            output_dir=output_dir,
            load_path=checkpoint_root,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        _checkpoint_on_train_begin(resume_trainer)
        resumed_parameters = driver.sample_v2_framework_parameters(resume_model, sample_context)
        resume_lrs = resume_trainer.base.lr_scheduler.get_last_lr()
        resume_global_step = int(resume_trainer.base.state.global_step)
        _run_checkpoint_train_step(resume_trainer, resume_batch, seed=driver.case.model.seed)
        final_global_step = int(resume_trainer.base.state.global_step)

    reports = [
        _framework_report(driver, "framework.checkpoint_layout", saved_layout, True, "exact"),
        _framework_report(driver, "framework.resume_global_step", resume_global_step, 1, "exact"),
        _framework_report(driver, "framework.final_global_step", final_global_step, 2, "exact"),
        _framework_report(
            driver,
            "framework.scheduler_lrs_after_resume",
            torch.tensor(resume_lrs),
            torch.tensor(saved_lrs),
            "exact",
        ),
        _framework_report(
            driver, "framework.parameters_after_resume", resumed_parameters, saved_parameters, "gradient"
        ),
    ]
    del resume_trainer, resume_model, resume_batch, sample_context
    _release_cuda_memory()
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_launcher_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = driver.case.run.options
    launcher_path = Path(str(options.get("launcher_path", "tasks/omni/train_omni.py")))
    config_path = Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml")))
    output_dir = _policy_output_dir(driver, "launcher")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = driver.v2_request_kwargs(reference_output, device=device)
    report = _run_launcher_smoke(
        driver,
        batch,
        launcher_path=launcher_path,
        config_path=config_path,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
    )
    reports = [
        _framework_report(driver, "framework.launcher_train_called", report["train_called"], True, "exact"),
        _framework_report(driver, "framework.launcher_global_step", report["global_step"], 1, "exact"),
        _framework_report(driver, "framework.launcher_zero_grad", report["zero_grad_passes"], True, "exact"),
        _framework_report(
            driver,
            "framework.launcher_module_count",
            report["module_count"],
            len(driver.load_v2_model(device=device, dtype=dtype).modules_dict),
            "exact",
        ),
    ]
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_optimizer_trajectory_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = driver.case.run.options
    steps = int(options.get("steps", 2))
    lr = float(options.get("lr", 1.0e-4))
    first = _run_optimizer_trajectory(driver, reference_output, device=device, dtype=dtype, steps=steps, lr=lr)
    _release_cuda_memory()
    second = _run_optimizer_trajectory(driver, reference_output, device=device, dtype=dtype, steps=steps, lr=lr)
    reports = [
        _framework_report(driver, "framework.optimizer_trajectory.losses", second["losses"], first["losses"], "loss"),
        _framework_report(
            driver,
            "framework.optimizer_trajectory.parameters",
            second["parameters_after_step"],
            first["parameters_after_step"],
            "gradient",
        ),
    ]
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_distributed_train_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = driver.case.run.options
    strategy = str(options.get("strategy", ""))
    if strategy != "fsdp2":
        raise NotImplementedError(f"Unsupported distributed_train strategy {strategy!r}.")

    nproc = int(options.get("nproc_per_node", 2))
    lr = float(options.get("lr", 1.0e-4))
    max_grad_norm = float(options.get("max_grad_norm", 1.0))
    num_micro_steps = int(options.get("num_micro_steps", 1))
    compare_direct = bool(options.get("compare_direct", False))
    dp_replicate_size = _optional_int(options.get("dp_replicate_size"))
    dp_shard_size = _optional_int(options.get("dp_shard_size"))
    output_dir = _policy_output_dir(driver, "distributed_train_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_kwargs = driver.v2_request_kwargs(reference_output, device=torch.device("cpu"))
    baseline_report: Mapping[str, Any] | None = None
    if compare_direct:
        baseline_report = _run_v2_train_fsdp2(
            driver,
            batch_kwargs,
            config_path=Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml"))),
            output_dir=output_dir / "baseline_rank1",
            dtype=dtype,
            nproc_per_node=1,
            timeout=int(options.get("timeout", 900)),
            lr=lr,
            max_grad_norm=max_grad_norm,
            num_micro_steps=num_micro_steps,
            collect_parameter_samples=True,
            dp_replicate_size=None,
            dp_shard_size=None,
            fsdp_mode="ddp",
            init_device="cuda",
            mixed_precision=False,
            require_fsdp_modules=False,
        )
    report = _run_v2_train_fsdp2(
        driver,
        batch_kwargs,
        config_path=Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml"))),
        output_dir=output_dir / "target",
        dtype=dtype,
        nproc_per_node=nproc,
        timeout=int(options.get("timeout", 900)),
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_micro_steps=num_micro_steps,
        collect_parameter_samples=compare_direct,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        fsdp_mode="fsdp2",
        init_device="meta",
        mixed_precision=False,
        require_fsdp_modules=True,
    )
    reports = [
        _framework_report(driver, "framework.fsdp2_exit_code", report["exit_code"], 0, "exact"),
        _framework_report(driver, "framework.fsdp2_all_pass", bool(report.get("all_pass", False)), True, "exact"),
        _framework_report(driver, "framework.fsdp2_dp_mode", report.get("dp_mode"), "fsdp2", "exact"),
        _framework_report(driver, "framework.fsdp2_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        _framework_report(
            driver,
            "framework.fsdp2_num_micro_steps",
            int(report.get("num_micro_steps", 0)),
            num_micro_steps,
            "exact",
        ),
    ]
    if dp_replicate_size is not None:
        reports.append(
            _framework_report(
                driver,
                "framework.fsdp2_dp_replicate_size",
                int(report.get("dp_replicate_size", 0)),
                dp_replicate_size,
                "exact",
            )
        )
    if dp_shard_size is not None:
        reports.append(
            _framework_report(
                driver,
                "framework.fsdp2_dp_shard_size",
                int(report.get("dp_shard_size", 0)),
                dp_shard_size,
                "exact",
            )
        )
    if compare_direct:
        if baseline_report is None:
            raise RuntimeError("FSDP2 numeric policy did not build a baseline report.")
        reports.extend(
            _fsdp2_numeric_reports(
                driver,
                report=report,
                baseline_report=baseline_report,
            )
        )
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_distributed_infer_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = driver.case.run.options
    strategy = str(options.get("strategy", ""))
    if strategy != "fsdp2":
        raise NotImplementedError(f"Unsupported distributed_infer strategy {strategy!r}.")

    nproc = int(options.get("nproc_per_node", 2))
    compare_eager = bool(options.get("compare_eager", False))
    output_dir = _policy_output_dir(driver, "distributed_infer_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_kwargs = driver.v2_request_kwargs(reference_output, device=torch.device("cpu"))
    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    generation_kwargs = driver.generation_kwargs(config, reference_output)
    probe_names = tuple(str(name) for name in options.get("compare_probes", ()) or driver.case.run.probes)
    baseline_report: Mapping[str, Any] | None = None
    if compare_eager:
        baseline_report = _run_v2_infer_fsdp2(
            driver,
            request_kwargs,
            generation_kwargs=generation_kwargs,
            config_path=Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml"))),
            modules_config=Path(
                str(
                    options.get(
                        "baseline_modules_config",
                        driver.case.model.v2_model.config_dir / "modules_infer_eager.yaml",
                    )
                )
            ),
            infer_type=str(options.get("infer_type", driver.case.graph.name)),
            output_dir=output_dir / "baseline_rank1",
            dtype=dtype,
            nproc_per_node=1,
            timeout=int(options.get("timeout", 1800)),
            probe_names=probe_names,
            require_fsdp_modules=False,
        )
    report = _run_v2_infer_fsdp2(
        driver,
        request_kwargs,
        generation_kwargs=generation_kwargs,
        config_path=Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml"))),
        modules_config=Path(
            str(
                options.get(
                    "modules_config",
                    driver.case.model.v2_model.config_dir / "modules_infer_fsdp.yaml",
                )
            )
        ),
        infer_type=str(options.get("infer_type", driver.case.graph.name)),
        output_dir=output_dir / "target",
        dtype=dtype,
        nproc_per_node=nproc,
        timeout=int(options.get("timeout", 1800)),
        probe_names=probe_names,
        require_fsdp_modules=True,
    )
    reports = [
        _framework_report(driver, "framework.infer_fsdp2_exit_code", report["exit_code"], 0, "exact"),
        _framework_report(
            driver, "framework.infer_fsdp2_all_pass", bool(report.get("all_pass", False)), True, "exact"
        ),
        _framework_report(driver, "framework.infer_fsdp2_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        _framework_report(
            driver,
            "framework.infer_fsdp2_has_fsdp_modules",
            int(report.get("fsdp_module_count", 0)) > 0,
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.infer_fsdp2_trace_has_image_flow",
            bool(report.get("trace_has_image_flow", False)),
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.infer_fsdp2_trace_has_prompt_encode",
            bool(report.get("trace_has_prompt_encode", False)),
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.infer_fsdp2_rank0_finalized",
            bool(report.get("rank0_finalized", False)),
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.infer_fsdp2_finite_observations",
            bool(report.get("finite_observations", False)),
            True,
            "exact",
        ),
    ]
    if compare_eager:
        if baseline_report is None:
            raise RuntimeError("Inference FSDP2 numeric policy did not build a baseline report.")
        reports.extend(
            _infer_fsdp2_numeric_reports(
                driver,
                report=report,
                baseline_report=baseline_report,
                probe_names=probe_names,
            )
        )
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _infer_fsdp2_numeric_reports(
    driver: Any,
    *,
    report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    probe_names: tuple[str, ...],
) -> list[ProbeReport]:
    reports = [
        _framework_report(
            driver,
            "framework.infer_fsdp2_baseline_exit_code",
            baseline_report["exit_code"],
            0,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.infer_fsdp2_baseline_all_pass",
            bool(baseline_report.get("all_pass", False)),
            True,
            "exact",
        ),
    ]
    baseline_observations = baseline_report.get("observations", {})
    target_observations = report.get("observations", {})
    for probe_name in probe_names:
        mapping = driver.case.model.probes.for_probe_names((probe_name,))
        if not mapping:
            continue
        probe_mapping = mapping[0]
        tol = probe_mapping.tol
        actual = target_observations.get(probe_name)
        expected = baseline_observations.get(probe_name)
        reports.append(
            _framework_report(
                driver,
                f"framework.infer_fsdp2_{probe_name.replace('.', '_')}",
                actual,
                expected,
                tol,
            )
        )
    return reports


def _run_v2_infer_fsdp2(
    driver: Any,
    request_kwargs: Mapping[str, Any],
    *,
    generation_kwargs: Mapping[str, Any],
    config_path: Path,
    modules_config: Path,
    infer_type: str,
    output_dir: Path,
    dtype: torch.dtype,
    nproc_per_node: int,
    timeout: int,
    probe_names: tuple[str, ...],
    require_fsdp_modules: bool,
) -> Mapping[str, Any]:
    """Run the inference FSDP2 worker and load its rank-0 ``results.json`` report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "infer_fsdp2_payload.pt"
    torch.save(
        {
            "request_kwargs": dict(request_kwargs),
            "generation_kwargs": dict(generation_kwargs),
            "probe_names": probe_names,
            "seed": driver.case.model.seed,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "require_fsdp_modules": require_fsdp_modules,
        },
        payload_path,
    )
    port = _find_free_port()
    worker_path = Path("tests/seed_omni/parity_suite/v2/workers/infer_fsdp2_worker.py")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(worker_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.model.v2_model.model_root),
        "--infer.model_path",
        str(driver.case.model.v2_model.model_root),
        "--infer.infer_type",
        infer_type,
        "--infer.modules",
        str(modules_config),
        "--infer.prompt",
        "parity-suite-infer",
        "--infer.output_dir",
        str(output_dir),
        "--infer.seed",
        str(driver.case.model.seed),
    ]
    env = {
        **os.environ,
        "VEOMNI_PARITY_INFER_FSDP2_PAYLOAD": str(payload_path),
        "VEOMNI_PARITY_INFER_FSDP2_OUTPUT_DIR": str(output_dir),
    }
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stdout_chunks: list[str] = []
        deadline = time.monotonic() + timeout
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_chunks.append(line)
                log_file.write(line)
                log_file.flush()
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report_path = output_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stdout = "".join(stdout_chunks)
    report["exit_code"] = returncode
    report["log_path"] = str(log_path)
    report["stdout_tail"] = stdout[-4000:]
    report["stderr_tail"] = ""
    return report


def _run_data_loss_smoke_policy(
    driver: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    del device, dtype
    options = driver.case.run.options
    strategy = str(options.get("strategy", "fsdp2"))
    if strategy != "fsdp2":
        raise NotImplementedError(f"Unsupported data_loss_smoke strategy {strategy!r}.")

    nproc = int(options.get("nproc_per_node", 8))
    steps = int(options.get("steps", 100))
    loss_window = int(options.get("loss_window", 10))
    output_dir = _policy_output_dir(driver, "data_loss_smoke_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _run_v2_train_data_loss_smoke(
        driver,
        config_path=Path(str(options.get("config_path", driver.case.model.v2_model.config_dir / "base.yaml"))),
        output_dir=output_dir,
        nproc_per_node=nproc,
        steps=steps,
        timeout=int(options.get("timeout", 7200)),
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
        loss_window=loss_window,
        expect_loss_decrease=bool(options.get("expect_loss_decrease", True)),
        data_source_names=tuple(str(name) for name in options.get("data_source_names", ()) or ()),
    )
    _write_data_loss_manifest(driver, report=report, output_dir=output_dir)
    reports = [
        _framework_report(driver, "framework.data_loss_exit_code", report["exit_code"], 0, "exact"),
        _framework_report(driver, "framework.data_loss_all_pass", bool(report.get("all_pass", False)), True, "exact"),
        _framework_report(driver, "framework.data_loss_dp_mode", report.get("dp_mode"), "fsdp2", "exact"),
        _framework_report(driver, "framework.data_loss_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        _framework_report(
            driver,
            "framework.data_loss_steps",
            int(report.get("steps", 0)),
            steps,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.data_loss_gradient_checkpointing",
            bool(report.get("gradient_checkpointing", False)),
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.data_loss_decreased",
            bool(report.get("loss_decreased", False)),
            True,
            "exact",
        ),
    ]
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _run_direct_train_step(
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
    optimizer = _build_multi_optimizer(model, lr=lr)
    scheduler = _build_multi_scheduler(optimizer)
    torch.manual_seed(seed)
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
    with _single_rank_ddp_clip_state():
        grad_norm = veomni_clip_grad_norm(model, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return {
        "loss": torch.tensor(total_loss),
        "grad_norm": grad_norm,
        "scheduler_lrs": scheduler.get_last_lr(),
        "scheduler_epochs": {name: int(scheduler.last_epoch) for name, scheduler in scheduler.schedulers.items()},
    }


def _run_trainer_train_step(
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
    trainer.base.optimizer = _build_multi_optimizer(model, lr=lr)
    trainer.base.lr_scheduler = _build_multi_scheduler(trainer.base.optimizer)
    events: dict[str, Any] = {}
    trainer.on_step_begin = lambda micro_batches=None: events.setdefault("micro_batches", micro_batches)
    trainer.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None: events.update(
        {"loss": loss, "loss_dict": loss_dict, "grad_norm": grad_norm}
    )
    torch.manual_seed(seed)
    with _single_rank_ddp_clip_state():
        trainer.train_step(iter([[dict(batch)]]))
    events["global_step"] = int(trainer.base.state.global_step)
    events["scheduler_lrs"] = trainer.base.lr_scheduler.get_last_lr()
    events["scheduler_epochs"] = {
        name: int(scheduler.last_epoch) for name, scheduler in trainer.base.lr_scheduler.schedulers.items()
    }
    return events


def _trainer_step_reports(
    driver: Any,
    *,
    trainer_result: Mapping[str, Any],
    direct_result: Mapping[str, Any],
    trainer_parameters: Mapping[str, torch.Tensor],
    direct_parameters: Mapping[str, torch.Tensor],
    zero_grad_passes: bool,
    options: _TrainerStepOptions,
) -> list[ProbeReport]:
    reports = [
        _framework_report(
            driver, "framework.loss", torch.tensor([trainer_result["loss"]]), direct_result["loss"].reshape(1), "loss"
        ),
        _framework_report(
            driver,
            "framework.grad_norm",
            torch.tensor([float(trainer_result["grad_norm"])]),
            torch.tensor([float(direct_result["grad_norm"])]),
            "gradient",
        ),
        _framework_report(
            driver,
            "framework.scheduler_lrs",
            torch.tensor(trainer_result["scheduler_lrs"]),
            torch.tensor(direct_result["scheduler_lrs"]),
            "exact",
        ),
        _framework_report(
            driver,
            "framework.scheduler_epochs",
            trainer_result["scheduler_epochs"],
            direct_result["scheduler_epochs"],
            "exact",
        ),
        _framework_report(driver, "framework.global_step", trainer_result["global_step"], 1, "exact"),
        _framework_report(driver, "framework.zero_grad", zero_grad_passes, True, "exact"),
    ]
    if trainer_parameters or direct_parameters:
        reports.append(
            _framework_report(
                driver,
                "framework.parameters_after_step",
                trainer_parameters,
                direct_parameters,
                "gradient",
            )
        )
    if options.require_active_clip:
        reports.append(
            _framework_report(
                driver,
                "framework.active_clip",
                float(trainer_result["grad_norm"]) > options.max_grad_norm
                and float(direct_result["grad_norm"]) > options.max_grad_norm,
                True,
                "exact",
            )
        )
    return reports


def _fsdp2_numeric_reports(
    driver: Any,
    *,
    report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> list[ProbeReport]:
    reports = [
        _framework_report(
            driver,
            "framework.fsdp2_baseline_exit_code",
            baseline_report["exit_code"],
            0,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.fsdp2_baseline_all_pass",
            bool(baseline_report.get("all_pass", False)),
            True,
            "exact",
        ),
        _framework_report(
            driver,
            "framework.fsdp2_loss",
            torch.tensor([float(report.get("loss", float("nan")))]),
            torch.tensor([float(baseline_report.get("loss", float("nan")))]),
            "distributed_loss",
        ),
        _framework_report(
            driver,
            "framework.fsdp2_grad_norm",
            torch.tensor([float(report.get("grad_norm", float("nan")))]),
            torch.tensor([float(baseline_report.get("grad_norm", float("nan")))]),
            "gradient",
        ),
        _framework_report(
            driver,
            "framework.fsdp2_scheduler_lrs",
            torch.tensor(report.get("scheduler_lrs", [])),
            torch.tensor(baseline_report.get("scheduler_lrs", [])),
            "exact",
        ),
        _framework_report(
            driver,
            "framework.fsdp2_scheduler_epochs",
            report.get("scheduler_epochs", {}),
            baseline_report.get("scheduler_epochs", {}),
            "exact",
        ),
        _framework_report(
            driver, "framework.fsdp2_zero_grad", bool(report.get("zero_grad_passes", False)), True, "exact"
        ),
    ]
    baseline_parameters = baseline_report.get("parameters_after_step", {})
    if baseline_parameters:
        reports.append(
            _framework_report(
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


def _build_multi_optimizer(model: OmniModel, *, lr: float) -> MultiOptimizer:
    return MultiOptimizer(
        {
            name: torch.optim.SGD(module.parameters(), lr=lr)
            for name, module in model.modules_dict.items()
            if any(param.requires_grad for param in module.parameters())
        }
    )


def _build_multi_scheduler(optimizer: MultiOptimizer) -> MultiLRScheduler:
    return MultiLRScheduler(
        {
            name: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 0.5**step)
            for name, opt in optimizer.optimizers.items()
        }
    )


def _all_grads_are_cleared(model: OmniModel) -> bool:
    return all(param.grad is None for module in model.modules_dict.values() for param in module.parameters())


def _framework_report(driver: Any, probe: str, actual: Any, expected: Any, tol: str) -> ProbeReport:
    metric = compare_values(
        actual,
        expected,
        tolerance=tolerance_from_policy(tol, driver.case.model.tolerance),
        path=probe,
    )
    return ProbeReport(node="framework", probe=probe, passed=metric.passed, metric=metric)


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _StatefulStub:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = dict(state)


class _CheckpointModuleTrainer:
    def __init__(
        self,
        name: str,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        args: SimpleNamespace,
    ) -> None:
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.model = module
        self.base.optimizer = optimizer
        self.base.lr_scheduler = lr_scheduler
        self.base.args = args
        self.parallel_state = _single_rank_ddp_parallel_state()
        self.callback = OmniModuleDcpCallback(self.base, name)

    def on_train_begin(self, state: TrainerState) -> None:
        self.callback.on_train_begin(state)

    def on_step_end(self, state: TrainerState, loss: float, loss_dict: dict[str, float], grad_norm: float) -> None:
        self.callback.on_step_end(state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


def _checkpoint_args(output_dir: Path, *, load_path: Path | None, max_grad_norm: float) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(lora_config=None, fqn_to_index_mapping=None),
        train=SimpleNamespace(
            enable_batch_invariant_mode=False,
            local_rank=0,
            global_rank=0,
            optimizer=SimpleNamespace(max_grad_norm=max_grad_norm),
            accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="ddp")),
            checkpoint=SimpleNamespace(
                output_dir=str(output_dir),
                save_path=str(output_dir / "checkpoints"),
                model_assets_dir=str(output_dir / "model_assets"),
                load_path=None if load_path is None else str(load_path),
                manager="dcp",
                save_async=False,
                save_steps=1,
                save_epochs=0,
                save_hf_weights=False,
                hf_save_steps=0,
                hf_save_epochs=0,
            ),
        ),
        train_steps=2,
    )


def _attach_checkpoint_state(
    trainer: OmniTrainer,
    *,
    output_dir: Path,
    load_path: Path | None,
    lr: float,
    max_grad_norm: float,
) -> None:
    args = _checkpoint_args(output_dir, load_path=load_path, max_grad_norm=max_grad_norm)
    trainer.base.args = args
    trainer.base.state = TrainerState()
    trainer.base.start_epoch = 0
    trainer.base.start_step = 0
    trainer.base.train_dataloader = None
    trainer.base.environ_meter = _StatefulStub()
    trainer.base.optimizer = _build_multi_optimizer(trainer.base.model, lr=lr)
    trainer.base.lr_scheduler = _build_multi_scheduler(trainer.base.optimizer)
    trainer.base.checkpointer_callback = OmniGlobalStateCallback(trainer)
    trainer.module_trainers = {}
    for name, module in trainer.base.model.modules_dict.items():
        optimizer = trainer.base.optimizer.optimizers[name]
        scheduler = trainer.base.lr_scheduler.schedulers[name]
        trainer.module_trainers[name] = _CheckpointModuleTrainer(name, module, optimizer, scheduler, args)


def _checkpoint_on_train_begin(trainer: OmniTrainer) -> None:
    with _single_rank_ddp_clip_state():
        trainer.base.checkpointer_callback.on_train_begin(trainer.base.state)
        for module_trainer in trainer.module_trainers.values():
            module_trainer.on_train_begin(trainer.base.state)


def _run_checkpoint_train_step(trainer: OmniTrainer, batch: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    events: dict[str, Any] = {}
    trainer.on_step_begin = lambda micro_batches=None: events.setdefault("micro_batches", micro_batches)

    def _on_step_end(loss: float, loss_dict: dict[str, float], grad_norm: float) -> None:
        events.update({"loss": loss, "loss_dict": loss_dict, "grad_norm": grad_norm})
        for module_trainer in trainer.module_trainers.values():
            module_trainer.on_step_end(trainer.base.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        trainer.base.checkpointer_callback.on_step_end(
            trainer.base.state,
            loss=loss,
            loss_dict=loss_dict,
            grad_norm=grad_norm,
        )

    trainer.on_step_end = _on_step_end
    torch.manual_seed(seed)
    with _single_rank_ddp_clip_state():
        trainer.train_step(iter([[dict(batch)]]))
    return events


def _checkpoint_layout(model: OmniModel, checkpoint_root: Path) -> bool:
    trainer_state_path = checkpoint_root / "trainer_state.pt"
    module_dirs = [checkpoint_root / name for name in model.modules_dict]
    return bool(trainer_state_path.exists() and all(path.exists() and any(path.iterdir()) for path in module_dirs))


@contextmanager
def _single_rank_process_group():
    if dist.is_initialized():
        yield
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12359")
    if torch.cuda.is_available():
        get_torch_device().set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=get_dist_comm_backend(), rank=0, world_size=1)
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_optimizer_trajectory(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    driver.configure_determinism(driver.case.model.seed)
    model = driver.load_v2_model(device=device, dtype=dtype).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[torch.Tensor] = []
    parameters_after_step: list[Mapping[str, torch.Tensor]] = []
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch = driver.v2_request_kwargs(reference_output, device=device)
        with torch.enable_grad(), autocast_for_dtype(device, dtype):
            outputs = model(**dict(batch))
            loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("Optimizer trajectory produced no loss.")
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().reshape(1))
        parameters_after_step.append(driver.sample_v2_framework_parameters(model, batch))
    return {"losses": losses, "parameters_after_step": parameters_after_step}


def _run_launcher_smoke(
    driver: Any,
    batch: Mapping[str, Any],
    *,
    launcher_path: Path,
    config_path: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Exercise the training launcher path with a recipe-backed trainer stub."""

    import veomni.trainer.omni_trainer as omni_trainer_module

    report: dict[str, Any] = {"train_called": False}
    lr = 1.0e-4
    max_grad_norm = 1.0e9

    class RecipeBackedLauncherTrainer:
        def __init__(self, args: Any) -> None:
            self.args = args
            model = driver.load_v2_model(device=device, dtype=dtype).train()
            model.set_node_executors(build_trainer_node_executors(model))
            self.trainer = build_minimal_omni_trainer(model, device=device, dtype=dtype)
            self.trainer.base.args.train.optimizer = SimpleNamespace(max_grad_norm=max_grad_norm)
            self.trainer.base.state = SimpleNamespace(global_step=0)
            self.trainer.base.optimizer = _build_multi_optimizer(model, lr=lr)
            self.trainer.base.lr_scheduler = _build_multi_scheduler(self.trainer.base.optimizer)
            self.batch = batch
            report.update(
                {
                    "modules_preloaded": isinstance(args.model.modules, dict),
                    "module_count": len(model.modules_dict),
                    "training_edge_count": len(args.load_omni_config().training_edges),
                    "global_batch_size": args.train.global_batch_size,
                    "micro_batch_size": args.train.micro_batch_size,
                    "fsdp_mode": args.train.accelerator.fsdp_config.fsdp_mode,
                    "wandb_enabled": args.train.wandb.enable,
                }
            )

        def train(self) -> None:
            events = _run_trainer_train_step(
                driver,
                self.trainer.base.model,
                self.batch,
                seed=driver.case.model.seed,
                device=device,
                dtype=dtype,
                lr=1.0e-4,
                max_grad_norm=1.0e9,
            )
            report.update(
                {
                    "train_called": True,
                    "loss": events["loss"],
                    "global_step": int(events["global_step"]),
                    "zero_grad_passes": _all_grads_are_cleared(self.trainer.base.model),
                }
            )

    old_argv = sys.argv[:]
    original_trainer = omni_trainer_module.OmniTrainer
    sys.argv = [
        str(launcher_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.model.v2_model.model_root),
        "--train.global_batch_size",
        "1",
        "--train.micro_batch_size",
        "1",
        "--train.max_steps",
        "1",
        "--train.seed",
        str(driver.case.model.seed),
        "--train.optimizer.lr",
        str(lr),
        "--train.optimizer.max_grad_norm",
        str(max_grad_norm),
        "--train.wandb.enable",
        "false",
        "--train.checkpoint.output_dir",
        str(output_dir),
        "--train.checkpoint.save_steps",
        "0",
        "--train.checkpoint.save_hf_weights",
        "false",
        "--accelerator.fsdp_config.fsdp_mode",
        "ddp",
        "--accelerator.fsdp_config.mixed_precision.enable",
        "false",
        "--train.gradient_checkpointing.enable",
        "false",
        "--data.dataloader.num_workers",
        "0",
        "--data.dataloader.drop_last",
        "false",
    ]
    # Patch argv and OmniTrainer only inside the smoke so the production launcher
    # parses config normally while the training body stays deterministic.
    omni_trainer_module.OmniTrainer = RecipeBackedLauncherTrainer
    try:
        runpy.run_path(str(launcher_path), run_name="__main__")
    finally:
        omni_trainer_module.OmniTrainer = original_trainer
        sys.argv = old_argv
    return report


def _run_v2_train_fsdp2(
    driver: Any,
    batch_kwargs: Mapping[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    dtype: torch.dtype,
    nproc_per_node: int,
    timeout: int,
    lr: float,
    max_grad_norm: float,
    num_micro_steps: int,
    collect_parameter_samples: bool,
    dp_replicate_size: int | None,
    dp_shard_size: int | None,
    fsdp_mode: str,
    init_device: str,
    mixed_precision: bool,
    require_fsdp_modules: bool,
) -> Mapping[str, Any]:
    """Run the FSDP2 worker and load its rank-0 ``results.json`` report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "fsdp2_payload.pt"
    torch.save(
        {
            "batch_kwargs": dict(batch_kwargs),
            "seed": driver.case.model.seed,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "lr": lr,
            "max_grad_norm": max_grad_norm,
            "num_micro_steps": num_micro_steps,
            "collect_parameter_samples": collect_parameter_samples,
            "require_fsdp_modules": require_fsdp_modules,
        },
        payload_path,
    )
    port = _find_free_port()
    worker_path = Path("tests/seed_omni/parity_suite/v2/workers/train_fsdp2_worker.py")
    # Distributed policies run in a child process so FSDP process groups and
    # device state are isolated from the parent pytest process.
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(worker_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.model.v2_model.model_root),
        "--train.global_batch_size",
        str(nproc_per_node),
        "--train.micro_batch_size",
        "1",
        "--train.max_steps",
        "1",
        "--train.wandb.enable",
        "false",
        "--train.checkpoint.output_dir",
        str(output_dir),
        "--train.checkpoint.save_steps",
        "0",
        "--train.checkpoint.save_hf_weights",
        "false",
        "--train.checkpoint.hf_save_steps",
        "0",
        "--accelerator.fsdp_config.fsdp_mode",
        fsdp_mode,
        "--train.init_device",
        init_device,
        "--accelerator.fsdp_config.mixed_precision.enable",
        str(mixed_precision).lower(),
        "--train.gradient_checkpointing.enable",
        "false",
        "--data.dataloader.num_workers",
        "0",
        "--data.dataloader.drop_last",
        "false",
    ]
    if dp_replicate_size is not None:
        cmd.extend(["--accelerator.dp_replicate_size", str(dp_replicate_size)])
    if dp_shard_size is not None:
        cmd.extend(["--accelerator.dp_shard_size", str(dp_shard_size)])
    env = {
        **os.environ,
        "VEOMNI_PARITY_FSDP2_PAYLOAD": str(payload_path),
        "VEOMNI_PARITY_FSDP2_OUTPUT_DIR": str(output_dir),
    }
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stdout_chunks: list[str] = []
        deadline = time.monotonic() + timeout
        try:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_chunks.append(line)
                log_file.write(line)
                log_file.flush()
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report_path = output_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stdout = "".join(stdout_chunks)
    report["exit_code"] = returncode
    report["log_path"] = str(log_path)
    report["stdout_tail"] = stdout[-4000:]
    report["stderr_tail"] = ""
    return report


def _run_v2_train_data_loss_smoke(
    driver: Any,
    *,
    config_path: Path,
    output_dir: Path,
    nproc_per_node: int,
    steps: int,
    timeout: int,
    gradient_checkpointing: bool,
    loss_window: int,
    expect_loss_decrease: bool,
    data_source_names: tuple[str, ...],
) -> Mapping[str, Any]:
    """Run the data/loss worker and load its rank-0 ``results.json`` report."""

    data_config = dict(driver.case.recipe.data or {})
    data_config = _filter_data_sources(data_config, data_source_names)
    data_path = _materialize_recipe_data_config(data_config, output_dir)
    port = _find_free_port()
    worker_path = Path("tests/seed_omni/parity_suite/v2/workers/train_data_loss_worker.py")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(worker_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.model.v2_model.model_root),
        "--train.global_batch_size",
        str(nproc_per_node),
        "--train.micro_batch_size",
        "1",
        "--train.max_steps",
        str(steps),
        "--train.wandb.enable",
        "false",
        "--train.checkpoint.output_dir",
        str(output_dir),
        "--train.checkpoint.save_steps",
        "0",
        "--train.checkpoint.save_hf_weights",
        "false",
        "--train.checkpoint.hf_save_steps",
        "0",
        "--accelerator.fsdp_config.fsdp_mode",
        "fsdp2",
        "--train.gradient_checkpointing.enable",
        str(gradient_checkpointing).lower(),
        "--data.train_path",
        str(data_path),
        "--data.dataloader.num_workers",
        "0",
        "--data.dataloader.drop_last",
        "false",
    ]
    for key in ("data_type", "datasets_type", "multisource_datasets_type", "max_seq_len", "train_sample"):
        if key in data_config:
            cmd.extend([f"--data.{key}", str(data_config[key])])
    env = {
        **os.environ,
        "VEOMNI_PARITY_DATA_LOSS_OUTPUT_DIR": str(output_dir),
        "VEOMNI_PARITY_DATA_LOSS_WINDOW": str(loss_window),
        "VEOMNI_PARITY_DATA_LOSS_EXPECT_DECREASE": str(expect_loss_decrease).lower(),
    }
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stdout_chunks: list[str] = []
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            assert process.stdout is not None
            while True:
                if process.poll() is not None:
                    for line in process.stdout:
                        stdout_chunks.append(line)
                        log_file.write(line)
                        sys.__stdout__.write(line)
                    log_file.flush()
                    sys.__stdout__.flush()
                    returncode = int(process.returncode)
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    returncode = 124
                    raise subprocess.TimeoutExpired(cmd, timeout)
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                stdout_chunks.append(line)
                log_file.write(line)
                log_file.flush()
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            timed_out = True
            returncode = 124
    report_path = output_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stdout = "".join(stdout_chunks)
    report.setdefault("all_pass", False)
    report["exit_code"] = returncode
    report["timed_out"] = timed_out
    report["data_sources"] = _data_source_names(data_config)
    report["log_path"] = str(log_path)
    report["stdout_tail"] = stdout[-4000:]
    report["stderr_tail"] = ""
    return report


def _filter_data_sources(data_config: Mapping[str, Any], source_names: tuple[str, ...]) -> dict[str, Any]:
    filtered = dict(data_config)
    if not source_names:
        return filtered
    train_path = filtered.get("train_path")
    if not isinstance(train_path, Mapping):
        raise ValueError("data_source_names can only filter mapping-style data.train_path.")
    names = list(train_path.get("names") or ())
    sources = list(train_path.get("sources") or ())
    if len(names) != len(sources):
        raise ValueError("data.train_path names and sources must have the same length.")
    wanted = set(source_names)
    indexes = [index for index, name in enumerate(names) if str(name) in wanted]
    missing = sorted(wanted - {str(names[index]) for index in indexes})
    if missing:
        raise ValueError(f"Requested unknown data source name(s): {missing}.")
    next_train_path = dict(train_path)
    next_train_path["names"] = [names[index] for index in indexes]
    next_train_path["sources"] = [sources[index] for index in indexes]
    next_schedule = []
    for schedule in train_path.get("schedule", []) or []:
        next_item = dict(schedule)
        weights = list(next_item.get("weights") or [])
        if len(weights) == len(names):
            selected_weights = [float(weights[index]) for index in indexes]
            total = sum(selected_weights)
            next_item["weights"] = (
                [weight / total for weight in selected_weights] if total > 0 else [1.0 / len(indexes)] * len(indexes)
            )
        else:
            next_item["weights"] = [1.0 / len(indexes)] * len(indexes)
        next_schedule.append(next_item)
    next_train_path["schedule"] = next_schedule
    filtered["train_path"] = next_train_path
    return filtered


def _data_source_names(data_config: Mapping[str, Any]) -> list[str]:
    train_path = data_config.get("train_path")
    if isinstance(train_path, Mapping):
        return [str(name) for name in train_path.get("names", []) or []]
    return [str(train_path)] if train_path is not None else []


def _write_data_loss_manifest(driver: Any, *, report: Mapping[str, Any], output_dir: Path) -> None:
    manifest_path = Path("outputs") / "parity_suite" / "data_loss_smoke_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"runs": {}}
    losses = report.get("losses")
    manifest["runs"][driver.case.node_id] = {
        "case_id": driver.case.node_id,
        "data_sources": list(report.get("data_sources", [])),
        "all_pass": bool(report.get("all_pass", False)),
        "exit_code": int(report.get("exit_code", -1)),
        "timed_out": bool(report.get("timed_out", False)),
        "steps": int(report.get("steps", 0) or 0),
        "loss_decreased": bool(report.get("loss_decreased", False)),
        "first_loss": float(losses[0]) if isinstance(losses, list) and losses else None,
        "last_loss": float(losses[-1]) if isinstance(losses, list) and losses else None,
        "output_dir": str(output_dir),
        "log_path": str(report.get("log_path", output_dir / "run.log")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_recipe_data_config(data_config: Mapping[str, Any], output_dir: Path) -> Path | str:
    train_path = data_config.get("train_path")
    if isinstance(train_path, Mapping):
        path = output_dir / "data.yaml"
        path.write_text(yaml.safe_dump(dict(train_path), sort_keys=False), encoding="utf-8")
        return path
    if train_path is None:
        raise ValueError(f"{data_config!r} must include data.train_path for data_loss_smoke.")
    return str(train_path)


def _policy_output_dir(driver: Any, name: str) -> Path:
    safe_id = driver.case.node_id.replace(".", "_").replace("/", "_")
    return Path("outputs") / "parity_suite" / name / safe_id


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


@contextmanager
def _single_rank_ddp_clip_state():
    """Temporarily fake DDP state for shared grad clipping in local policies."""

    previous = parallel_state_module._PARALLEL_STATE
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(dp_mode="ddp")
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = previous


__all__ = ["run_v2_infer_framework", "run_v2_train_framework"]
