"""Default framework-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import (
    ParityReport,
    ProbeReport,
    RunCaptureOptions,
    zero_module_grads,
)
from tests.seed_omni.parity_suite.v2.cpu_preprocess import apply_training_cpu_preprocessors
from tests.seed_omni.parity_suite.v2.model import load_graph_active_omni_config
from tests.seed_omni.parity_suite.v2.tier_runners.framework_support import (
    TrainerStepOptions,
    all_grads_are_cleared,
    build_minimal_omni_trainer,
    build_trainer_node_executors,
    framework_report,
    fsdp2_numeric_reports,
    optional_int,
    policy_output_dir,
    release_cuda_memory,
    run_direct_train_step,
    run_trainer_train_step,
    trainer_step_reports,
)
from tests.seed_omni.parity_suite.v2.workers.framework_policy_runner import (
    attach_checkpoint_state,
    checkpoint_layout,
    checkpoint_on_train_begin,
    run_checkpoint_train_step,
    run_data_loss_smoke_policy,
    run_launcher_smoke,
    run_optimizer_trajectory,
    run_script_data_smoke_policy,
    run_v2_infer_eager,
    run_v2_infer_fsdp2,
    run_v2_train_fsdp2,
    single_rank_process_group,
)


def run_v2_train_framework(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> dict[str, Any] | ParityReport:
    """Run training framework tier through mapping comparison or policy reports."""

    del capture_options
    run = driver.case.run
    if run.kind == "forward_backward":
        return _run_v2_train_framework_batch(
            driver,
            driver.v2_request_kwargs(reference_output, device=device),
            whitelist,
            device=device,
            dtype=dtype,
        )

    release_cuda_memory()
    try:
        policy = _TRAIN_FRAMEWORK_POLICIES.get(run.kind)
        if policy is None:
            raise NotImplementedError(f"Unsupported training framework kind {run.kind!r}.")
        return policy(driver, reference_output, device=device, dtype=dtype)
    finally:
        release_cuda_memory()


def run_v2_infer_framework(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> dict[str, Any] | ParityReport:
    """Run inference framework tier through distributed policy reports."""

    del capture_options
    run = driver.case.run
    release_cuda_memory()
    try:
        policy = _INFER_FRAMEWORK_POLICIES.get(run.kind)
        if policy is None:
            raise NotImplementedError(f"Unsupported inference framework kind {run.kind!r}.")
        return policy(driver, reference_output, device=device, dtype=dtype)
    finally:
        release_cuda_memory()


def _run_v2_train_framework_batch(
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
    apply_training_cpu_preprocessors(model, batch)
    trainer = build_minimal_omni_trainer(model, device=device, dtype=dtype)
    loss, loss_dict = trainer.forward_backward_step(batch)
    if loss is None:
        raise RuntimeError(f"{type(driver).__name__} V2 train framework tier produced no loss.")
    observations = driver.collect_v2_train_framework_observations(model, loss_dict, whitelist, batch=batch)
    return {"observations": observations, "ctx": {"loss": loss, "losses": loss_dict}, "trace": ["train:framework"]}


def _run_train_step_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = TrainerStepOptions.from_options(driver.case.run.options)
    direct_batch = driver.v2_request_kwargs(reference_output, device=device)
    trainer_batch = driver.v2_request_kwargs(reference_output, device=device)

    # Compare the production trainer step against a direct optimizer step using
    # fresh models with identical seeds, batches, optimizer, and scheduler.
    driver.configure_determinism(driver.case.model.seed)
    direct_model = driver.load_v2_model(device=device, dtype=dtype).train()
    direct_result = run_direct_train_step(
        direct_model,
        direct_batch,
        seed=driver.case.model.seed,
        dtype=dtype,
        lr=options.lr,
        max_grad_norm=options.max_grad_norm,
    )
    direct_parameters = driver.sample_v2_framework_parameters(direct_model, direct_batch)
    del direct_model, direct_batch
    release_cuda_memory()

    driver.configure_determinism(driver.case.model.seed)
    trainer_model = driver.load_v2_model(device=device, dtype=dtype).train()
    trainer_model.set_node_executors(build_trainer_node_executors(trainer_model))
    trainer_result = run_trainer_train_step(
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
    zero_grad_passes = all_grads_are_cleared(trainer_model)
    reports = trainer_step_reports(
        driver,
        trainer_result=trainer_result,
        direct_result=direct_result,
        trainer_parameters=trainer_parameters,
        direct_parameters=direct_parameters,
        zero_grad_passes=zero_grad_passes,
        options=options,
    )
    del trainer_model, trainer_batch
    release_cuda_memory()
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def _run_checkpoint_resume_policy(
    driver: Any,
    reference_output: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    options = TrainerStepOptions.from_options(driver.case.run.options)
    output_dir = policy_output_dir(driver, "checkpoint_resume")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with single_rank_process_group():
        driver.configure_determinism(driver.case.model.seed)
        save_model = driver.load_v2_model(device=device, dtype=dtype).train()
        save_model.set_node_executors(build_trainer_node_executors(save_model))
        save_batch = driver.v2_request_kwargs(reference_output, device=device)
        save_trainer = build_minimal_omni_trainer(save_model, device=device, dtype=dtype)
        attach_checkpoint_state(
            save_trainer,
            output_dir=output_dir,
            load_path=None,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        run_checkpoint_train_step(save_trainer, save_batch, seed=driver.case.model.seed)
        saved_parameters = driver.sample_v2_framework_parameters(save_model, save_batch)
        # Reuse the post-step sampling context so resume compares the same rows
        # without keeping graph-bearing batch tensors alive across model reload.
        sample_context = getattr(driver, "framework_parameter_sample_context", lambda batch: batch)(save_batch)
        saved_lrs = save_trainer.base.lr_scheduler.get_last_lr()
        checkpoint_root = output_dir / "checkpoints" / "global_step_1"
        saved_layout = checkpoint_layout(save_model, checkpoint_root)
        del save_trainer, save_model, save_batch
        release_cuda_memory()

        driver.configure_determinism(driver.case.model.seed)
        resume_model = driver.load_v2_model(device=device, dtype=dtype).train()
        resume_model.set_node_executors(build_trainer_node_executors(resume_model))
        resume_batch = driver.v2_request_kwargs(reference_output, device=device)
        resume_trainer = build_minimal_omni_trainer(resume_model, device=device, dtype=dtype)
        attach_checkpoint_state(
            resume_trainer,
            output_dir=output_dir,
            load_path=checkpoint_root,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        checkpoint_on_train_begin(resume_trainer)
        resumed_parameters = driver.sample_v2_framework_parameters(resume_model, sample_context)
        resume_lrs = resume_trainer.base.lr_scheduler.get_last_lr()
        resume_global_step = int(resume_trainer.base.state.global_step)
        run_checkpoint_train_step(resume_trainer, resume_batch, seed=driver.case.model.seed)
        final_global_step = int(resume_trainer.base.state.global_step)

    reports = [
        framework_report(driver, "framework.checkpoint_layout", saved_layout, True, "exact"),
        framework_report(driver, "framework.resume_global_step", resume_global_step, 1, "exact"),
        framework_report(driver, "framework.final_global_step", final_global_step, 2, "exact"),
        framework_report(
            driver,
            "framework.scheduler_lrs_after_resume",
            torch.tensor(resume_lrs),
            torch.tensor(saved_lrs),
            "exact",
        ),
        framework_report(
            driver, "framework.parameters_after_resume", resumed_parameters, saved_parameters, "gradient"
        ),
    ]
    del resume_trainer, resume_model, resume_batch, sample_context
    release_cuda_memory()
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
    config_path = Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml")))
    output_dir = policy_output_dir(driver, "launcher")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = driver.v2_request_kwargs(reference_output, device=device)
    report = run_launcher_smoke(
        driver,
        batch,
        launcher_path=launcher_path,
        config_path=config_path,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
    )
    reports = [
        framework_report(driver, "framework.launcher_train_called", report["train_called"], True, "exact"),
        framework_report(driver, "framework.launcher_global_step", report["global_step"], 1, "exact"),
        framework_report(driver, "framework.launcher_zero_grad", report["zero_grad_passes"], True, "exact"),
        framework_report(
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
    first = run_optimizer_trajectory(driver, reference_output, device=device, dtype=dtype, steps=steps, lr=lr)
    release_cuda_memory()
    second = run_optimizer_trajectory(driver, reference_output, device=device, dtype=dtype, steps=steps, lr=lr)
    reports = [
        framework_report(driver, "framework.optimizer_trajectory.losses", second["losses"], first["losses"], "loss"),
        framework_report(
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
    collect_parameter_samples = bool(options.get("collect_parameter_samples", compare_direct))
    dp_replicate_size = optional_int(options.get("dp_replicate_size"))
    dp_shard_size = optional_int(options.get("dp_shard_size"))
    output_dir = policy_output_dir(driver, "distributed_train_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_kwargs = driver.v2_request_kwargs(reference_output, device=torch.device("cpu"))
    baseline_report: Mapping[str, Any] | None = None
    if compare_direct:
        baseline_report = run_v2_train_fsdp2(
            driver,
            batch_kwargs,
            config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
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
    report = run_v2_train_fsdp2(
        driver,
        batch_kwargs,
        config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
        output_dir=output_dir / "target",
        dtype=dtype,
        nproc_per_node=nproc,
        timeout=int(options.get("timeout", 900)),
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_micro_steps=num_micro_steps,
        collect_parameter_samples=collect_parameter_samples,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        fsdp_mode="fsdp2",
        init_device="meta",
        mixed_precision=False,
        require_fsdp_modules=True,
    )
    reports = [
        framework_report(driver, "framework.fsdp2_exit_code", report["exit_code"], 0, "exact"),
        framework_report(driver, "framework.fsdp2_all_pass", bool(report.get("all_pass", False)), True, "exact"),
        framework_report(driver, "framework.fsdp2_dp_mode", report.get("dp_mode"), "fsdp2", "exact"),
        framework_report(driver, "framework.fsdp2_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        framework_report(
            driver,
            "framework.fsdp2_num_micro_steps",
            int(report.get("num_micro_steps", 0)),
            num_micro_steps,
            "exact",
        ),
    ]
    if dp_replicate_size is not None:
        reports.append(
            framework_report(
                driver,
                "framework.fsdp2_dp_replicate_size",
                int(report.get("dp_replicate_size", 0)),
                dp_replicate_size,
                "exact",
            )
        )
    if dp_shard_size is not None:
        reports.append(
            framework_report(
                driver,
                "framework.fsdp2_dp_shard_size",
                int(report.get("dp_shard_size", 0)),
                dp_shard_size,
                "exact",
            )
        )
    if collect_parameter_samples:
        reports.append(
            framework_report(
                driver,
                "framework.fsdp2_parameter_samples_collected",
                bool(report.get("parameters_after_step", {})),
                True,
                "exact",
            )
        )
    if compare_direct:
        if baseline_report is None:
            raise RuntimeError("FSDP2 numeric policy did not build a baseline report.")
        reports.extend(
            fsdp2_numeric_reports(
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
    output_dir = policy_output_dir(driver, "distributed_infer_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_kwargs = driver.v2_request_kwargs(reference_output, device=torch.device("cpu"))
    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    generation_kwargs = driver.generation_kwargs(config, reference_output)
    probe_names = tuple(str(name) for name in options.get("compare_probes", ()) or driver.case.run.probes)
    baseline_report: Mapping[str, Any] | None = None
    if compare_eager:
        baseline_report = run_v2_infer_eager(
            driver,
            request_kwargs,
            generation_kwargs=generation_kwargs,
            config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
            modules_config=Path(
                str(
                    options.get(
                        "baseline_modules_config",
                        driver.case.v2_model.config_dir / "modules_infer_eager.yaml",
                    )
                )
            ),
            infer_type=str(options.get("infer_type", driver.case.graph.name)),
            output_dir=output_dir / "baseline_eager",
            dtype=dtype,
            timeout=int(options.get("timeout", 1800)),
            probe_names=probe_names,
        )
    report = run_v2_infer_fsdp2(
        driver,
        request_kwargs,
        generation_kwargs=generation_kwargs,
        config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
        modules_config=Path(
            str(
                options.get(
                    "modules_config",
                    driver.case.v2_model.config_dir / "modules_infer_fsdp.yaml",
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
        framework_report(driver, "framework.infer_fsdp2_exit_code", report["exit_code"], 0, "exact"),
        framework_report(driver, "framework.infer_fsdp2_all_pass", bool(report.get("all_pass", False)), True, "exact"),
        framework_report(driver, "framework.infer_fsdp2_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        framework_report(
            driver,
            "framework.infer_fsdp2_has_fsdp_modules",
            int(report.get("fsdp_module_count", 0)) > 0,
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.infer_fsdp2_trace_has_image_flow",
            bool(report.get("trace_has_image_flow", False)),
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.infer_fsdp2_trace_has_prompt_encode",
            bool(report.get("trace_has_prompt_encode", False)),
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.infer_fsdp2_rank0_finalized",
            bool(report.get("rank0_finalized", False)),
            True,
            "exact",
        ),
        framework_report(
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
        framework_report(
            driver,
            "framework.infer_fsdp2_baseline_exit_code",
            baseline_report["exit_code"],
            0,
            "exact",
        ),
        framework_report(
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
            framework_report(
                driver,
                f"framework.infer_fsdp2_{probe_name.replace('.', '_')}",
                actual,
                expected,
                tol,
            )
        )
    return reports


_TRAIN_FRAMEWORK_POLICIES = {
    "train_step": _run_train_step_policy,
    "checkpoint_resume": _run_checkpoint_resume_policy,
    "launcher": _run_launcher_policy,
    "optimizer_trajectory": _run_optimizer_trajectory_policy,
    "distributed_train": _run_distributed_train_policy,
    "data_loss_smoke": run_data_loss_smoke_policy,
    "script_data_smoke": run_script_data_smoke_policy,
}

_INFER_FRAMEWORK_POLICIES = {
    "distributed_infer": _run_distributed_infer_policy,
}


__all__ = ["run_v2_infer_framework", "run_v2_train_framework"]
