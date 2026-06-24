"""Subprocess worker runners used by framework-tier parity policies."""

from __future__ import annotations

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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ParityReport,
    RunCaptureOptions,
    RunWorkerOptions,
    autocast_for_dtype,
    configure_torch_determinism,
    run_worker_context,
)
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext, canonical_from_reference_output
from tests.seed_omni.parity_suite.v2.tier_runners.framework_support import (
    all_grads_are_cleared,
    build_minimal_omni_trainer,
    build_multi_optimizer,
    build_multi_scheduler,
    build_trainer_node_executors,
    find_free_port,
    framework_report,
    policy_output_dir,
    run_trainer_train_step,
    single_rank_ddp_clip_state,
    single_rank_ddp_parallel_state,
)
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks import TrainerState
from veomni.trainer.omni_trainer import OmniGlobalStateCallback, OmniModuleDcpCallback, OmniTrainer
from veomni.utils.device import get_dist_comm_backend, get_torch_device


_DISTRIBUTED_LAUNCH_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _subprocess_env_without_distributed_launch() -> dict[str, str]:
    env = dict(os.environ)
    for key in _DISTRIBUTED_LAUNCH_ENV_VARS:
        env.pop(key, None)
    return env


def _worker_subprocess_env(
    driver: Any,
    extra_env: Mapping[str, str] | None = None,
    *,
    inherit_distributed_launch: bool = True,
    worker_options: RunWorkerOptions | None = None,
) -> dict[str, str]:
    base_env = dict(os.environ) if inherit_distributed_launch else _subprocess_env_without_distributed_launch()
    if worker_options is None:
        with run_worker_context(driver.case.run.options) as resolved_options:
            worker_options = resolved_options
    return {
        **base_env,
        **worker_options.env(),
        **dict(extra_env or {}),
    }


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


def _infer_worker_argv(
    driver: Any,
    *,
    config_path: Path,
    modules_config: Path,
    infer_type: str,
    output_dir: Path,
    worker_path: Path,
) -> list[str]:
    return [
        str(worker_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.v2_model.model_root),
        "--infer.model_path",
        str(driver.case.v2_model.model_root),
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


def _run_infer_worker_subprocess(
    cmd: list[str],
    *,
    env: Mapping[str, str],
    output_dir: Path,
    timeout: int,
) -> Mapping[str, Any]:
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            env=dict(env),
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
            _terminate_process_group(process)
            raise
        except BaseException:
            _terminate_process_group(process)
            raise
    report_path = output_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stdout = "".join(stdout_chunks)
    report["exit_code"] = returncode
    report["log_path"] = str(log_path)
    report["stdout_tail"] = stdout[-4000:]
    report["stderr_tail"] = ""
    return report


def run_v2_infer_eager(
    driver: Any,
    request_kwargs: Mapping[str, Any],
    *,
    generation_kwargs: Mapping[str, Any],
    config_path: Path,
    modules_config: Path,
    infer_type: str,
    output_dir: Path,
    dtype: torch.dtype,
    timeout: int,
    probe_names: tuple[str, ...],
) -> Mapping[str, Any]:
    """Run the inference worker as a single-process eager baseline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "infer_eager_payload.pt"
    torch.save(
        {
            "request_kwargs": dict(request_kwargs),
            "generation_kwargs": dict(generation_kwargs),
            "probe_names": probe_names,
            "probes_path": str(driver.case.model.root / "probes.yaml"),
            "seed": driver.case.model.seed,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "require_fsdp_modules": False,
        },
        payload_path,
    )
    worker_path = Path("tests/seed_omni/parity_suite/v2/workers/infer_fsdp2_worker.py")
    cmd = [
        sys.executable,
        *_infer_worker_argv(
            driver,
            config_path=config_path,
            modules_config=modules_config,
            infer_type=infer_type,
            output_dir=output_dir,
            worker_path=worker_path,
        ),
    ]
    env = _worker_subprocess_env(
        driver,
        {
            "VEOMNI_PARITY_INFER_FSDP2_PAYLOAD": str(payload_path),
            "VEOMNI_PARITY_INFER_FSDP2_OUTPUT_DIR": str(output_dir),
        },
        inherit_distributed_launch=False,
    )
    return _run_infer_worker_subprocess(cmd, env=env, output_dir=output_dir, timeout=timeout)


def run_v2_infer_fsdp2(
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
            "probes_path": str(driver.case.model.root / "probes.yaml"),
            "seed": driver.case.model.seed,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "require_fsdp_modules": require_fsdp_modules,
        },
        payload_path,
    )
    port = find_free_port()
    worker_path = Path("tests/seed_omni/parity_suite/v2/workers/infer_fsdp2_worker.py")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        *_infer_worker_argv(
            driver,
            config_path=config_path,
            modules_config=modules_config,
            infer_type=infer_type,
            output_dir=output_dir,
            worker_path=worker_path,
        ),
    ]
    env = _worker_subprocess_env(
        driver,
        {
            "VEOMNI_PARITY_INFER_FSDP2_PAYLOAD": str(payload_path),
            "VEOMNI_PARITY_INFER_FSDP2_OUTPUT_DIR": str(output_dir),
        },
    )
    return _run_infer_worker_subprocess(cmd, env=env, output_dir=output_dir, timeout=timeout)


def run_data_loss_smoke_policy(
    driver: Any,
    reference_output: Any = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    del reference_output, device, dtype
    options = driver.case.run.options
    strategy = str(options.get("strategy", "fsdp2"))
    if strategy != "fsdp2":
        raise NotImplementedError(f"Unsupported data_loss_smoke strategy {strategy!r}.")

    nproc = int(options.get("nproc_per_node", 8))
    steps = int(options.get("steps", 100))
    loss_window = int(options.get("loss_window", 10))
    output_dir = policy_output_dir(driver, "data_loss_smoke_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with run_worker_context(options) as worker_options:
        report = _run_v2_train_data_loss_smoke(
            driver,
            config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
            output_dir=output_dir,
            nproc_per_node=nproc,
            steps=steps,
            timeout=int(options.get("timeout", 7200)),
            gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
            loss_window=loss_window,
            expect_loss_decrease=bool(options.get("expect_loss_decrease", True)),
            worker_options=worker_options,
            data_source_names=tuple(str(name) for name in options.get("data_source_names", ()) or ()),
        )
    _write_data_loss_manifest(driver, report=report, output_dir=output_dir)
    reports = [
        framework_report(driver, "framework.data_loss_exit_code", report["exit_code"], 0, "exact"),
        framework_report(driver, "framework.data_loss_all_pass", bool(report.get("all_pass", False)), True, "exact"),
        framework_report(driver, "framework.data_loss_dp_mode", report.get("dp_mode"), "fsdp2", "exact"),
        framework_report(driver, "framework.data_loss_rank_count", len(report.get("ranks", [])), nproc, "exact"),
        framework_report(
            driver,
            "framework.data_loss_steps",
            int(report.get("steps", 0)),
            steps,
            "exact",
        ),
        framework_report(
            driver,
            "framework.data_loss_gradient_checkpointing",
            bool(report.get("gradient_checkpointing", False)),
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.data_loss_decreased",
            bool(report.get("loss_decreased", False)),
            True,
            "exact",
        ),
    ]
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


def run_script_data_smoke_policy(
    driver: Any,
    reference_output: Any = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ParityReport:
    del reference_output, device, dtype
    options = driver.case.run.options
    strategy = str(options.get("strategy", "fsdp2"))
    if strategy != "fsdp2":
        raise NotImplementedError(f"Unsupported script_data_smoke strategy {strategy!r}.")

    nproc = int(options.get("nproc_per_node", 8))
    steps = int(options.get("steps", 1))
    output_dir = policy_output_dir(driver, "script_data_smoke_fsdp2")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _run_v2_train_script_data_smoke(
        driver,
        config_path=Path(str(options.get("config_path", driver.case.v2_model.config_dir / "base.yaml"))),
        launcher_path=Path(str(options.get("launcher_path", "tasks/omni/train_omni.py"))),
        output_dir=output_dir,
        nproc_per_node=nproc,
        steps=steps,
        timeout=int(options.get("timeout", 3600)),
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
    )
    reports = [
        framework_report(driver, "framework.script_data_exit_code", report["exit_code"], 0, "exact"),
        framework_report(
            driver,
            "framework.script_data_training_started",
            bool(report.get("training_started", False)),
            True,
            "exact",
        ),
        framework_report(
            driver,
            "framework.script_data_steps",
            int(report.get("steps", 0)),
            steps,
            "exact",
        ),
        framework_report(
            driver,
            "framework.script_data_gradient_checkpointing",
            bool(report.get("gradient_checkpointing", False)),
            bool(options.get("gradient_checkpointing", True)),
            "exact",
        ),
    ]
    return ParityReport(case_id=driver.case.node_id, probes=tuple(reports))


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
        self.parallel_state = single_rank_ddp_parallel_state()
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


def attach_checkpoint_state(
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
    trainer.base.optimizer = build_multi_optimizer(trainer.base.model, lr=lr)
    trainer.base.lr_scheduler = build_multi_scheduler(trainer.base.optimizer)
    trainer.base.checkpointer_callback = OmniGlobalStateCallback(trainer)
    trainer.module_trainers = {}
    for name, module in trainer.base.model.modules_dict.items():
        optimizer = trainer.base.optimizer.optimizers[name]
        scheduler = trainer.base.lr_scheduler.schedulers[name]
        trainer.module_trainers[name] = _CheckpointModuleTrainer(name, module, optimizer, scheduler, args)


def checkpoint_on_train_begin(trainer: OmniTrainer) -> None:
    with single_rank_ddp_clip_state():
        trainer.base.checkpointer_callback.on_train_begin(trainer.base.state)
        for module_trainer in trainer.module_trainers.values():
            module_trainer.on_train_begin(trainer.base.state)


def run_checkpoint_train_step(trainer: OmniTrainer, batch: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
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
    configure_torch_determinism(seed)
    with single_rank_ddp_clip_state():
        trainer.train_step(iter([[dict(batch)]]))
    return events


def checkpoint_layout(model: OmniModel, checkpoint_root: Path) -> bool:
    trainer_state_path = checkpoint_root / "trainer_state.pt"
    module_dirs = [checkpoint_root / name for name in model.modules_dict]
    return bool(trainer_state_path.exists() and all(path.exists() and any(path.iterdir()) for path in module_dirs))


@contextmanager
def single_rank_process_group():
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


def run_optimizer_trajectory(
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
    driver.configure_determinism(driver.case.model.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[torch.Tensor] = []
    parameters_after_step: list[Mapping[str, torch.Tensor]] = []
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        run_ctx = V2RunContext(
            case=driver.case,
            tier=driver.case.tier,
            domain=driver.case.graph.domain,
            reference_output=reference_output,
            canonical=canonical_from_reference_output(reference_output),
            whitelist={},
            device=device,
            dtype=dtype,
            capture_options=RunCaptureOptions(),
            purpose="optimizer_trajectory",
        )
        batch = driver.build_v2_request(run_ctx)
        with torch.enable_grad(), autocast_for_dtype(device, dtype):
            outputs = model(**dict(batch))
            loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("Optimizer trajectory produced no loss.")
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().reshape(1))
        parameters_after_step.append(driver.v2_parameter_samples(run_ctx, model, batch))
    return {"losses": losses, "parameters_after_step": parameters_after_step}


def run_launcher_smoke(
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
            driver.configure_determinism(driver.case.model.seed)
            model.set_node_executors(build_trainer_node_executors(model))
            self.trainer = build_minimal_omni_trainer(model, device=device, dtype=dtype)
            self.trainer.base.args.train.optimizer = SimpleNamespace(max_grad_norm=max_grad_norm)
            self.trainer.base.state = SimpleNamespace(global_step=0)
            self.trainer.base.optimizer = build_multi_optimizer(model, lr=lr)
            self.trainer.base.lr_scheduler = build_multi_scheduler(self.trainer.base.optimizer)
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
            events = run_trainer_train_step(
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
                    "zero_grad_passes": all_grads_are_cleared(self.trainer.base.model),
                }
            )

    old_argv = sys.argv[:]
    original_trainer = omni_trainer_module.OmniTrainer
    sys.argv = [
        str(launcher_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.v2_model.model_root),
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


def run_v2_train_fsdp2(
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
            "module_overrides": driver.case.recipe.v2_model.module_overrides,
        },
        payload_path,
    )
    port = find_free_port()
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
        str(driver.case.v2_model.model_root),
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
    env = _worker_subprocess_env(
        driver,
        {
            "VEOMNI_PARITY_FSDP2_PAYLOAD": str(payload_path),
            "VEOMNI_PARITY_FSDP2_OUTPUT_DIR": str(output_dir),
        },
    )
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
            _terminate_process_group(process)
            raise
        except BaseException:
            _terminate_process_group(process)
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
    worker_options: RunWorkerOptions,
    data_source_names: tuple[str, ...],
) -> Mapping[str, Any]:
    """Run the data/loss worker and load its rank-0 ``results.json`` report."""

    data_config = dict(driver.case.recipe.data or {})
    data_config = _filter_data_sources(data_config, data_source_names)
    data_path = _materialize_recipe_data_config(data_config, output_dir)
    port = find_free_port()
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
        str(driver.case.v2_model.model_root),
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
    env = _worker_subprocess_env(
        driver,
        {
            "VEOMNI_PARITY_DATA_LOSS_OUTPUT_DIR": str(output_dir),
            "VEOMNI_PARITY_DATA_LOSS_WINDOW": str(loss_window),
            "VEOMNI_PARITY_DATA_LOSS_EXPECT_DECREASE": str(expect_loss_decrease).lower(),
        },
        worker_options=worker_options,
    )
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
            _terminate_process_group(process)
            timed_out = True
            returncode = 124
        except BaseException:
            _terminate_process_group(process)
            raise
    report_path = output_dir / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stdout = "".join(stdout_chunks)
    report.setdefault("all_pass", False)
    report["exit_code"] = returncode
    report["timed_out"] = timed_out
    report["data_sources"] = _data_source_names(data_config)
    report["debug_log"] = worker_options.debug_log
    report["log_path"] = str(log_path)
    report["stdout_tail"] = stdout[-4000:]
    report["stderr_tail"] = ""
    return report


def _run_v2_train_script_data_smoke(
    driver: Any,
    *,
    config_path: Path,
    launcher_path: Path,
    output_dir: Path,
    nproc_per_node: int,
    steps: int,
    timeout: int,
    gradient_checkpointing: bool,
) -> Mapping[str, Any]:
    """Run the production training launcher and verify a lightweight data-backed step."""

    output_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(launcher_path),
        str(config_path),
        "--model.model_path",
        str(driver.case.v2_model.model_root),
        "--train.global_batch_size",
        str(nproc_per_node),
        "--train.micro_batch_size",
        "1",
        "--train.max_steps",
        str(steps),
        "--train.seed",
        str(driver.case.model.seed),
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
        "--data.dataloader.num_workers",
        "0",
        "--data.dataloader.drop_last",
        "false",
    ]
    log_path = output_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            env=_worker_subprocess_env(driver),
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
            _terminate_process_group(process)
            timed_out = True
            returncode = 124
        except BaseException:
            _terminate_process_group(process)
            raise
    stdout = "".join(stdout_chunks)
    training_started = "Start training" in stdout and f"Train steps: {steps}" in stdout
    return {
        "exit_code": returncode,
        "timed_out": timed_out,
        "training_started": training_started,
        "steps": steps,
        "gradient_checkpointing": gradient_checkpointing,
        "log_path": str(log_path),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": "",
    }


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


__all__ = [
    "attach_checkpoint_state",
    "checkpoint_layout",
    "checkpoint_on_train_begin",
    "run_checkpoint_train_step",
    "run_data_loss_smoke_policy",
    "run_launcher_smoke",
    "run_optimizer_trajectory",
    "run_script_data_smoke_policy",
    "run_v2_infer_eager",
    "run_v2_infer_fsdp2",
    "run_v2_train_fsdp2",
    "single_rank_process_group",
]
