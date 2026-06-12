"""Compare BAGEL ``OmniTrainer.forward_backward_step`` against official gradient fixtures."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.seed_omni.fixtures.bagel.compare_gradient_graph import (
    _gradient_modules,
    _load_graph_config,
    _load_graph_modules,
)
from tests.seed_omni.fixtures.bagel.compare_gradient_module import (
    _collect_gradients,
    _configure_determinism,
    _gradient_targets,
    _loss_passes,
    _passes,
    _resolve_dtype,
    _tensor_metrics,
    _to_device,
)
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
    OmniTrainer,
)
from veomni.utils.device import get_dist_comm_backend, get_torch_device


def _build_minimal_trainer(model: OmniModel, *, device: torch.device, dtype: torch.dtype) -> OmniTrainer:
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
            optimizer=SimpleNamespace(max_grad_norm=1.0e9),
        )
    )
    base.preforward = BaseTrainer.preforward.__get__(base, BaseTrainer)
    base.model_fwd_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    base.model_bwd_context = nullcontext()
    trainer.base = base
    return trainer


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
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        args: SimpleNamespace,
    ) -> None:
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.model = module
        self.base.optimizer = optimizer
        self.base.lr_scheduler = lr_scheduler
        self.base.args = args
        self.callback = OmniModuleDcpCallback(self.base, name)

    def on_train_begin(self, state: TrainerState) -> None:
        self.callback.on_train_begin(state)

    def on_step_end(self, state: TrainerState, loss: float, loss_dict: dict[str, float], grad_norm: float) -> None:
        self.callback.on_step_end(state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)


@contextmanager
def _single_rank_ddp_clip_state():
    previous = parallel_state_module._PARALLEL_STATE
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(dp_mode="ddp")
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = previous


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


def _build_multi_optimizer(
    graph_modules: dict[str, torch.nn.Module],
    *,
    lr: float,
) -> MultiOptimizer:
    return MultiOptimizer(
        {
            name: torch.optim.SGD(module.parameters(), lr=lr)
            for name, module in graph_modules.items()
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


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_param(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    data = param.detach().cpu()
    if rows is not None:
        return data[rows.detach().cpu().to(dtype=torch.long)]
    if data.dim() >= 2:
        return data[:4, :4]
    return data[:16]


def _param(module: torch.nn.Module, name: str) -> torch.nn.Parameter:
    params = dict(module.named_parameters())
    try:
        return params[name]
    except KeyError as exc:
        raise KeyError(f"Missing parameter {name!r}") from exc


def _collect_parameter_samples(
    modules: dict[str, torch.nn.Module],
    expected_gradients: dict[str, Any],
) -> dict[str, torch.Tensor]:
    actual: dict[str, torch.Tensor] = {}
    gradient_modules = _gradient_modules(modules)
    for name, expected in expected_gradients.items():
        module = gradient_modules[expected["v2_module"]]
        actual[name] = _sample_param(_param(module, expected["v2_param"]), expected.get("rows"))
    return actual


def _selected_grads_are_cleared(
    modules: dict[str, torch.nn.Module],
    expected_gradients: dict[str, Any],
) -> bool:
    gradient_modules = _gradient_modules(modules)
    for expected in expected_gradients.values():
        module = gradient_modules[expected["v2_module"]]
        if _param(module, expected["v2_param"]).grad is not None:
            return False
    return True


def _checkpoint_args(
    output_dir: Path, *, load_path: Path | None = None, max_grad_norm: float = 1.0e9
) -> SimpleNamespace:
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
    graph_modules: dict[str, torch.nn.Module],
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

    trainer.base.optimizer = _build_multi_optimizer(graph_modules, lr=lr)
    trainer.base.lr_scheduler = _build_multi_scheduler(trainer.base.optimizer)
    trainer.base.checkpointer_callback = OmniGlobalStateCallback(trainer)

    trainer.module_trainers = {}
    for name, module in graph_modules.items():
        optimizer = trainer.base.optimizer.optimizers[name]
        scheduler = trainer.base.lr_scheduler.schedulers[name]
        trainer.module_trainers[name] = _CheckpointModuleTrainer(name, module, optimizer, scheduler, args)


def _checkpoint_on_train_begin(trainer: OmniTrainer) -> None:
    with _single_rank_ddp_clip_state():
        trainer.base.checkpointer_callback.on_train_begin(trainer.base.state)
        for module_trainer in trainer.module_trainers.values():
            module_trainer.on_train_begin(trainer.base.state)


def _run_checkpoint_train_step(trainer: OmniTrainer, batch: dict[str, Any], *, seed: int) -> dict[str, Any]:
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
    _configure_determinism(seed)
    with _single_rank_ddp_clip_state():
        trainer.train_step(iter([[{"bagel_packed_batch": batch}]]))
    return events


def _run_direct_optimizer_scheduler_step(
    model: OmniModel,
    graph_modules: dict[str, torch.nn.Module],
    batch: dict[str, Any],
    *,
    seed: int,
    dtype: torch.dtype,
    lr: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    optimizer = _build_multi_optimizer(graph_modules, lr=lr)
    scheduler = _build_multi_scheduler(optimizer)
    device = batch["packed_text_ids"].device
    autocast_context = (
        torch.amp.autocast("cuda", enabled=True, dtype=dtype)
        if device.type == "cuda" and dtype != torch.float32
        else nullcontext()
    )

    _configure_determinism(seed)
    with autocast_context:
        outputs = model(bagel_packed_batch=batch)
        loss = outputs["loss"]
    if loss is None:
        raise RuntimeError("BAGEL direct train-step fixture produced no V2 loss.")
    loss.backward()

    with _single_rank_ddp_clip_state():
        grad_norm = veomni_clip_grad_norm(model, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return {
        "loss": loss.detach().cpu(),
        "loss_dict": {name: value.detach().cpu() for name, value in outputs.get("losses", {}).items()},
        "grad_norm": grad_norm,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def _run_trainer_optimizer_scheduler_step(
    trainer: OmniTrainer,
    batch: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    events: dict[str, Any] = {}
    trainer.on_step_begin = lambda micro_batches=None: events.setdefault("micro_batches", micro_batches)
    trainer.on_step_end = lambda loss=None, loss_dict=None, grad_norm=None: events.update(
        {"loss": loss, "loss_dict": loss_dict, "grad_norm": grad_norm}
    )

    _configure_determinism(seed)
    with _single_rank_ddp_clip_state():
        trainer.train_step(iter([[{"bagel_packed_batch": batch}]]))
    return events


def compare_optimizer_scheduler_trainer(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
    lr: float = 1.0e-4,
    max_grad_norm: float = 1.0e9,
    require_active_clip: bool = False,
) -> dict[str, Any]:
    _release_cuda_memory()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    case_id = fixture.get("metadata", {}).get("case_id")
    if case_id != "gradient_ce_mse":
        raise ValueError(f"BAGEL optimizer/scheduler trainer smoke expects gradient_ce_mse, got {case_id!r}")

    seed = int(fixture["metadata"].get("seed", 1234))
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    tolerance = fixture["tolerances"][fixture["metadata"]["dtype"]]
    gradients = fixture["gradients"]

    direct_batch = _to_device(fixture["prepared"], torch_device)
    direct_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
    direct_model = OmniModel(_load_graph_config(config_dir), direct_modules).train()
    direct_report = _run_direct_optimizer_scheduler_step(
        direct_model,
        direct_modules,
        direct_batch,
        seed=seed,
        dtype=torch_dtype,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )
    direct_parameters = _collect_parameter_samples(direct_modules, gradients)
    direct_loss = direct_report["loss"]
    direct_grad_norm = direct_report["grad_norm"]
    direct_lrs = direct_report["scheduler"].get_last_lr()
    del direct_report, direct_model, direct_modules, direct_batch
    _release_cuda_memory()

    trainer_batch = _to_device(fixture["prepared"], torch_device)
    trainer_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
    trainer_model = OmniModel(_load_graph_config(config_dir), trainer_modules).train()
    trainer = _build_minimal_trainer(trainer_model, device=torch_device, dtype=torch_dtype)
    trainer.base.args.train.optimizer.max_grad_norm = max_grad_norm
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.optimizer = _build_multi_optimizer(trainer_modules, lr=lr)
    trainer.base.lr_scheduler = _build_multi_scheduler(trainer.base.optimizer)
    trainer_events = _run_trainer_optimizer_scheduler_step(trainer, trainer_batch, seed=seed)
    trainer_parameters = _collect_parameter_samples(trainer_modules, gradients)

    loss_metrics = _tensor_metrics(
        torch.tensor([trainer_events["loss"]]),
        direct_loss.reshape(1),
    )
    loss_metrics["passes"] = _loss_passes(loss_metrics, tolerance)

    parameter_metrics: dict[str, Any] = {}
    for name, actual in trainer_parameters.items():
        metrics = _tensor_metrics(actual, direct_parameters[name])
        metrics["passes"] = _passes(metrics, tolerance)
        parameter_metrics[name] = metrics

    trainer_lrs = trainer.base.lr_scheduler.get_last_lr()
    lr_metrics = _tensor_metrics(torch.tensor(trainer_lrs), torch.tensor(direct_lrs))
    lr_metrics["passes"] = _passes(lr_metrics, tolerance)
    grad_norm_metrics = _tensor_metrics(
        torch.tensor([float(trainer_events["grad_norm"])]),
        torch.tensor([float(direct_grad_norm)]),
    )
    grad_norm_metrics["passes"] = _passes(grad_norm_metrics, tolerance)
    scheduler_epochs = {
        name: int(scheduler.last_epoch) for name, scheduler in trainer.base.lr_scheduler.schedulers.items()
    }
    scheduler_passes = bool(all(epoch == 1 for epoch in scheduler_epochs.values()) and lr_metrics["passes"])
    zero_grad_passes = _selected_grads_are_cleared(trainer_modules, gradients)
    clipping_active = bool(
        float(trainer_events["grad_norm"]) > max_grad_norm and float(direct_grad_norm) > max_grad_norm
    )

    all_pass = bool(
        loss_metrics["passes"]
        and all(item["passes"] for item in parameter_metrics.values())
        and grad_norm_metrics["passes"]
        and scheduler_passes
        and zero_grad_passes
        and trainer.base.state.global_step == 1
        and (clipping_active or not require_active_clip)
    )
    report = {
        "case_id": case_id,
        "dtype": dtype,
        "lr": lr,
        "max_grad_norm": max_grad_norm,
        "loss": loss_metrics,
        "parameters_after_step": parameter_metrics,
        "trainer_grad_norm": trainer_events["grad_norm"],
        "direct_grad_norm": direct_grad_norm,
        "grad_norm": grad_norm_metrics,
        "clipping_active": clipping_active,
        "scheduler_lrs": lr_metrics,
        "scheduler_epochs": scheduler_epochs,
        "global_step": int(trainer.base.state.global_step),
        "zero_grad_passes": zero_grad_passes,
        "all_pass": all_pass,
    }
    del trainer, trainer_model, trainer_modules, trainer_batch, fixture
    _release_cuda_memory()
    return report


def compare_active_gradient_clipping_trainer(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
    lr: float = 1.0e-4,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    return compare_optimizer_scheduler_trainer(
        fixture_path,
        model_root,
        config_dir=config_dir,
        device=device,
        dtype=dtype,
        lr=lr,
        max_grad_norm=max_grad_norm,
        require_active_clip=True,
    )


def compare_checkpoint_save_resume_trainer(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    output_dir: Path = Path("outputs/bagel_v2/checkpoint_smoke"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
    lr: float = 1.0e-4,
    max_grad_norm: float = 1.0e9,
) -> dict[str, Any]:
    _release_cuda_memory()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    case_id = fixture.get("metadata", {}).get("case_id")
    if case_id != "gradient_ce_mse":
        raise ValueError(f"BAGEL checkpoint trainer smoke expects gradient_ce_mse, got {case_id!r}")

    seed = int(fixture["metadata"].get("seed", 1234))
    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    gradients = fixture["gradients"]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _single_rank_process_group():
        save_batch = _to_device(fixture["prepared"], torch_device)
        save_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
        save_model = OmniModel(_load_graph_config(config_dir), save_modules).train()
        save_trainer = _build_minimal_trainer(save_model, device=torch_device, dtype=torch_dtype)
        _attach_checkpoint_state(
            save_trainer,
            save_modules,
            output_dir=output_dir,
            load_path=None,
            lr=lr,
            max_grad_norm=max_grad_norm,
        )
        save_events = _run_checkpoint_train_step(save_trainer, save_batch, seed=seed)
        saved_parameters = _collect_parameter_samples(save_modules, gradients)
        saved_lrs = save_trainer.base.lr_scheduler.get_last_lr()

        checkpoint_root = output_dir / "checkpoints" / "global_step_1"
        module_dirs = {name: checkpoint_root / name for name in save_modules}
        trainer_state_path = checkpoint_root / "trainer_state.pt"
        saved_layout = {
            "trainer_state": trainer_state_path.exists(),
            "modules": {name: path.exists() and any(path.iterdir()) for name, path in module_dirs.items()},
        }

        del save_trainer, save_model, save_modules, save_batch
        _release_cuda_memory()

        resume_batch = _to_device(fixture["prepared"], torch_device)
        resume_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
        resume_model = OmniModel(_load_graph_config(config_dir), resume_modules).train()
        resume_trainer = _build_minimal_trainer(resume_model, device=torch_device, dtype=torch_dtype)
        _attach_checkpoint_state(
            resume_trainer,
            resume_modules,
            output_dir=output_dir,
            load_path=checkpoint_root,
            lr=lr,
            max_grad_norm=max_grad_norm,
        )
        _checkpoint_on_train_begin(resume_trainer)
        resumed_parameters = _collect_parameter_samples(resume_modules, gradients)
        resume_lrs = resume_trainer.base.lr_scheduler.get_last_lr()
        resume_global_step = int(resume_trainer.base.state.global_step)

        parameter_metrics: dict[str, Any] = {}
        tolerance = fixture["tolerances"][fixture["metadata"]["dtype"]]
        for name, actual in resumed_parameters.items():
            metrics = _tensor_metrics(actual, saved_parameters[name])
            metrics["passes"] = _passes(metrics, tolerance)
            parameter_metrics[name] = metrics

        lr_metrics = _tensor_metrics(torch.tensor(resume_lrs), torch.tensor(saved_lrs))
        lr_metrics["passes"] = _passes(lr_metrics, tolerance)

        resume_events = _run_checkpoint_train_step(resume_trainer, resume_batch, seed=seed)
        final_global_step = int(resume_trainer.base.state.global_step)

        all_pass = bool(
            saved_layout["trainer_state"]
            and all(saved_layout["modules"].values())
            and resume_global_step == 1
            and final_global_step == 2
            and lr_metrics["passes"]
            and all(item["passes"] for item in parameter_metrics.values())
        )
        report = {
            "case_id": case_id,
            "dtype": dtype,
            "output_dir": str(output_dir),
            "checkpoint_root": str(checkpoint_root),
            "saved_layout": saved_layout,
            "resume_global_step": resume_global_step,
            "final_global_step": final_global_step,
            "save_loss": save_events["loss"],
            "resume_loss": resume_events["loss"],
            "parameters_after_resume": parameter_metrics,
            "scheduler_lrs_after_resume": lr_metrics,
            "all_pass": all_pass,
        }
        del resume_trainer, resume_model, resume_modules, resume_batch, fixture
        _release_cuda_memory()
        return report


def compare_gradient_trainer(
    fixture_path: Path,
    model_root: Path,
    *,
    config_dir: Path = Path("configs/seed_omni/Bagel/bagel_7b_mot"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "bf16",
) -> dict[str, Any]:
    _release_cuda_memory()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    case_id = fixture.get("metadata", {}).get("case_id")
    if not isinstance(case_id, str) or not case_id.startswith("gradient_"):
        raise ValueError(f"Unsupported BAGEL gradient fixture case: {case_id!r}")

    torch_device = torch.device(device)
    torch_dtype = _resolve_dtype(dtype)
    _configure_determinism(int(fixture["metadata"].get("seed", 1234)))
    tolerance = fixture["tolerances"][fixture["metadata"]["dtype"]]
    batch = _to_device(fixture["prepared"], torch_device)
    graph_modules = _load_graph_modules(model_root, device=torch_device, dtype=torch_dtype)
    model = OmniModel(_load_graph_config(config_dir), graph_modules).train()
    trainer = _build_minimal_trainer(model, device=torch_device, dtype=torch_dtype)

    loss, loss_dict = trainer.forward_backward_step({"bagel_packed_batch": batch})
    if loss is None:
        raise RuntimeError("BAGEL trainer-level gradient fixture produced no V2 loss.")

    loss_metrics = _tensor_metrics(loss.detach().cpu().reshape(1), fixture["losses"]["total"].reshape(1))
    loss_metrics["passes"] = _loss_passes(loss_metrics, tolerance)

    actual_gradients = _collect_gradients(_gradient_modules(graph_modules), fixture["gradients"])
    expected_gradients = _gradient_targets(fixture["gradients"])
    gradient_metrics: dict[str, Any] = {}
    for name, actual in actual_gradients.items():
        metrics = _tensor_metrics(actual, expected_gradients[name])
        metrics["passes"] = _passes(metrics, tolerance)
        gradient_metrics[name] = metrics

    loss_dict_metrics: dict[str, Any] = {}
    for name, value in loss_dict.items():
        loss_dict_metrics[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": float(value.detach().float().cpu().item()) if value.numel() == 1 else None,
        }

    all_pass = bool(loss_metrics["passes"] and all(item["passes"] for item in gradient_metrics.values()))
    report = {
        "case_id": case_id,
        "dtype": dtype,
        "tolerance": tolerance,
        "loss": loss_metrics,
        "loss_dict": loss_dict_metrics,
        "gradients": gradient_metrics,
        "all_pass": all_pass,
    }
    del trainer, model, graph_modules, batch, fixture
    _release_cuda_memory()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/seed_omni/Bagel/bagel_7b_mot"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_gradient_trainer(
        args.fixture,
        args.model_root,
        config_dir=args.config_dir,
        device=args.device,
        dtype=args.dtype,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
