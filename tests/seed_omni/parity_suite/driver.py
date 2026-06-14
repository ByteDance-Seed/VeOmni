"""Base driver contract for model-specific parity execution."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import (
    ParityCase,
    ParityReport,
    ProbeMapping,
    ProbeReport,
    compare_values,
    configure_torch_determinism,
    to_device,
    tolerance_from_policy,
)
from tests.seed_omni.parity_suite.core.utilities import (
    autocast_for_dtype,
    sample_named_grad,
    sum_losses,
    zero_module_grads,
)
from tests.seed_omni.parity_suite.reference.capture import (
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    capture_reference_taps,
)
from tests.seed_omni.parity_suite.v2.infer_fsm import InferModulePolicy, run_infer_module_fsm
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer, record_module_output
from veomni.distributed import parallel_state as parallel_state_module
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.trainer.base import BaseTrainer
from veomni.trainer.omni_trainer import MultiLRScheduler, MultiOptimizer, OmniModuleTrainer, OmniTrainer


@dataclass(frozen=True)
class _TrainerStepOptions:
    lr: float
    max_grad_norm: float
    require_active_clip: bool

    @classmethod
    def from_policy(cls, options: Mapping[str, Any]) -> _TrainerStepOptions:
        return cls(
            lr=float(options.get("lr", 1.0e-4)),
            max_grad_norm=float(options.get("max_grad_norm", 1.0e9)),
            require_active_clip=bool(options.get("require_active_clip", False)),
        )


class ParityDriver(ABC):
    """Model-specific execution contract used by the shared parity runner."""

    generation_defaults: Mapping[str, Any] = {
        "max_new_tokens": 1,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    def __init__(self, case: ParityCase) -> None:
        self.case = case

    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def configure_determinism(self, seed: int) -> None:
        configure_torch_determinism(seed)

    def reference_inputs(self) -> Mapping[str, Any]:
        return self.case.scenario.stimulus

    def generation_kwargs(self, model_or_config: Any) -> dict[str, Any]:
        config = getattr(model_or_config, "config", model_or_config)
        kwargs = dict(getattr(config, "generation_kwargs", None) or {})
        for key, default in self.generation_defaults.items():
            kwargs[key] = self.case.scenario.stimulus.get(key, default)
        return kwargs

    @abstractmethod
    def load_reference(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Load the independent reference oracle."""

    @abstractmethod
    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        """Load the V2 model under test."""

    @abstractmethod
    def run_reference(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> Any:
        """Run the reference recipe and return driver-owned canonical output."""

    def run_v2_infer_graph(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through ``OmniModel.generate``."""

        model = self.load_v2_model(device=device, dtype=dtype)
        request = self.v2_infer_request(reference_output, device=device)
        generation_kwargs = self.generation_kwargs(model)
        trace: list[str] = []
        with torch.no_grad(), arm_generation_observer(whitelist) as observations:
            ctx = model.generate(request, trace=trace, generation_kwargs=dict(generation_kwargs))
        return {"observations": dict(observations), "ctx": ctx, "trace": trace}

    def run_v2_train_graph(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through ``OmniModel.forward``."""

        return self.run_v2_train_graph_batch(
            self.v2_train_batch_kwargs(reference_output, device=device),
            whitelist,
            device=device,
            dtype=dtype,
        )

    def run_v2_infer_module(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through direct FSM module steps."""

        model = self.load_v2_model(device=device, dtype=dtype)
        request = self.v2_infer_request(reference_output, device=device)
        generation_kwargs = self.generation_kwargs(model)
        return run_infer_module_fsm(
            model,
            request,
            whitelist,
            generation_kwargs=generation_kwargs,
            policy=self.v2_infer_module_policy(reference_output, whitelist),
        )

    def run_v2_train_module(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph one node at a time."""

        return self.run_v2_train_module_batch(
            self.v2_train_batch_kwargs(reference_output, device=device),
            whitelist,
            device=device,
            dtype=dtype,
        )

    def run_v2_infer_framework(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not implement inference framework-tier execution.")

    def run_v2_train_framework(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training batch through ``OmniTrainer.forward_backward_step``."""

        return self.run_v2_train_framework_batch(
            self.v2_train_batch_kwargs(reference_output, device=device),
            whitelist,
            device=device,
            dtype=dtype,
        )

    def run_framework_policy(self, *, device: torch.device, dtype: torch.dtype) -> ParityReport:
        """Run a scenario-declared framework behavior beyond ``forward_backward_step``."""

        policy = self.case.scenario.framework_policy
        if policy is None:
            raise RuntimeError(f"{type(self).__name__} framework policy runner requires scenario.framework.")
        if policy.kind == "trainer_step":
            return self.run_v2_train_step_policy(device=device, dtype=dtype)
        raise NotImplementedError(f"{type(self).__name__} does not implement framework policy {policy.kind!r}.")

    def run_v2_train_step_policy(self, *, device: torch.device, dtype: torch.dtype) -> ParityReport:
        """Compare direct graph optimization with ``OmniTrainer.train_step``."""

        options = _TrainerStepOptions.from_policy(self.case.scenario.framework_policy.options)
        reference = capture_reference_taps(
            reference_factory=lambda: self.load_reference(device=device, dtype=dtype),
            driver=self,
            inputs=self.reference_inputs(),
            plan=ReferenceCapturePlan(),
        )
        direct_batch = self.v2_train_batch_kwargs(reference.run_output, device=device)
        trainer_batch = self.v2_train_batch_kwargs(reference.run_output, device=device)

        self.configure_determinism(self.case.model.seed)
        direct_model = self.load_v2_model(device=device, dtype=dtype).train()
        direct_result = _run_direct_train_step(
            direct_model,
            direct_batch,
            seed=self.case.model.seed,
            dtype=dtype,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        direct_parameters = self.sample_v2_framework_parameters(direct_model, direct_batch)
        del direct_model, direct_batch
        _release_cuda_memory()

        self.configure_determinism(self.case.model.seed)
        trainer_model = self.load_v2_model(device=device, dtype=dtype).train()
        trainer_model.set_node_executors(self.build_trainer_node_executors(trainer_model))
        trainer_result = _run_trainer_train_step(
            self,
            trainer_model,
            trainer_batch,
            seed=self.case.model.seed,
            device=device,
            dtype=dtype,
            lr=options.lr,
            max_grad_norm=options.max_grad_norm,
        )
        trainer_parameters = self.sample_v2_framework_parameters(trainer_model, trainer_batch)
        zero_grad_passes = _all_grads_are_cleared(trainer_model)
        reports = _trainer_step_reports(
            self,
            trainer_result=trainer_result,
            direct_result=direct_result,
            trainer_parameters=trainer_parameters,
            direct_parameters=direct_parameters,
            zero_grad_passes=zero_grad_passes,
            options=options,
        )
        del trainer_model, trainer_batch, reference
        _release_cuda_memory()
        return ParityReport(case_id=self.case.node_id, probes=tuple(reports))

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del model, batch
        return {}

    def v2_infer_request(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into an ``OmniModel.generate`` request."""

        del reference_output, device
        raise NotImplementedError(f"{type(self).__name__} does not implement inference request adaptation.")

    def v2_train_batch_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        """Adapt driver-owned canonical output into kwargs for ``OmniModel.forward``."""

        if not isinstance(reference_output, Mapping):
            raise TypeError(f"{type(self).__name__} expected a mapping reference output.")
        canonical = reference_output.get("canonical")
        if isinstance(canonical, Mapping) and isinstance(canonical.get("train_batch"), Mapping):
            return dict(to_device(canonical["train_batch"], device))
        raise NotImplementedError(f"{type(self).__name__} does not implement training batch adaptation.")

    def v2_infer_module_policy(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        """Return the module-tier FSM policy for inference parity."""

        del reference_output, whitelist
        policy = self.case.scenario.module_policy
        return InferModulePolicy(
            max_steps=policy.max_steps,
            required_nodes=frozenset(policy.required_nodes),
            allow_finalize=policy.allow_finalize,
        )

    def run_v2_train_graph_batch(
        self,
        batch_kwargs: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        model = self.load_v2_model(device=device, dtype=dtype).train()
        zero_module_grads(model.modules_dict.values())
        batch = dict(batch_kwargs)
        with torch.enable_grad(), autocast_for_dtype(device, dtype):
            forward_result = model(**batch)
            loss = forward_result["loss"]
            if loss is None:
                raise RuntimeError(f"{type(self).__name__} V2 train graph produced no loss.")
        loss.backward()
        observations = self.collect_v2_train_observations(model, forward_result, whitelist, batch=batch)
        return {"observations": observations, "ctx": forward_result, "trace": ["train:graph"]}

    def run_v2_train_module_batch(
        self,
        batch_kwargs: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        model = self.load_v2_model(device=device, dtype=dtype).train()
        zero_module_grads(model.modules_dict.values())
        batch = dict(batch_kwargs)
        with torch.enable_grad(), autocast_for_dtype(device, dtype):
            forward_result = self.run_v2_train_nodes(model, batch)
            loss = forward_result["loss"]
            if loss is None:
                raise RuntimeError(f"{type(self).__name__} V2 train module tier produced no loss.")
        loss.backward()
        observations = self.collect_v2_train_observations(model, forward_result, whitelist, batch=batch)
        return {"observations": observations, "ctx": forward_result, "trace": ["train:module"]}

    def run_v2_train_framework_batch(
        self,
        batch_kwargs: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        model = self.load_v2_model(device=device, dtype=dtype).train()
        model.set_node_executors(self.build_trainer_node_executors(model))
        zero_module_grads(model.modules_dict.values())
        batch = dict(batch_kwargs)
        trainer = self.build_minimal_omni_trainer(model, device=device, dtype=dtype)
        loss, loss_dict = trainer.forward_backward_step(batch)
        if loss is None:
            raise RuntimeError(f"{type(self).__name__} V2 train framework tier produced no loss.")
        observations = self.collect_v2_train_framework_observations(model, loss_dict, whitelist, batch=batch)
        return {"observations": observations, "ctx": {"loss": loss, "losses": loss_dict}, "trace": ["train:framework"]}

    def run_v2_train_nodes(self, model: OmniModel, batch: dict[str, Any]) -> dict[str, Any]:
        if model.training_graph is None:
            raise RuntimeError(f"{type(self).__name__} train module tier requires a training graph.")
        node_outputs: dict[str, dict[str, Any]] = {}
        losses: dict[str, torch.Tensor] = {}
        for node_name in model.training_graph.execution_order:
            module_name = model.training_graph.module_of(node_name)
            method = model.training_graph.method_of(node_name)
            module = model.get_module(module_name)
            kwargs = model.training_graph.collect_inputs(node_name, node_outputs, batch)
            call_kwargs = module.pre_forward(method, **kwargs)
            fn = module if method == "forward" else getattr(module, method)
            outputs = fn(**call_kwargs)
            out = module.post_forward(method, **outputs)
            node_outputs[node_name] = out
            for key, value in out.items():
                if key in batch:
                    batch[key] = value
            if "_loss" in out:
                losses[node_name] = out["_loss"]
        return {"loss": sum_losses(losses), "losses": losses, "outputs": node_outputs}

    def collect_v2_train_observations(
        self,
        model: OmniModel,
        forward_result: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node, out in forward_result["outputs"].items():
            record_module_output(observations, whitelist, state="train", node=node, out=out)
        self.record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self.record_v2_train_extra_observations(model, observations, whitelist, batch=batch)
        return observations

    def collect_v2_train_framework_observations(
        self,
        model: OmniModel,
        loss_dict: Mapping[str, torch.Tensor],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        observations: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node, loss in loss_dict.items():
            record_module_output(observations, whitelist, state="train", node=node, out={"_loss": loss})
        self.record_v2_train_gradient_observations(model, observations, whitelist, batch=batch)
        self.record_v2_train_extra_observations(model, observations, whitelist, batch=batch)
        return observations

    def record_v2_train_gradient_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        for mapping in self.v2_train_gradient_mappings():
            fields = whitelist.get(("train", mapping.node), frozenset())
            if mapping.v2_field not in fields or mapping.v2_grad is None:
                continue
            rows = _rows_from_batch(batch, mapping.v2_grad.rows_from)
            out = {
                mapping.v2_field: sample_named_grad(
                    model.get_module(mapping.v2_grad.module),
                    mapping.v2_grad.parameter,
                    rows=rows,
                )
            }
            record_module_output(observations, whitelist, state="train", node=mapping.node, out=out)

    def v2_train_gradient_mappings(self) -> tuple[ProbeMapping, ...]:
        selected = self.case.model.mapping.for_probe_names(self.case.scenario.probes)
        return tuple(mapping for mapping in selected if mapping.v2_grad is not None)

    def record_v2_train_extra_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        del model, observations, whitelist, batch

    def build_minimal_omni_trainer(self, model: OmniModel, *, device: torch.device, dtype: torch.dtype) -> OmniTrainer:
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
        return trainer

    def build_trainer_node_executors(self, model: OmniModel) -> dict[str, Any]:
        executors: dict[str, Any] = {}
        for name, module in model.modules_dict.items():
            module_trainer = OmniModuleTrainer.__new__(OmniModuleTrainer)
            module_trainer.base = BaseTrainer.__new__(BaseTrainer)
            module_trainer.base.model = module
            executors[name] = module_trainer.forward
        return executors


def _rows_from_batch(batch: Mapping[str, Any], path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    value: Any = batch
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Unable to resolve v2_grad rows_from path {path!r}.")
        value = value[part]
    if not torch.is_tensor(value):
        raise TypeError(f"v2_grad rows_from path {path!r} must resolve to a tensor.")
    return torch.unique(value.detach().cpu()).to(dtype=torch.long)


def _run_direct_train_step(
    model: OmniModel,
    batch: Mapping[str, Any],
    *,
    seed: int,
    dtype: torch.dtype,
    lr: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    zero_module_grads(model.modules_dict.values())
    optimizer = _build_multi_optimizer(model, lr=lr)
    scheduler = _build_multi_scheduler(optimizer)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    with torch.enable_grad(), autocast_for_dtype(device, dtype):
        outputs = model(**dict(batch))
        loss = outputs["loss"]
    if loss is None:
        raise RuntimeError("Direct framework policy produced no loss.")
    loss.backward()
    with _single_rank_ddp_clip_state():
        grad_norm = veomni_clip_grad_norm(model, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return {
        "loss": loss.detach().cpu(),
        "grad_norm": grad_norm,
        "scheduler_lrs": scheduler.get_last_lr(),
        "scheduler_epochs": {name: int(scheduler.last_epoch) for name, scheduler in scheduler.schedulers.items()},
    }


def _run_trainer_train_step(
    driver: ParityDriver,
    model: OmniModel,
    batch: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    lr: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    trainer = driver.build_minimal_omni_trainer(model, device=device, dtype=dtype)
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
    driver: ParityDriver,
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


def _framework_report(driver: ParityDriver, probe: str, actual: Any, expected: Any, tol: str) -> ProbeReport:
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


@contextmanager
def _single_rank_ddp_clip_state():
    previous = parallel_state_module._PARALLEL_STATE
    parallel_state_module._PARALLEL_STATE = parallel_state_module.ParallelState(dp_mode="ddp")
    try:
        yield
    finally:
        parallel_state_module._PARALLEL_STATE = previous


__all__ = ["ParityDriver"]
