"""Base driver contract for model-specific parity execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import ParityCase, configure_torch_determinism, to_device
from tests.seed_omni.parity_suite.core.utilities import autocast_for_dtype, sum_losses, zero_module_grads
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer, record_module_output
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.trainer.base import BaseTrainer
from veomni.trainer.omni_trainer import OmniModuleTrainer, OmniTrainer


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
        raise NotImplementedError(f"{type(self).__name__} does not implement inference module-tier execution.")

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
        self.record_v2_train_extra_observations(model, observations, whitelist, batch=batch)
        return observations

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


__all__ = ["ParityDriver"]
