"""BAGEL parity driver for the real official oracle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.bagel import requests
from tests.seed_omni.bagel.reference_model import run_reference_only_recipe
from tests.seed_omni.bagel.transformers import BagelConfig
from tests.seed_omni.parity_suite.core import ParityCase, ParityReport, ProbeMapping, sample_named_param
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.contract import canonical_from_reference_output
from tests.seed_omni.parity_suite.v2.tier_runners.module import InferModulePolicy
from veomni.models.seed_omni.modeling_omni import OmniModel
from veomni.models.seed_omni.modules.bagel.training_pack import packed_label_rows


def create_driver(case: ParityCase) -> BagelParityDriver:
    return BagelParityDriver(case)


class BagelParityDriver(ParityDriver):
    """Own BAGEL-specific reference/V2 execution wiring."""

    def build_text_und_request(self, ctx):
        return requests.build_text_und_request(ctx)

    def build_text_image_und_request(self, ctx):
        return requests.build_text_image_und_request(ctx)

    def build_image_gen_request(self, ctx):
        return requests.build_image_gen_request(ctx)

    def build_image_edit_request(self, ctx):
        return requests.build_image_edit_request(ctx)

    def build_train_request(self, ctx):
        return requests.build_train_request(ctx)

    def generation_kwargs(self, model_or_config: Any, reference_output: Any) -> dict[str, Any]:
        kwargs = super().generation_kwargs(model_or_config, reference_output)
        if self._v2_request_kind() not in {"image_gen", "image_edit"}:
            return kwargs
        canonical = canonical_from_reference_output(reference_output)
        latent_input = canonical.get("latent_input")
        if isinstance(latent_input, Mapping) and "packed_init_noises" in latent_input:
            kwargs["fixed_init_noise"] = latent_input["packed_init_noises"]
        if "image_height" in canonical and "image_width" in canonical:
            kwargs["image_height"] = int(canonical["image_height"])
            kwargs["image_width"] = int(canonical["image_width"])
        return kwargs

    def v2_infer_module_policy(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        options = self.case.run.options
        max_steps = options.get("max_steps")
        selected = self.case.model.probes.for_probe_names(self.case.run.probes)
        needs_all_steps = any(mapping.step == "all" for mapping in selected)
        return InferModulePolicy(
            max_steps=None if max_steps is None else int(max_steps),
            required_nodes=frozenset() if needs_all_steps else frozenset(whitelist.keys()),
            allow_finalize=bool(options.get("allow_finalize", False)),
        )

    def reference_inputs(self) -> Mapping[str, Any]:
        inputs = dict(super().reference_inputs())
        loss_mode = self._loss_mode()
        if loss_mode is not None:
            inputs["loss_mode"] = loss_mode
        return inputs

    def reference_model_load_kwargs(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        return {
            "config": BagelConfig(),
            "init_on_meta": True,
            "torch_dtype": dtype,
            "device": device,
            "latent_patch_size": 2,
            "max_latent_size": 64,
            "timestep_shift": float(self.case.recipe.stimulus.get("timestep_shift", 3.0)),
            "vit_max_num_patch_per_side": 70,
            "connector_act": "gelu_pytorch_tanh",
        }

    def run_reference_only_recipe(self) -> ParityReport:
        return run_reference_only_recipe(
            recipe_id=self.case.recipe.id,
            run_kind=self.case.run.kind,
            case_id=self.case.node_id,
        )

    def gradient_rows(
        self,
        batch: Mapping[str, Any],
        mapping: ProbeMapping,
    ) -> torch.Tensor | None:
        if (
            mapping.v2_grad is not None
            and mapping.v2_grad.module == "bagel_text_encoder"
            and mapping.v2_grad.parameter == "lm_head.weight"
        ):
            return _maybe_packed_label_rows(batch)
        return None

    def _loss_mode(self) -> str | None:
        loss_mode = self.case.recipe.reference.get("loss_mode")
        return None if loss_mode is None else str(loss_mode)

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        label_rows = _maybe_packed_label_rows(batch)
        return {
            "qwen_early_q_proj": sample_named_param(
                model.get_module("bagel_qwen2_mot"),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            "lm_head_rows": sample_named_param(
                model.get_module("bagel_text_encoder"),
                "lm_head.weight",
                rows=label_rows,
            ),
            "flow_llm2vae": sample_named_param(
                model.get_module("bagel_flow_connector"),
                "llm2vae.weight",
            ),
        }


def _maybe_packed_label_rows(batch: Mapping[str, Any]) -> torch.Tensor | None:
    try:
        return packed_label_rows(batch.get("conversation_list"))
    except (KeyError, ValueError):
        return None


__all__ = [
    "BagelParityDriver",
    "create_driver",
]
