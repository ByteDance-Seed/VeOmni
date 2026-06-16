"""BAGEL parity driver for the real official oracle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel import requests
from tests.seed_omni.bagel.reference_model import run_reference_only_recipe
from tests.seed_omni.bagel.transformers import BagelConfig
from tests.seed_omni.parity_suite.core import ParityCase, ParityReport, sample_named_grad
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.v2.observation import record_module_output
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

    def generation_kwargs(self, model_or_config: Any) -> dict[str, Any]:
        kwargs = super().generation_kwargs(model_or_config)
        for key in (
            "infer_mode",
            "max_flow_steps",
            "num_timesteps",
            "timestep_shift",
            "latent_downsample",
            "cfg_text_scale",
            "cfg_img_scale",
            "cfg_interval",
            "cfg_renorm_min",
            "cfg_renorm_type",
            "enable_taylorseer",
        ):
            if key in self.case.recipe.stimulus:
                kwargs[key] = self.case.recipe.stimulus[key]
        return kwargs

    def run_reference_only_recipe(self) -> ParityReport:
        return run_reference_only_recipe(
            recipe_id=self.case.recipe.id,
            run_kind=self.case.run.kind,
            case_id=self.case.node_id,
        )

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
            "qwen_early_q_proj": _sample_param(
                model.get_module("bagel_qwen2_mot"),
                "model.layers.0.self_attn.q_proj.weight",
            ),
            "lm_head_rows": _sample_param(
                model.get_module("bagel_text_encoder"),
                "lm_head.weight",
                rows=label_rows,
            ),
            "flow_llm2vae": _sample_param(
                model.get_module("bagel_flow_connector"),
                "llm2vae.weight",
            ),
        }

    def record_v2_train_gradient_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        label_rows: torch.Tensor | None = None
        for mapping in self.v2_train_gradient_mappings():
            fields = whitelist.get(("train", mapping.node), frozenset())
            if mapping.v2_field not in fields or mapping.v2_grad is None:
                continue
            rows = None
            if mapping.v2_grad.module == "bagel_text_encoder" and mapping.v2_grad.parameter == "lm_head.weight":
                label_rows = _maybe_packed_label_rows(batch) if label_rows is None else label_rows
                rows = label_rows
            out = {
                mapping.v2_field: sample_named_grad(
                    model.get_module(mapping.v2_grad.module),
                    mapping.v2_grad.parameter,
                    rows=rows,
                )
            }
            record_module_output(observations, whitelist, state="train", node=mapping.node, out=out)


def _sample_param(module: nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    value = dict(module.named_parameters())[name].detach().cpu()
    if rows is not None:
        return value[rows]
    if value.dim() >= 2:
        return value[:4, :4]
    return value[:16]


def _maybe_packed_label_rows(batch: Mapping[str, Any]) -> torch.Tensor | None:
    try:
        return packed_label_rows(batch.get("conversation_list"))
    except (KeyError, ValueError):
        return None


__all__ = [
    "BagelParityDriver",
    "create_driver",
]
