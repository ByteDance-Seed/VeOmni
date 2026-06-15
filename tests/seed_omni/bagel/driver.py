"""BAGEL parity driver for the real official oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel import reference_execution
from tests.seed_omni.bagel.v2_execution import build_conversation_from_canonical
from tests.seed_omni.parity_suite.core import ParityCase, ParityReport, to_device
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.loader import load_reference_model
from tests.seed_omni.parity_suite.v2.model import load_omni_config_from_dir, load_omni_module_from_pretrained
from tests.seed_omni.parity_suite.v2.module import InferModulePolicy
from veomni.models.seed_omni.modeling_omni import OmniModel


@dataclass(frozen=True)
class BagelRecipe:
    reference_case: str
    loss_mode: str | None = None
    training: bool = False
    visual_gen: bool = False
    visual_und: bool = False
    needs_siglip: bool = False
    needs_flow: bool = False
    needs_vae: bool = False


RECIPES: dict[str, BagelRecipe] = {
    "train_ce_mse": BagelRecipe("train_ce_mse", loss_mode="ce_mse", training=True, visual_gen=True),
    "train_ce": BagelRecipe("train_ce_mse", loss_mode="ce_only", training=True),
    "train_text_image_ce": BagelRecipe("train_ce_mse", loss_mode="text_image_ce", training=True, visual_und=True),
    "train_mse": BagelRecipe("train_ce_mse", loss_mode="mse_only", training=True, visual_gen=True),
    "text_und": BagelRecipe("text_und"),
    "text_image_und": BagelRecipe("text_image_und", visual_und=True, needs_siglip=True),
    "image_gen": BagelRecipe("image_gen", visual_gen=True, needs_flow=True),
    "image_edit": BagelRecipe(
        "image_edit",
        visual_gen=True,
        visual_und=True,
        needs_siglip=True,
        needs_flow=True,
        needs_vae=True,
    ),
    "image_edit_text_und": BagelRecipe("text_und"),
    "interleave_text_und": BagelRecipe("text_und"),
    "interleave_text_image_und": BagelRecipe("text_image_und", visual_und=True, needs_siglip=True),
    "interleave_image_gen": BagelRecipe("image_gen", visual_gen=True, needs_flow=True),
    "transformers_reference_smoke": BagelRecipe("transformers_reference_smoke"),
}
SUPPORTED_RECIPES = frozenset(RECIPES)


def create_driver(case: ParityCase) -> BagelParityDriver:
    return BagelParityDriver(case)


class BagelParityDriver(ParityDriver):
    """Own BAGEL-specific reference/V2 execution wiring."""

    def __init__(self, case: ParityCase) -> None:
        super().__init__(case)
        if case.recipe.id not in SUPPORTED_RECIPES:
            raise NotImplementedError(f"Unsupported BAGEL recipe: {case.recipe.id!r}")

    def reference_inputs(self) -> Mapping[str, Any]:
        inputs = dict(super().reference_inputs())
        recipe = self._recipe()
        if recipe.loss_mode is not None:
            inputs["loss_mode"] = recipe.loss_mode
        return inputs

    def load_reference(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        recipe = self._recipe()
        return load_reference_model(
            self.case.model.reference,
            visual_gen=recipe.visual_gen,
            visual_und=recipe.visual_und,
            init_on_meta=True,
            torch_dtype=dtype,
            device=device,
            latent_patch_size=2,
            max_latent_size=64,
            timestep_shift=float(self.case.recipe.stimulus.get("timestep_shift", 3.0)),
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
        )

    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        model_root = self.case.model.v2_model.model_root
        if model_root is None:
            raise ValueError("BAGEL V2 model_root is required.")
        recipe = self._recipe()
        graph = None if recipe.training else self.case.graph.name
        config = load_omni_config_from_dir(self.case.model.v2_model.config_dir, graph=graph)
        text_encoder = load_omni_module_from_pretrained(model_root / "bagel_text_encoder", device=device, dtype=dtype)
        qwen2_mot = load_omni_module_from_pretrained(model_root / "bagel_qwen2_mot", device=device, dtype=dtype)
        if recipe.training or recipe.needs_flow:
            flow_connector: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_flow_connector", device=device, dtype=dtype
            )
        else:
            flow_connector = _UnusedModule()
        if recipe.training or recipe.needs_siglip:
            siglip: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_siglip_navit", device=device, dtype=dtype
            )
        else:
            siglip = _NoopGenerateModule()
        if recipe.training or recipe.needs_vae or bool(self.case.recipe.stimulus.get("enable_decode_smoke", False)):
            vae: nn.Module = load_omni_module_from_pretrained(model_root / "bagel_vae", device=device, dtype=dtype)
        else:
            vae = _UnusedModule()
        modules: dict[str, nn.Module] = {
            "bagel_text_encoder": text_encoder.eval(),
            "bagel_siglip_navit": siglip.eval(),
            "bagel_vae": vae.eval(),
            "bagel_flow_connector": flow_connector.eval(),
            "bagel_qwen2_mot": qwen2_mot.eval(),
        }
        return OmniModel(config, modules).eval()

    def _recipe(self) -> BagelRecipe:
        return RECIPES[self.case.recipe.id]

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

    @torch.no_grad()
    def run_reference_recipe(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> dict[str, Any]:
        return reference_execution.run_reference_recipe(self._recipe().reference_case, ref_model, inputs, context)

    def run_reference_only_recipe(self) -> ParityReport:
        return reference_execution.run_reference_only_recipe(
            recipe_id=self.case.recipe.id,
            run_kind=self.case.run.kind,
            case_id=self.case.node_id,
        )

    def v2_infer_request(self, reference_output: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
        return {"conversation_list": build_conversation_from_canonical(reference_output["canonical"], device=device)}

    def v2_train_batch_kwargs(self, reference_output: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
        return {"bagel_packed_batch": to_device(reference_output["canonical"]["train_batch"], device)}

    def v2_infer_module_policy(
        self,
        reference_output: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
    ) -> InferModulePolicy:
        return super().v2_infer_module_policy(reference_output, whitelist)

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        packed_batch = _packed_batch(batch)
        label_rows = torch.unique(packed_batch["packed_label_ids"].detach().cpu()).to(dtype=torch.long)
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


def _sample_param(module: nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    value = dict(module.named_parameters())[name].detach().cpu()
    if rows is not None:
        return value[rows]
    if value.dim() >= 2:
        return value[:4, :4]
    return value[:16]


def _packed_batch(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    packed = batch.get("bagel_packed_batch")
    if isinstance(packed, Mapping):
        return packed
    return batch


class _NoopGenerateModule(nn.Module):
    def generate(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


class _UnusedModule(nn.Module):
    def encode(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}

    def decode(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}

    def embed_latent(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


__all__ = [
    "BagelRecipe",
    "BagelParityDriver",
    "RECIPES",
    "SUPPORTED_RECIPES",
    "create_driver",
]
