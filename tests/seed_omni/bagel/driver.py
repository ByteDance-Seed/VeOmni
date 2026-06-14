"""BAGEL parity driver for the real official oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel.reference_execution import (
    run_reference as run_reference_execution,
)
from tests.seed_omni.bagel.v2_execution import build_conversation_from_canonical
from tests.seed_omni.parity_suite.core import ParityCase, to_device
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.loader import load_reference_model
from tests.seed_omni.parity_suite.v2.infer_fsm import InferModulePolicy
from tests.seed_omni.parity_suite.v2.model import load_omni_config_from_dir, load_omni_module_from_pretrained
from veomni.models.seed_omni.modeling_omni import OmniModel


SUPPORTED_DRIVER_CASES = frozenset({"text_und", "text_image_und", "image_gen", "image_edit", "train_ce_mse"})
_TRAIN_CASE = "train_ce_mse"
_IMAGE_GENERATION_CASES = frozenset({"image_gen", "image_edit"})
_IMAGE_UNDERSTANDING_CASES = frozenset({"text_image_und", "image_edit"})
_FLOW_CASES = frozenset({"image_gen", "image_edit"})


@dataclass(frozen=True)
class _V2ModuleNeeds:
    siglip: bool
    flow: bool
    vae: bool


def create_driver(case: ParityCase) -> BagelParityDriver:
    return BagelParityDriver(case)


class BagelParityDriver(ParityDriver):
    """Own BAGEL-specific reference/V2 execution wiring."""

    def __init__(self, case: ParityCase) -> None:
        super().__init__(case)
        if case.scenario.driver_case not in SUPPORTED_DRIVER_CASES:
            raise NotImplementedError(f"Unsupported BAGEL driver_case: {case.scenario.driver_case!r}")

    def load_reference(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        visual_gen, visual_und = self._reference_visual_flags()
        return load_reference_model(
            self.case.model.reference,
            visual_gen=visual_gen,
            visual_und=visual_und,
            init_on_meta=True,
            torch_dtype=dtype,
            device=device,
            latent_patch_size=2,
            max_latent_size=64,
            timestep_shift=float(self.case.scenario.stimulus.get("timestep_shift", 3.0)),
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
        )

    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        model_root = self.case.model.v2_model.model_root
        if model_root is None:
            raise ValueError("BAGEL V2 model_root is required.")
        is_training = self.case.graph.domain == "training"
        graph = None if is_training else self.case.graph.name
        config = load_omni_config_from_dir(self.case.model.v2_model.config_dir, graph=graph)
        needs = self._v2_module_needs(is_training=is_training)
        text_encoder = load_omni_module_from_pretrained(model_root / "bagel_text_encoder", device=device, dtype=dtype)
        qwen2_mot = load_omni_module_from_pretrained(model_root / "bagel_qwen2_mot", device=device, dtype=dtype)
        if needs.flow:
            flow_connector: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_flow_connector", device=device, dtype=dtype
            )
        else:
            flow_connector = _UnusedModule()
        if needs.siglip:
            siglip: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_siglip_navit", device=device, dtype=dtype
            )
        else:
            siglip = _NoopGenerateModule()
        if needs.vae:
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

    def _reference_visual_flags(self) -> tuple[bool, bool]:
        driver_case = self.case.scenario.driver_case
        loss_mode = str(self.case.scenario.stimulus.get("loss_mode", "ce_mse"))
        is_train = driver_case == _TRAIN_CASE
        visual_gen = driver_case in _IMAGE_GENERATION_CASES or (is_train and loss_mode in {"ce_mse", "mse_only"})
        visual_und = driver_case in _IMAGE_UNDERSTANDING_CASES or (is_train and loss_mode == "text_image_ce")
        return visual_gen, visual_und

    def _v2_module_needs(self, *, is_training: bool) -> _V2ModuleNeeds:
        driver_case = self.case.scenario.driver_case
        return _V2ModuleNeeds(
            siglip=is_training or driver_case in _IMAGE_UNDERSTANDING_CASES,
            flow=is_training or driver_case in _FLOW_CASES,
            vae=is_training
            or driver_case == "image_edit"
            or bool(self.case.scenario.stimulus.get("enable_decode_smoke", False)),
        )

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
            if key in self.case.scenario.stimulus:
                kwargs[key] = self.case.scenario.stimulus[key]
        return kwargs

    @torch.no_grad()
    def run_reference(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> dict[str, Any]:
        return run_reference_execution(self.case.scenario.driver_case, ref_model, inputs, context)

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
    "BagelParityDriver",
    "SUPPORTED_DRIVER_CASES",
    "create_driver",
]
