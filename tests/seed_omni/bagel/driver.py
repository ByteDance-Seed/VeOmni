"""BAGEL parity driver for the real official oracle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.bagel.reference_execution import (
    extract_train_ce_loss,
    extract_train_grad_early_q_proj,
    extract_train_grad_gen_q_proj,
    extract_train_grad_llm2vae,
    extract_train_grad_lm_head_rows,
    extract_train_mse_loss,
)
from tests.seed_omni.bagel.reference_execution import (
    run_reference as run_reference_execution,
)
from tests.seed_omni.bagel.v2_execution import (
    build_conversation_from_canonical,
    run_v2_infer_module,
)
from tests.seed_omni.parity_suite.core import ParityCase, to_device
from tests.seed_omni.parity_suite.core.utilities import sample_named_grad
from tests.seed_omni.parity_suite.driver import ParityDriver
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.loader import load_reference_model
from tests.seed_omni.parity_suite.v2.model import load_omni_config_from_dir, load_omni_module_from_pretrained
from tests.seed_omni.parity_suite.v2.observation import record_module_output
from veomni.models.seed_omni.modeling_omni import OmniModel


SUPPORTED_DRIVER_CASES = frozenset({"text_und", "image_gen", "image_edit", "train_ce_mse"})


def create_driver(case: ParityCase) -> BagelParityDriver:
    return BagelParityDriver(case)


def extract_hidden_state(context: ReferenceCaptureContext) -> torch.Tensor:
    return context.output["reference"]["hidden_state"]


def extract_greedy_token(context: ReferenceCaptureContext) -> torch.Tensor:
    return context.output["reference"]["greedy_token"]


def extract_image_velocity(context: ReferenceCaptureContext) -> torch.Tensor:
    return context.output["reference"]["velocity"]


def extract_image_x_t(context: ReferenceCaptureContext) -> torch.Tensor:
    return context.output["reference"]["x_t"]


class BagelParityDriver(ParityDriver):
    """Own BAGEL-specific reference/V2 execution wiring."""

    def __init__(self, case: ParityCase) -> None:
        super().__init__(case)
        if case.scenario.driver_case not in SUPPORTED_DRIVER_CASES:
            raise NotImplementedError(f"Unsupported BAGEL driver_case: {case.scenario.driver_case!r}")

    def load_reference(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        is_train = self.case.scenario.driver_case == "train_ce_mse"
        is_image_generation = is_train or self.case.scenario.driver_case in {"image_gen", "image_edit"}
        is_image_edit = self.case.scenario.driver_case == "image_edit"
        return load_reference_model(
            self.case.model.reference,
            visual_gen=is_image_generation,
            visual_und=is_image_edit,
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
        is_image_edit = self.case.scenario.driver_case == "image_edit"
        needs_flow = is_training or self.case.scenario.driver_case in {"image_gen", "image_edit"}
        text_encoder = load_omni_module_from_pretrained(model_root / "bagel_text_encoder", device=device, dtype=dtype)
        qwen2_mot = load_omni_module_from_pretrained(model_root / "bagel_qwen2_mot", device=device, dtype=dtype)
        if needs_flow:
            flow_connector: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_flow_connector", device=device, dtype=dtype
            )
        else:
            flow_connector = _UnusedModule()
        if is_training or is_image_edit:
            siglip: nn.Module = load_omni_module_from_pretrained(
                model_root / "bagel_siglip_navit", device=device, dtype=dtype
            )
            vae: nn.Module = load_omni_module_from_pretrained(model_root / "bagel_vae", device=device, dtype=dtype)
        else:
            siglip = _NoopGenerateModule()
            vae = _UnusedModule()
        modules: dict[str, nn.Module] = {
            "bagel_text_encoder": text_encoder.eval(),
            "bagel_siglip_navit": siglip.eval(),
            "bagel_vae": vae.eval(),
            "bagel_flow_connector": flow_connector.eval(),
            "bagel_qwen2_mot": qwen2_mot.eval(),
        }
        return OmniModel(config, modules).eval()

    def generation_kwargs(self, model_or_config: Any) -> dict[str, Any]:
        kwargs = super().generation_kwargs(model_or_config)
        if "infer_mode" in self.case.scenario.stimulus:
            kwargs["infer_mode"] = self.case.scenario.stimulus["infer_mode"]
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

    def record_v2_train_extra_observations(
        self,
        model: OmniModel,
        observations: dict[tuple[str, str], list[dict[str, Any]]],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        batch: Mapping[str, Any],
    ) -> None:
        packed_batch = batch["bagel_packed_batch"]
        gradient_records = {
            "bagel_qwen2_mot.forward": {
                "train_grad_early_q_proj": sample_named_grad(
                    model.get_module("bagel_qwen2_mot"),
                    "model.layers.0.self_attn.q_proj.weight",
                ),
                "train_grad_gen_q_proj": sample_named_grad(
                    model.get_module("bagel_qwen2_mot"),
                    "model.layers.0.self_attn.q_proj_moe_gen.weight",
                ),
            },
            "bagel_text_encoder.decode": {
                "train_grad_lm_head_rows": sample_named_grad(
                    model.get_module("bagel_text_encoder"),
                    "lm_head.weight",
                    rows=torch.unique(packed_batch["packed_label_ids"].detach().cpu()).to(dtype=torch.long),
                ),
            },
            "bagel_flow_connector.decode_velocity": {
                "train_grad_llm2vae": sample_named_grad(
                    model.get_module("bagel_flow_connector"),
                    "llm2vae.weight",
                ),
            },
        }
        for node, out in gradient_records.items():
            record_module_output(observations, whitelist, state="train", node=node, out=out)

    @torch.no_grad()
    def run_v2_infer_module(
        self,
        reference_output: Mapping[str, Any],
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        model = self.load_v2_model(device=device, dtype=dtype)
        generation_kwargs = self.generation_kwargs(model)
        return run_v2_infer_module(
            model,
            self.case.scenario.driver_case,
            reference_output,
            whitelist,
            generation_kwargs=generation_kwargs,
            device=device,
        )


class _NoopGenerateModule(nn.Module):
    def generate(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


class _UnusedModule(nn.Module):
    def encode(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}

    def embed_latent(self, conversation_list: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"conversation_list": conversation_list}


__all__ = [
    "BagelParityDriver",
    "SUPPORTED_DRIVER_CASES",
    "create_driver",
    "extract_greedy_token",
    "extract_hidden_state",
    "extract_image_velocity",
    "extract_image_x_t",
    "extract_train_ce_loss",
    "extract_train_grad_early_q_proj",
    "extract_train_grad_gen_q_proj",
    "extract_train_grad_lm_head_rows",
    "extract_train_grad_llm2vae",
    "extract_train_mse_loss",
]
