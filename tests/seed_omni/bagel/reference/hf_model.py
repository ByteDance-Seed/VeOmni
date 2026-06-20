"""Full-model BAGEL reference subject for graph/framework parity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from tests.seed_omni.parity_suite.core import to_cpu
from tests.seed_omni.parity_suite.reference.capture import (
    NullReferenceObservationCapture,
    ReferenceCaptureContext,
    ReferenceObservationCapture,
)
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult
from tests.seed_omni.parity_suite.reference.oracles.hf_model import HfModelSubject
from tests.seed_omni.parity_suite.v2.request import conversation_list_from_specs

from .assembly import load_full_bagel_reference_bundle
from .data import (
    conversation_to_interleaved_reference_inputs,
    encode_reference_vae_latents,
    first_output_of_type,
    inferencer_generation_kwargs,
    reduce_reference_train_losses,
    reference_train_batch_from_inputs,
    train_options_from_inputs,
)
from .observation import BagelInferencerObservationCapture, BagelTrainObservationCapture
from .vendor.data.transforms import ImageTransform
from .vendor.inference import InterleaveInferencer
from .vendor.modeling.bagel import (
    Bagel,
)
from .vendor.modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer


# Full-model subject -----------------------------------------------------------


class BagelHfModelReference(HfModelSubject):
    """Composition wrapper around the vendored BAGEL inferencer."""

    def __init__(
        self,
        case: Any,
        *,
        model: Bagel,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: Mapping[str, int],
        vae_model: nn.Module | None,
        reference_device: torch.device | None = None,
    ) -> None:
        super().__init__(case)
        self.model = model
        self.tokenizer = tokenizer
        self.new_token_ids = dict(new_token_ids)
        self.vae_model = vae_model
        self.reference_device = reference_device
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=self.new_token_ids,
        )

    # Oracle metadata -----------------------------------------------------------

    @property
    def hook_root(self) -> Bagel:
        return self.model

    @property
    def owned_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.model]
        if isinstance(self.vae_model, nn.Module):
            modules.append(self.vae_model)
        return tuple(modules)

    @classmethod
    def create_hf_module_subject(cls, request: Any):
        from tests.seed_omni.bagel.reference.hf_module import BagelModuleSubject

        return BagelModuleSubject.load_reference_subject(model_subject_cls=cls, request=request)

    # Loading -------------------------------------------------------------------

    @classmethod
    def load_reference_subject(
        cls,
        *,
        case: Any,
        checkpoint: str | Path | None,
        device: torch.device,
        dtype: torch.dtype,
        torch_dtype: torch.dtype | str | None = None,
        load_weights: bool = True,
        init_on_meta: bool = True,
        **kwargs: Any,
    ) -> BagelHfModelReference:
        del kwargs
        if checkpoint is None:
            raise ValueError("BAGEL hf_model reference requires reference.hf_model.checkpoint.")
        bundle = load_full_bagel_reference_bundle(
            checkpoint=Path(checkpoint),
            device=device,
            dtype=dtype,
            torch_dtype=torch_dtype,
            load_weights=load_weights,
            init_on_meta=init_on_meta,
        )
        return cls(
            case=case,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            new_token_ids=bundle.new_token_ids,
            vae_model=bundle.vae_model,
            reference_device=bundle.device,
        )

    # Observation capture -------------------------------------------------------

    def reference_observation_capture(
        self,
        kind: str | None,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        options: Mapping[str, Any],
    ) -> ReferenceObservationCapture:
        del inputs, context, options
        observation_capture = {
            "infer_und": BagelInferencerObservationCapture(mode="und"),
            "infer_gen": BagelInferencerObservationCapture(mode="gen"),
            "infer_edit": BagelInferencerObservationCapture(mode="gen"),
            "infer_interleave": BagelInferencerObservationCapture(mode="gen"),
            "train_forward_backward": BagelTrainObservationCapture(),
        }
        return observation_capture.get(kind, NullReferenceObservationCapture())

    # Reference inference -------------------------------------------------------

    def run_reference_infer_und(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        capture: ReferenceObservationCapture | None = None,
    ) -> ReferenceRunResult:
        del capture
        return self._run_reference_infer(inputs, context, understanding_output=True)

    def run_reference_infer_gen(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        capture: ReferenceObservationCapture | None = None,
    ) -> ReferenceRunResult:
        del capture
        return self._run_reference_infer(inputs, context, understanding_output=False)

    def run_reference_infer_edit(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        capture: ReferenceObservationCapture | None = None,
    ) -> ReferenceRunResult:
        del capture
        return self._run_reference_infer(inputs, context, understanding_output=False)

    def run_reference_infer_interleave(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        capture: ReferenceObservationCapture | None = None,
    ) -> ReferenceRunResult:
        del capture
        return self._run_reference_infer(inputs, context, understanding_output=False)

    def run_reference_train_forward_backward(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        capture: ReferenceObservationCapture,
    ) -> ReferenceRunResult:
        del context
        device = self._reference_device()
        batch = reference_train_batch_from_inputs(inputs, device=device)
        batch = encode_reference_vae_latents(vae_model=self.vae_model, batch=batch)
        train_options = train_options_from_inputs(inputs)

        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss_dict = self.model(**batch)
        reduced = reduce_reference_train_losses(loss_dict, batch, **train_options)
        reduced.loss.backward()

        capture.record("train_loss", reduced.loss.detach().cpu())
        capture.record("train_ce", reduced.losses["ce"].detach().cpu())
        capture.record("train_mse", reduced.losses["mse"].detach().cpu())
        return ReferenceRunResult(
            canonical={
                "train_batch": to_cpu(batch),
                "train_kwargs": dict(train_options),
            },
            observations={},
            raw_output={
                "loss": reduced.loss.detach().cpu(),
                "losses": to_cpu(reduced.losses),
            },
        )

    def _run_reference_infer(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
        *,
        understanding_output: bool,
    ) -> ReferenceRunResult:
        del context
        device = self._reference_device()
        conversation = conversation_list_from_specs(
            inputs["conversation_list"],
            case=self.case,
            canonical=inputs,
            device=device,
        )
        if len(conversation) != 1:
            raise ValueError("BAGEL hf_model inference reference currently expects one inference sample.")

        generation_kwargs = inferencer_generation_kwargs(inputs.get("generation_kwargs", {}))
        input_list = conversation_to_interleaved_reference_inputs(conversation[0], tokenizer=self.inferencer.tokenizer)

        with torch.inference_mode():
            output_list = self.inferencer.interleave_inference(
                input_list,
                understanding_output=understanding_output,
                **generation_kwargs,
            )

        return ReferenceRunResult(
            canonical={"conversation_list": inputs["conversation_list"]},
            observations={},
            raw_output=first_output_of_type(output_list, str if understanding_output else Image.Image),
        )

    def _reference_device(self) -> torch.device:
        if self.reference_device is not None:
            return self.reference_device
        return next(self.model.parameters()).device


__all__ = ["BagelHfModelReference"]
