"""BAGEL-specific reference observation capture for parity tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from tests.seed_omni.parity_suite.core import to_device
from tests.seed_omni.parity_suite.reference.capture import MethodPatchObservationCapture


if TYPE_CHECKING:
    from .hf_model import BagelHfModelReference


class BagelInferencerObservationCapture(MethodPatchObservationCapture):
    """Capture-only hooks for tensors produced by the vendored inferencer.

    For ``infer_gen``, ``timestep``, ``latent_query``, and ``velocity`` are
    round-level observations captured once per official ``_forward_flow`` call.
    ``denoise_hidden`` is branch-level because it is captured from each
    ``language_model.forward_inference`` call inside that round.
    """

    def __init__(self, mode: str = "und") -> None:
        super().__init__()
        self.mode = mode
        self._capture_hidden = False
        self._capture_denoise_hidden = False
        self._text_update_count = 0
        for name in ("prompt_input_ids", "vision_embeds", "mot_prefill_hidden"):
            self.ensure_field(name)
        if self.mode == "gen":
            for name in ("latent_query", "denoise_hidden", "velocity", "x_t", "timestep"):
                self.ensure_field(name)

    def configure(self, subject: BagelHfModelReference) -> None:
        model = subject.model
        device = next(model.parameters()).device

        def update_text_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                primary_update = self._text_update_count % 2 == 0
                self._text_update_count += 1
                packed_text_ids = kwargs.get("packed_text_ids")
                if primary_update and torch.is_tensor(packed_text_ids):
                    self.record("prompt_input_ids", packed_text_ids.detach().cpu())
                self._capture_hidden = primary_update
                try:
                    return original(*args, **to_device(kwargs, device))
                finally:
                    self._capture_hidden = False

            return wrapper

        def update_vit_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                connector_outputs: list[torch.Tensor] = []
                handle = model.connector.register_forward_hook(
                    lambda _module, _hook_args, output: connector_outputs.append(output.detach())
                    if torch.is_tensor(output)
                    else None
                )
                self._capture_hidden = True
                try:
                    return original(*args, **to_device(kwargs, device))
                finally:
                    handle.remove()
                    packed_vit_position_ids = kwargs.get("packed_vit_position_ids")
                    if connector_outputs and torch.is_tensor(packed_vit_position_ids):
                        pos_embeds = model.vit_pos_embed(
                            packed_vit_position_ids.to(device=model.vit_pos_embed.pos_embed.device)
                        )
                        vision_embeds = connector_outputs[-1] + pos_embeds.to(
                            device=connector_outputs[-1].device,
                            dtype=connector_outputs[-1].dtype,
                        )
                        self.record("vision_embeds", vision_embeds.detach().cpu())
                    self._capture_hidden = False

            return wrapper

        def generate_text_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return original(*args, **to_device(kwargs, device))

            return wrapper

        def forward_inference_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                output = original(*args, **kwargs)
                hidden = getattr(output, "packed_query_sequence", None)
                if self._capture_hidden and torch.is_tensor(hidden):
                    self.record("mot_prefill_hidden", hidden[-1:].detach().cpu())
                if self._capture_denoise_hidden and torch.is_tensor(hidden):
                    self.record("denoise_hidden", hidden.detach().cpu())
                return output

            return wrapper

        def prepare_vae_latent_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                output = original(*args, **to_device(kwargs, device))
                init_noise = output.get("packed_init_noises")
                if torch.is_tensor(init_noise):
                    self.record("x_t", init_noise.detach().cpu())
                return output

            return wrapper

        def forward_flow_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                moved_kwargs = to_device(kwargs, device)
                x_t = moved_kwargs.get("x_t")
                timestep = moved_kwargs.get("timestep")
                position_ids = moved_kwargs.get("packed_vae_position_ids")
                if torch.is_tensor(x_t):
                    self.record("x_t", x_t.detach().cpu())
                if torch.is_tensor(timestep):
                    unique_timestep = timestep.detach().reshape(-1).unique()
                    if unique_timestep.numel() != 1:
                        raise ValueError("BAGEL infer_gen reference expected one timestep per denoise round.")
                    self.record("timestep", unique_timestep.reshape(()).cpu())
                if torch.is_tensor(x_t) and torch.is_tensor(timestep) and torch.is_tensor(position_ids):
                    latent_query = model.vae2llm(x_t)
                    latent_query = latent_query + model.time_embedder(timestep).to(
                        device=latent_query.device,
                        dtype=latent_query.dtype,
                    )
                    latent_query = latent_query + model.latent_pos_embed(
                        position_ids.to(device=model.latent_pos_embed.pos_embed.device)
                    ).to(device=latent_query.device, dtype=latent_query.dtype)
                    latent_query = latent_query.to(dtype=model.language_model.model.embed_tokens.weight.dtype)
                    self.record("latent_query", latent_query.detach().cpu())
                self._capture_denoise_hidden = True
                try:
                    velocity = original(*args, **moved_kwargs)
                finally:
                    self._capture_denoise_hidden = False
                if torch.is_tensor(velocity):
                    self.record("velocity", velocity.detach().cpu())
                return velocity

            return wrapper

        self.patch_method(model, "forward_cache_update_text", update_text_capture)
        self.patch_method(model, "forward_cache_update_vit", update_vit_capture)
        self.patch_method(model, "generate_text", generate_text_capture)
        self.patch_method(model.language_model, "forward_inference", forward_inference_capture)
        if self.mode == "gen":
            self.patch_method(model, "prepare_vae_latent", prepare_vae_latent_capture)
            self.patch_method(model, "_forward_flow", forward_flow_capture)


class BagelTrainObservationCapture(MethodPatchObservationCapture):
    """Capture-only hooks for official BAGEL training ``Bagel.forward``.

    Training parity runs the official VAE encode followed by one packed
    ``Bagel.forward`` and ``loss.backward``. Keep this separate from inference
    capture so train probes cannot perturb the generation/cache hooks.
    """

    def __init__(self) -> None:
        super().__init__()
        for name in ("train_loss", "train_ce", "train_mse", "train_last_hidden_state", "train_velocity_pred"):
            self.ensure_field(name)

    def configure(self, subject: BagelHfModelReference) -> None:
        model = subject.model

        def language_model_forward_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                output = original(*args, **kwargs)
                if torch.is_tensor(output):
                    self.record("train_last_hidden_state", output.detach().cpu())
                return output

            return wrapper

        def llm2vae_forward_capture(original: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                output = original(*args, **kwargs)
                if torch.is_tensor(output):
                    self.record("train_velocity_pred", output.detach().cpu())
                return output

            return wrapper

        self.patch_method(model.language_model, "forward", language_model_forward_capture)
        if hasattr(model, "llm2vae"):
            self.patch_method(model.llm2vae, "forward", llm2vae_forward_capture)


__all__ = ["BagelInferencerObservationCapture", "BagelTrainObservationCapture"]
