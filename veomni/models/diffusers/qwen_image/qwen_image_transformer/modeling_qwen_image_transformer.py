from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import QwenImageTransformer2DModel as _QwenImageTransformer2DModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_qwen_image_transformer import QWEN_IMAGE_INIT_SIGNATURE, QwenImageTransformer2DModelConfig


@dataclass
class QwenImageModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class _QwenImageTransformerInitShim(_QwenImageTransformer2DModel):
    """Avoid constructing the large diffusers default model during PreTrainedModel init."""

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class QwenImageTransformer2DModel(PreTrainedModel, _QwenImageTransformerInitShim):
    config_class = QwenImageTransformer2DModelConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]

    def __init__(self, config: QwenImageTransformer2DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        if hasattr(self, "_internal_dict"):
            del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _QwenImageTransformer2DModel.__init__(self, **config.to_diffuser_dict())
        self.config: QwenImageTransformer2DModelConfig = config
        self.config.tie_word_embeddings = False

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            if length is None:
                return []
            return [None] * length
        if isinstance(value, list):
            return value
        if length is not None and isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == length:
            return [value[idx : idx + 1] for idx in range(length)]
        return [value]

    @staticmethod
    def _normalize_img_shapes(sample_img_shapes: Any) -> list[list[tuple[int, int, int]]]:
        if isinstance(sample_img_shapes, torch.Tensor):
            sample_img_shapes = sample_img_shapes.detach().cpu().tolist()

        def _is_fhw(value: Any) -> bool:
            return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(x, int) for x in value)

        if _is_fhw(sample_img_shapes):
            return [[tuple(int(x) for x in sample_img_shapes)]]

        if isinstance(sample_img_shapes, list):
            if len(sample_img_shapes) == 1 and _is_fhw(sample_img_shapes[0]):
                return [[tuple(int(x) for x in sample_img_shapes[0])]]
            if len(sample_img_shapes) == 1 and isinstance(sample_img_shapes[0], list):
                nested = sample_img_shapes[0]
                if all(_is_fhw(item) for item in nested):
                    return [[tuple(int(x) for x in item) for item in nested]]
            if all(_is_fhw(item) for item in sample_img_shapes):
                return [[tuple(int(x) for x in item) for item in sample_img_shapes]]

        raise ValueError(f"Unsupported img_shapes format: {sample_img_shapes}")

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor],
        img_shapes: Any,
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | list[torch.Tensor] | None = None,
        additional_t_cond: torch.Tensor | list[torch.Tensor] | None = None,
        latents: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        hidden_states_list = self._as_list(hidden_states)
        sample_count = len(hidden_states_list)
        timestep_list = self._as_list(timestep, sample_count)
        encoder_hidden_states_list = self._as_list(encoder_hidden_states, sample_count)
        target_list = self._as_list(training_target, sample_count)
        img_shapes_list = self._as_list(img_shapes, sample_count)
        mask_list = self._as_list(encoder_hidden_states_mask, sample_count)
        guidance_list = self._as_list(guidance, sample_count)
        additional_t_cond_list = self._as_list(additional_t_cond, sample_count)

        param_dtype = self.dtype

        per_sample_losses = []
        predictions = []
        for (
            sample_hs,
            sample_ts,
            sample_enc_hs,
            sample_target,
            sample_img_shapes,
            sample_mask,
            sample_guidance,
            sample_add_t_cond,
        ) in zip(
            hidden_states_list,
            timestep_list,
            encoder_hidden_states_list,
            target_list,
            img_shapes_list,
            mask_list,
            guidance_list,
            additional_t_cond_list,
        ):
            sample_hs = sample_hs.to(dtype=param_dtype)
            sample_enc_hs = sample_enc_hs.to(dtype=param_dtype)
            prediction = _QwenImageTransformer2DModel.forward(
                self,
                hidden_states=sample_hs,
                timestep=sample_ts,
                encoder_hidden_states=sample_enc_hs,
                encoder_hidden_states_mask=sample_mask,
                img_shapes=self._normalize_img_shapes(sample_img_shapes),
                guidance=sample_guidance,
                additional_t_cond=sample_add_t_cond,
                return_dict=False,
            )[0]
            predictions.append(prediction)
            per_sample_loss = F.mse_loss(prediction.float(), sample_target.float(), reduction="none")
            per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)

        loss = torch.stack(per_sample_losses).mean()
        return QwenImageModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _QwenImageTransformer2DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        diffusers_model = _QwenImageTransformer2DModel.from_pretrained(path, **kwargs)
        diffusers_model.__class__ = cls

        valid_keys = set(QWEN_IMAGE_INIT_SIGNATURE.parameters) - {"self"}
        diffusers_cfg = dict(diffusers_model.config)
        veomni_cfg = cls.config_class(**{k: v for k, v in diffusers_cfg.items() if k in valid_keys})
        diffusers_model.config = veomni_cfg
        diffusers_model.config.tie_word_embeddings = False
        return diffusers_model
