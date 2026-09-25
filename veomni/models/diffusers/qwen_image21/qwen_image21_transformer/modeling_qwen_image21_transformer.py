import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import QwenImage21Transformer2DModel as _QwenImage21Transformer2DModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_qwen_image21_transformer import (
    QWEN_IMAGE21_INIT_SIGNATURE,
    QwenImage21Transformer2DModelConfig,
)


class _QwenImage21TransformerInitShim(_QwenImage21Transformer2DModel):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


@dataclass
class QwenImage21ModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class QwenImage21Transformer2DModel(PreTrainedModel, _QwenImage21TransformerInitShim):
    config_class = QwenImage21Transformer2DModelConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImage21TransformerBlock"]

    def __init__(self, config: QwenImage21Transformer2DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        if hasattr(self, "_internal_dict"):
            del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _QwenImage21Transformer2DModel.__init__(self, **config.to_diffuser_dict())
        self.config = config
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
            return [] if length is None else [None] * length
        if isinstance(value, list):
            return value
        if length is not None and isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == length:
            return [value[index : index + 1] for index in range(length)]
        return [value]

    @staticmethod
    def _normalize_img_shapes(value: Any) -> list[list[tuple[int, int, int]]]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()

        def _is_shape(item):
            return isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(x, int) for x in item)

        if _is_shape(value):
            return [[tuple(value)]]
        if isinstance(value, list) and all(_is_shape(item) for item in value):
            return [[tuple(item) for item in value]]
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], list)
            and all(_is_shape(item) for item in value[0])
        ):
            return [[tuple(item) for item in value[0]]]
        raise ValueError(f"Unsupported Qwen-Image-2.1 img_shapes format: {value}")

    def predict_noise(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: Any,
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        prediction = _QwenImage21Transformer2DModel.forward(
            self,
            hidden_states=hidden_states.to(dtype=self.dtype),
            encoder_hidden_states=encoder_hidden_states.to(dtype=self.dtype),
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=self._normalize_img_shapes(img_shapes),
            img_mask=img_mask,
            return_dict=return_dict,
        )
        prediction = prediction.sample if return_dict else prediction[0]
        return prediction[:, -hidden_states.shape[1] :]

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        img_shapes: Any,
        img_mask: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        latents: torch.Tensor | list[torch.Tensor] | None = None,
        return_dict: bool = True,
    ):
        if training_target is None:
            return self.predict_noise(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                return_dict=return_dict,
            )

        hidden_states_list = self._as_list(hidden_states)
        sample_count = len(hidden_states_list)
        encoder_hidden_states_list = self._as_list(encoder_hidden_states, sample_count)
        encoder_hidden_states_mask_list = self._as_list(encoder_hidden_states_mask, sample_count)
        timestep_list = self._as_list(timestep, sample_count)
        img_shapes_list = self._as_list(img_shapes, sample_count)
        img_mask_list = self._as_list(img_mask, sample_count)
        target_list = self._as_list(training_target, sample_count)

        predictions = []
        per_sample_losses = []
        for sample_hs, sample_enc, sample_enc_mask, sample_ts, sample_shapes, sample_img_mask, sample_target in zip(
            hidden_states_list,
            encoder_hidden_states_list,
            encoder_hidden_states_mask_list,
            timestep_list,
            img_shapes_list,
            img_mask_list,
            target_list,
        ):
            prediction = self.predict_noise(
                hidden_states=sample_hs,
                encoder_hidden_states=sample_enc,
                encoder_hidden_states_mask=sample_enc_mask,
                timestep=sample_ts,
                img_shapes=sample_shapes,
                img_mask=sample_img_mask,
            )
            predictions.append(prediction)
            sample_loss = F.mse_loss(prediction.float(), sample_target.float(), reduction="none")
            per_sample_losses.append(sample_loss.reshape(sample_loss.shape[0], -1).mean(dim=1))

        loss = torch.stack(per_sample_losses).mean()
        return QwenImage21ModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        veomni_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _QwenImage21Transformer2DModel.save_pretrained(self, path, **kwargs)
        self.config = veomni_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = _QwenImage21Transformer2DModel.from_pretrained(path, **kwargs)
        model.__class__ = cls
        valid_keys = set(QWEN_IMAGE21_INIT_SIGNATURE.parameters) - {"self"}
        model.config = cls.config_class(
            **{key: value for key, value in dict(model.config).items() if key in valid_keys}
        )
        model.config.tie_word_embeddings = False
        return model
