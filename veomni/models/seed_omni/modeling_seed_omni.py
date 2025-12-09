# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.distributed.nn.functional as distF
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...data.constants import IGNORE_INDEX
from ...distributed.parallel_plan import ParallelPlan
from ...distributed.parallel_state import get_parallel_state
from ...distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)
from ...utils import logging
from ..loader import get_model_class
from .configuration_seed_omni import SeedOmniConfig, SeedOmniDecoderConfig, SeedOmniEncoderConfig
from .decoder import BaseDecoderModelMixin, BaseDecoderOutput
from .encoder import BaseEncoderModelMixin
from .foundation import BaseFoundationModelMixin


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Cache


def extract_model_inputs(prefix: str, kwargs: Dict[str, "torch.Tensor"]):
    model_inputs = {}
    for key, value in kwargs.items():
        if key.startswith(prefix):
            model_inputs[key[len(prefix) :]] = value
    return model_inputs


@dataclass
class SeedOmniOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None


class SeedOmniPreTrainedModel(PreTrainedModel):
    config_class = SeedOmniConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    @property
    def _no_split_modules(self):
        no_split_modules = []
        for module in self.children():
            if isinstance(module, PreTrainedModel) and module._no_split_modules:
                no_split_modules.extend(module._no_split_modules)
            elif isinstance(module, nn.ModuleDict):
                for sub_module in module.children():
                    if isinstance(sub_module, PreTrainedModel) and sub_module._no_split_modules:
                        no_split_modules.extend(sub_module._no_split_modules)

        return no_split_modules

    @_no_split_modules.setter
    def _no_split_modules(self, value):
        pass

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SeedOmniEncoderModel(SeedOmniPreTrainedModel):
    config_class = SeedOmniEncoderConfig

    def __init__(self, config: SeedOmniEncoderConfig):
        super().__init__(config)
        torch_dtype = torch.get_default_dtype()
        self.text_encoder = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            padding_idx=config.text_config.pad_token_id,
            dtype=torch_dtype,
        )
        self.modality = []
        if config.image_config.model_type:
            model_cls = get_model_class(config.image_config)
            self.image_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.image_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("image")
            self.modality.append("video")  # image encoder could be used for video embedding

        if config.video_config.model_type:
            model_cls = get_model_class(config.video_config)
            self.video_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.video_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("video") if "video" not in self.modality else None

        if config.audio_config.model_type:
            model_cls = get_model_class(config.audio_config)
            self.audio_encoder: BaseEncoderModelMixin = model_cls._from_config(
                config.audio_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("audio")

        self.encode_input = config.encode_input
        self.encode_output = config.encode_output

    def set_projector_trainable_only(self):
        for module in self.children():
            if isinstance(module, BaseEncoderModelMixin):
                module.set_projector_trainable_only()

    def image_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_image_inputs = extract_model_inputs("image_input_", kwargs)
            input_image_mask: torch.Tensor = input_image_inputs.pop("mask", None)
            if input_image_inputs:
                input_image_features: torch.Tensor = self.image_encoder.lm_encode(**input_image_inputs).to(
                    inputs_embeds
                )
                if self.training and get_parallel_state().sp_enabled:
                    input_image_features = gather_seq_scatter_heads(
                        input_image_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                input_image_features = input_image_features[: input_image_mask.sum()]
                image_mask = input_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, input_image_features)
            elif self.training:
                dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        if self.encode_output:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if output_image_inputs:
                output_image_features: torch.Tensor = self.image_encoder.lm_encode(**output_image_inputs).to(
                    inputs_embeds
                )
                if self.training and get_parallel_state().sp_enabled:
                    output_image_features = gather_seq_scatter_heads(
                        output_image_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                output_image_features = output_image_features[: output_image_mask.sum()]
                image_mask = output_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, output_image_features)
                decoder_inputs["image_output_labels"] = output_image_features
            elif self.training:
                dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0
        return inputs_embeds

    def video_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_video_inputs = extract_model_inputs("video_input_", kwargs)
            input_video_mask: torch.Tensor = input_video_inputs.pop("mask", None)
            if input_video_inputs:
                if getattr(self, "video_encoder", None) is not None:
                    input_video_features: torch.Tensor = self.video_encoder.lm_encode(**input_video_inputs).to(
                        inputs_embeds
                    )
                else:
                    input_video_features: torch.Tensor = self.image_encoder.lm_encode(**input_video_inputs).to(
                        inputs_embeds
                    )
                if self.training and get_parallel_state().sp_enabled:
                    input_video_features = gather_seq_scatter_heads(
                        input_video_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                input_video_features = input_video_features[: input_video_mask.sum()]
                video_mask = input_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, input_video_features)
            elif self.training:
                if getattr(self, "video_encoder", None) is not None:
                    dummy_embeds: torch.Tensor = self.video_encoder.lm_dummy_encode()
                else:
                    dummy_embeds: torch.Tensor = self.image_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        # TODO: video generation encode
        return inputs_embeds

    def audio_forward(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_audio_inputs = extract_model_inputs("audio_input_", kwargs)
            input_audio_mask: torch.Tensor = input_audio_inputs.pop("mask", None)
            if input_audio_inputs and input_audio_mask.sum() > 0:
                input_audio_features: torch.Tensor = self.audio_encoder.lm_encode(**input_audio_inputs).to(
                    inputs_embeds
                )
                # TODO: sp_check
                if self.training and get_parallel_state().sp_enabled:
                    input_audio_features = gather_seq_scatter_heads(
                        input_audio_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                input_audio_features = input_audio_features[: input_audio_mask.sum()]
                audio_mask = input_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, input_audio_features)
            elif self.training:
                dummy_embeds: torch.Tensor = self.audio_encoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        # TODO: audio generation encode
        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs_embeds: torch.Tensor = self.text_encoder(input_ids)
        decoder_inputs = {}

        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )

        if "image" in self.modality:
            inputs_embeds = self.image_forward(inputs_embeds, decoder_inputs, **kwargs)

        if "video" in self.modality:
            inputs_embeds = self.video_forward(inputs_embeds, decoder_inputs, **kwargs)

        if "audio" in self.modality:
            inputs_embeds = self.audio_forward(inputs_embeds, decoder_inputs, **kwargs)

        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )
        return {"inputs_embeds": inputs_embeds, "decoder_inputs": decoder_inputs}


class SeedOmniDecoderModel(SeedOmniPreTrainedModel):
    config_class = SeedOmniDecoderConfig

    def __init__(self, config: SeedOmniDecoderConfig):
        self.config = config
        super().__init__(config)
        torch_dtype = torch.get_default_dtype()
        self.modality = []
        if config.image_config.model_type:
            model_cls = get_model_class(config.image_config)
            self.image_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.image_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("image")  # TODO: config keys for sp_slice
        if config.video_config.model_type:
            model_cls = get_model_class(config.video_config)
            self.video_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.video_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("video")
        if config.audio_config.model_type:
            model_cls = get_model_class(config.audio_config)
            self.audio_decoder: BaseDecoderModelMixin = model_cls._from_config(
                config.audio_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
            )
            self.modality.append("audio")

        self.encode_input = config.encode_input
        self.encode_output = config.encode_output

    def set_projector_trainable_only(self):
        for module in self.children():
            if isinstance(module, BaseDecoderModelMixin):
                module.set_projector_trainable_only()

    def image_encode(self, inputs_embeds: torch.Tensor, decoder_inputs, **kwargs):
        if self.encode_input:
            input_image_inputs = extract_model_inputs("image_input_", kwargs)
            input_image_mask: torch.Tensor = input_image_inputs.pop("mask", None)
            if input_image_inputs:
                input_image_features, _ = self.image_decoder.lm_encode(**input_image_inputs)
                if self.training and get_parallel_state().sp_enabled:
                    input_image_features = gather_seq_scatter_heads(
                        input_image_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                input_image_features = input_image_features[: input_image_mask.sum()]
                image_mask = input_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_features = input_image_features.to(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            elif self.training:
                dummy_embeds, _ = self.image_decoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0

        if self.encode_output:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if output_image_inputs:
                output_image_features, output_image_indices = self.image_decoder.lm_encode(**output_image_inputs)
                if self.training and get_parallel_state().sp_enabled:
                    output_image_features = gather_seq_scatter_heads(
                        output_image_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                    )
                output_image_features = output_image_features[: output_image_mask.sum()]
                image_mask = output_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, output_image_features)
                decoder_inputs["image_output_labels"] = output_image_indices
            elif self.training:
                dummy_embeds, _ = self.image_decoder.lm_dummy_encode()
                inputs_embeds += dummy_embeds.mean() * 0.0
        return inputs_embeds

    def encode(self, inputs_embeds: torch.Tensor, **kwargs):
        """Encodes the output images to foundation input embeds and image indices labels.
        Only support descrete tokenizer encode.
        Returns of decoder['mm_type'].encode:
            - embeds (torch.Tensor(dtype=float32, shape=(batch_size, seq_len, hidden_size)): input_embeds
            - indices (torch.Tensor(dtype=int64, shape=(batch_size, seq_len))): feature code
        """
        decoder_inputs = kwargs.pop("decoder_inputs", {})

        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )

        if "image" in self.modality:
            inputs_embeds = self.image_encode(inputs_embeds, decoder_inputs, **kwargs)

        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )

        return {"inputs_embeds": inputs_embeds, "decoder_inputs": decoder_inputs}

    def image_decode(self, hidden_states: torch.Tensor, decoder_inputs, loss, **kwargs):
        image_output_labels = decoder_inputs.get("image_output_labels", None)
        target_inputs = extract_model_inputs("image_target", kwargs)
        if target_inputs or image_output_labels is not None:
            output_image_inputs = extract_model_inputs("image_output_", kwargs)
            output_image_mask: torch.Tensor = output_image_inputs.pop("mask", None)
            if self.training and get_parallel_state().sp_enabled:
                bs = output_image_mask.size(0)
                sp_size = get_parallel_state().sp_size
                sp_rank = get_parallel_state().sp_rank
                sp_chunk_size = output_image_mask.size(-1) // sp_size
                sp_slice_dim = 1  # bs, seq, ...

                output_image_mask = F.pad(output_image_mask[..., 1:], (0, 1), value=0)

                labels_shape = [bs] + list(image_output_labels.shape)
                labels_shape[1] = output_image_mask.shape[1]
                labels = torch.zeros(labels_shape).to(image_output_labels)

                gathered_image_output_labels = distF.all_gather(
                    image_output_labels, group=get_parallel_state().sp_group
                )
                gathered_image_output_labels = torch.cat(gathered_image_output_labels, dim=0)
                gathered_image_output_labels = gathered_image_output_labels[: output_image_mask.sum()]

                if len(labels_shape) == 3:  # bs, seq, dim
                    output_image_mask = output_image_mask.unsqueeze(-1).expand_as(labels)
                labels[~output_image_mask] = IGNORE_INDEX
                labels = labels.masked_scatter(output_image_mask, gathered_image_output_labels)
                labels = labels.narrow(sp_slice_dim, sp_rank * sp_chunk_size, sp_chunk_size)

                outputs: BaseDecoderOutput = self.image_decoder.lm_head(
                    hidden_states,
                    labels=labels,
                    **target_inputs,
                )
                loss["image_decoder_loss"] = outputs.loss
            else:
                output_image_hs = hidden_states[..., :-1, :][output_image_mask[..., 1:]]
                labels = image_output_labels
                outputs: BaseDecoderOutput = self.image_decoder.lm_head(
                    output_image_hs,
                    labels=labels,
                    **target_inputs,
                )
                loss["image_decoder_loss"] = outputs.loss
        elif self.training:
            dummy_hidden_states = hidden_states[..., -1, :]
            outputs: BaseDecoderOutput = self.image_decoder.lm_head(dummy_hidden_states)
            loss["image_decoder_loss"] = outputs.logits.mean() * 0.0

    def decode(self, hidden_states: torch.Tensor, decoder_inputs: dict = {}, **kwargs):
        loss = {}
        if "image" in self.modality:
            self.image_decode(hidden_states, decoder_inputs, loss, **kwargs)

        return loss

    def lm_embed(self, hidden_states: torch.Tensor, model_type: str = "image", **kwargs):
        if model_type == "image":
            outputs = self.image_decoder.lm_embed(hidden_states, **kwargs)
        else:
            raise NotImplementedError
        return outputs

    def generate(self, hidden_states: torch.Tensor, modal_type: str = "image", **kwargs):
        if modal_type == "image":
            outputs = self.image_decoder.lm_generate(hidden_states, **kwargs)
        else:
            raise NotImplementedError
        return outputs


class SeedOmniModel(SeedOmniPreTrainedModel, GenerationMixin):
    def __init__(self, config: SeedOmniConfig):
        super().__init__(config)
        self.config = config
        torch_dtype = torch.get_default_dtype()
        model_cls = get_model_class(config.foundation_config)
        self.foundation: BaseFoundationModelMixin = model_cls._from_config(
            config.foundation_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )
        self.encoder = SeedOmniEncoderModel._from_config(
            config.encoder_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )
        self.decoder = SeedOmniDecoderModel._from_config(
            config.decoder_config, attn_implementation=config._attn_implementation, torch_dtype=torch_dtype
        )

    def get_input_embeddings(self):
        return self.encoder.text_encoder

    def set_input_embeddings(self, value):
        self.encoder.text_encoder = value

    def get_output_embeddings(self):
        return self.foundation.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.foundation.set_output_embeddings(new_embeddings)

    def get_parallel_plan(self):
        parallel_plan: ParallelPlan = self.foundation.get_parallel_plan()
        parallel_plan.update_prefix("foundation")
        return parallel_plan

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ):
        model_embeds = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = self.vocab_size
        self.foundation.vocab_size = vocab_size
        self.foundation.config.get_text_config().vocab_size = vocab_size
        self.config.encoder_config.text_config.vocab_size = vocab_size
        return model_embeds

    def get_modality(self):
        input_modality = self.encoder.modality
        output_modality = self.decoder.modality
        return {"input": input_modality, "output": output_modality}

    def get_position_id_func(self):
        """
        func(input_ids=input_ids, **kwargs) -> dict(position_ids=position_ids, **kwargs)
        """
        return self.foundation.position_id_func

    def forward(self, **inputs: torch.Tensor):
        decoder_inputs = {}

        if "inputs_embeds" not in inputs:
            encoder_encodes = self.encoder.forward(**inputs)
            decoder_encodes = self.decoder.encode(**inputs, **encoder_encodes)
            inputs["inputs_embeds"] = decoder_encodes["inputs_embeds"]
            decoder_inputs = decoder_encodes["decoder_inputs"]

        inputs["return_dict"] = True
        inputs["output_hidden_states"] = True
        outputs = self.foundation(**inputs)

        loss = None
        if outputs.loss is not None:
            if torch.isnan(outputs.loss):
                outputs.loss = torch.nan_to_num(outputs.loss)
            loss = outputs.loss

            hidden_states = outputs.hidden_states[-1]
            decoder_loss = self.decoder.decode(hidden_states=hidden_states, decoder_inputs=decoder_inputs, **inputs)
            losses = {"foundation_loss": loss}
            for key, v in decoder_loss.items():
                losses[key] = v

            return SeedOmniOutput(
                losses=losses,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states,
            )
        return outputs

    def _prepare_image_generation_config(
        self,
        image_start_token: int = None,
        image_end_token: int = None,
        image_token_num: List = None,
        image_parallel_size: int = 16,
        image_classifier_free_guidance: bool = True,
        image_generation_config: dict = {},
        **kwargs,
    ):
        self.image_start_token = image_start_token
        self.image_token_num = image_token_num
        self.image_parallel_size = image_parallel_size
        self.image_classifier_free_guidance = image_classifier_free_guidance
        self.image_generation_config = image_generation_config
        # TODO: swift gen config based on text token (text token in image position should be trained)
        self.image_end_token = image_end_token
        return kwargs

    def _prepare_generation_config(self, *args, force_image_gen: bool = False, **kwargs):
        kwargs = self._prepare_image_generation_config(**kwargs)

        self.force_image_gen = force_image_gen
        if self.force_image_gen:
            self.parallel_size = self.image_parallel_size
            kwargs = self.setup_image_generation(**kwargs)
        else:
            kwargs = self.setup_text_generation(**kwargs)

        return super()._prepare_generation_config(*args, **kwargs)

    def setup_image_generation(self, **kwargs):
        self.gen_type = "image"
        self.tmp_image = []
        self.generated_images_sequence = []
        if hasattr(self.foundation, "get_generation_position_id"):
            # TODO: if customized rope for image generation
            self.generation_position_id_map = None
        else:  # 1d rope
            self.generation_position_id_map = None

        kwargs["input_ids"] = kwargs["input_ids"].repeat(self.image_parallel_size, 1)
        kwargs["attention_mask"] = kwargs["attention_mask"].repeat(self.image_parallel_size, 1)
        kwargs["position_ids"] = kwargs["position_ids"].repeat_interleave(self.image_parallel_size, dim=0)
        if self.image_classifier_free_guidance:
            if kwargs.get("bos_token_id", None) and kwargs["input_ids"][0][0] == kwargs["bos_token_id"]:
                start_id = 1
            else:
                start_id = 0
            if kwargs["input_ids"][0][-1] == self.image_start_token:
                end_id = -1
            else:
                end_id = 0

            kwargs["input_ids"][1::2, start_id:end_id] = kwargs.get("pad_token_id", 0)
        return kwargs

    def setup_text_generation(self, **kwargs):
        self.gen_type = "text"
        self.generation_position_id_map = None
        return kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):  # only support bs=1 inference
        foundation_args = set(inspect.signature(self.foundation.prepare_inputs_for_generation).parameters)
        encoder_decoder_inputs = {}

        for key in list(kwargs.keys()):
            encoder_decoder_inputs[key] = kwargs.pop(key)
            if key in foundation_args:
                kwargs[key] = encoder_decoder_inputs.pop(key)

        model_inputs = self.foundation.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )
        if cache_position[0] == 0:  # first time inference
            encoder_encodes = self.encoder(input_ids=input_ids, **encoder_decoder_inputs)
            model_inputs["inputs_embeds"] = encoder_encodes["inputs_embeds"]
            return model_inputs

        model_inputs.pop("position_ids", None)
        if self.gen_type == "text":
            if input_ids[0][-1] == self.image_start_token:
                self.setup_image_generation()
            # TODO: other modality control
        elif self.gen_type == "image":
            hidden_states = encoder_decoder_inputs["hidden_states"][-1]  # bs, cache_len, dim
            input_embeds, next_tokens = self.decoder.lm_embed(
                hidden_states[:, -1:], model_type="image", **self.image_generation_config
            )
            self.tmp_image.append(next_tokens)
            generated_tokens = len(self.tmp_image)
            model_inputs["inputs_embeds"] = input_embeds
            if self.generation_position_id_map is not None:  # TODO: customized rope for image generation
                position_ids = self.generation_position_id_map[..., generated_tokens : generated_tokens + 1]
                model_inputs["position_ids"] = position_ids + cache_position[0] - generated_tokens
            if generated_tokens == self.image_token_num:
                tmp_image = torch.cat(self.tmp_image, dim=-1)
                self.generated_images_sequence.append(tmp_image)
                self.tmp_image = []
                self.setup_text_generation()
        else:
            raise NotImplementedError
        return model_inputs

    def generate_multimodal(self, hidden_states, modal_type="image"):  # TODO: other modal_type
        return self.decoder.generate(hidden_states, modal_type=modal_type, **self.image_generation_config)

    def _validate_model_kwargs(self, model_kwargs):
        pass

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = self.foundation._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_kwargs, **kwargs
        )
        model_kwargs["hidden_states"] = outputs.hidden_states
        return model_kwargs

    def _has_unfinished_sequences(self, *args, **kwargs) -> bool:
        if self.gen_type != "text":
            return True
        return super()._has_unfinished_sequences(*args, **kwargs)
