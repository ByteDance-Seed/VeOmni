# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Patch configuration for Gemma 4 GPU modeling generation."""

from collections import UserDict
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4CausalLMOutputWithPast,
    Gemma4Config,
    Gemma4ModelOutputWithPast,
    Gemma4MultimodalEmbedder,
    Gemma4TextModel,
    Gemma4TextModelOutputWithPast,
    Gemma4VisionModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin


config = PatchConfig(
    source_module="transformers.models.gemma4.modeling_gemma4",
    target_file="patched_modeling_gemma4_gpu.py",
    description="Gemma 4 with VeOmni packed-sequence masks and fused-loss integration",
)

config.drop_import_names("create_causal_mask", "create_sliding_window_causal_mask")
config.add_import(
    "veomni.models.transformers.masking_utils",
    names=["create_causal_mask", "create_sliding_window_causal_mask"],
)
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"],
)
config.add_post_import_block(
    """
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    """
)

veomni_causal_lm_loss = None


@config.add_helper_after("Gemma4CausalLMOutputWithPast")
@dataclass
class Gemma4CausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Gemma4CausalLMOutputWithPast):
    r"""
    Args:
        loss (`torch.FloatTensor`, *optional*):
            Language-modeling loss.
        logits (`torch.FloatTensor`, *optional*):
            Prediction scores before softmax.
        past_key_values (`Cache`, *optional*):
            Cached key/value states.
        hidden_states (`tuple[torch.FloatTensor]`, *optional*):
            Hidden states returned by the model.
        attentions (`tuple[torch.FloatTensor]`, *optional*):
            Attention weights returned by the model.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            Encoded image states.
        audio_hidden_states (`torch.FloatTensor`, *optional*):
            Encoded audio states.
        shared_kv_states (`dict`, *optional*):
            Shared key/value states used by Gemma 4 layers.
        fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
            Per-token tensors produced by VeOmni's fused-linear loss path.
    """


@config.replace_function(
    "create_masks_for_vision_model",
    description="Compose Gemma 4 bidirectional vision masks with packed-sequence boundaries",
)
def gemma4_create_masks_for_vision_model_patched(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    block_sequence_ids: torch.Tensor,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> dict:
    """Create Gemma 4 full and sliding masks without crossing packed samples."""
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "cu_seq_lens_q": cu_seq_lens_q,
    }

    full_mask = create_causal_mask(**mask_kwargs)

    early_exit, _, _, _, kv_length, _, kv_offset = _preprocess_mask_arguments(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        layer_idx=0,
    )
    if early_exit:
        padded_block_sequence_ids = block_sequence_ids
    else:
        padded_block_sequence_ids = maybe_pad_block_sequence_ids(
            block_sequence_ids, attention_mask, kv_length, kv_offset
        )

    sliding_mask = create_causal_mask(
        **mask_kwargs,
        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
        and_mask_function=sliding_window_overlay(config.sliding_window),
    )
    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


@config.override_method(
    "Gemma4Model.__init__",
    description="Construct generated Gemma 4 towers directly so language-model patches are retained",
)
def gemma4_model_init_patched(self, config: Gemma4Config):
    super().__init__(config)
    self.vision_tower = Gemma4VisionModel(config.vision_config) if config.vision_config is not None else None
    self.vocab_size = config.text_config.vocab_size
    self.language_model = Gemma4TextModel(config.text_config)
    self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
    self.audio_tower = Gemma4AudioModel(config.audio_config) if config.audio_config is not None else None
    self.embed_vision = (
        Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        if config.vision_config is not None
        else None
    )
    self.embed_audio = (
        Gemma4MultimodalEmbedder(config.audio_config, config.text_config) if config.audio_config is not None else None
    )
    self.post_init()


@config.override_method(
    "Gemma4Model.forward",
    description="Build packed-sequence-aware masks for Gemma 4 text batches",
)
def gemma4_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4ModelOutputWithPast:
    r"""
    input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`):
        The attention mask for the input audio.
    image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
        2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
    video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
        2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
    per_layer_inputs (`torch.Tensor`, *optional*):
        Pre-computed Gemma 4 per-layer embeddings.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None and per_layer_inputs is not None:
        raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

    image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
    multimodal_mask = image_mask | video_mask | audio_mask

    llm_input_ids = None
    if inputs_embeds is None:
        llm_input_ids = input_ids.clone()
        llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if per_layer_inputs is None and self.config.get_text_config().hidden_size_per_layer_input:
        pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
        multimodal_mask = multimodal_mask.to(inputs_embeds.device)
        llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
        per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)

    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
        image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        n_image_tokens = image_mask.sum()
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
            f" {image_features.shape[0]}",
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
        )

    if pixel_values_videos is not None:
        video_features = self.get_video_features(
            pixel_values_videos, video_position_ids, return_dict=True
        ).pooler_output
        video_features = torch.cat(video_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        n_video_tokens = video_mask.sum()
        video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[video_mask].numel() == video_features.numel(),
            f"Video features and video tokens do not match, tokens: {n_video_tokens}, features:"
            f" {video_features.shape[0]}",
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            video_mask.to(inputs_embeds.device), video_features.to(inputs_embeds.device)
        )

    if input_features is not None and input_features_mask is not None:
        audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
        audio_features = audio_output.pooler_output
        audio_mask_from_encoder = audio_output.attention_mask
        audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]
        n_audio_tokens = audio_mask.sum()
        audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[audio_mask].numel() == audio_features.numel(),
            f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:"
            f" {audio_features.shape[0] * audio_features.shape[1]}",
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        )

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        text_config = self.config.get_text_config()
        use_bidir = text_config.use_bidirectional_attention == "vision"
        if use_bidir and mm_token_type_ids is not None:
            block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
            causal_mask_mapping = create_masks_for_vision_model(
                block_sequence_ids=block_sequence_ids,
                cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
                **mask_kwargs,
            )
        else:
            mask_kwargs["cu_seq_lens_q"] = kwargs.get("cu_seq_lens_q")
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

    outputs = self.language_model(
        per_layer_inputs=per_layer_inputs,
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        return_dict=True,
        **kwargs,
    )

    return Gemma4ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
        audio_hidden_states=audio_features if input_features is not None else None,
        shared_kv_states=outputs.shared_kv_states,
    )


@config.override_method(
    "Gemma4TextModel.forward",
    description="Pass packed-sequence boundaries into VeOmni causal and sliding-window masks",
)
def gemma4_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    per_layer_inputs: torch.Tensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4TextModelOutputWithPast:
    r"""
    per_layer_inputs (`torch.Tensor`, *optional*):
        Pre-computed Gemma 4 per-layer embeddings.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None and per_layer_inputs is not None:
        raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self.hidden_size_per_layer_input:
        if per_layer_inputs is None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "cu_seq_lens_q": kwargs.get("cu_seq_lens_q"),
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    hidden_states = inputs_embeds
    position_embeddings = {
        layer_type: self.rotary_emb(hidden_states, position_ids, layer_type) for layer_type in self.unique_layer_types
    }

    shared_kv_states = kwargs.pop("shared_kv_states", UserDict())
    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
        hidden_states = decoder_layer(
            hidden_states,
            per_layer_input,
            shared_kv_states=shared_kv_states,
            position_embeddings=position_embeddings[self.config.layer_types[i]],
            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return Gemma4TextModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
    )


@config.override_method(
    "Gemma4TextModel.get_per_layer_inputs",
    description="Preserve Gemma 4 PLE lookup while keeping generated code lint-clean",
)
def gemma4_get_per_layer_inputs_patched(
    self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
) -> torch.Tensor:
    if not self.hidden_size_per_layer_input:
        raise RuntimeError(
            "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not support"
            f" per-layer embeddings. {self.config}"
        )

    if input_ids is None:
        with torch.no_grad():
            input_ids = (
                (
                    inputs_embeds[:, :, None, :]
                    == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                )
                .all(dim=3)
                .nonzero()[:, 2]
            )
            try:
                input_ids = input_ids.view(inputs_embeds.shape[:2])
            except RuntimeError:
                raise RuntimeError(
                    "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and "
                    "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs "
                    "to reverse the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                ) from None

    return self.embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape,
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )


@config.add_helper
def _gemma4_loss(
    model,
    hidden_states: torch.Tensor,
    labels: torch.LongTensor | None,
    vocab_size: int,
    final_logit_softcapping: float | None,
    **kwargs,
):
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            if final_logit_softcapping is not None:
                raise ValueError(
                    "Gemma 4 fused-linear loss does not support final_logit_softcapping; "
                    "use cross_entropy_loss_implementation='eager'."
                )
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=None,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=hidden_states,
                weights=model.lm_head.weight,
                **kwargs,
            )
        else:
            logits = model.lm_head(hidden_states)
            if final_logit_softcapping is not None:
                logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
            loss, _, fused_linear_aux = model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=hidden_states,
                weights=model.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = model.lm_head(hidden_states)
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return loss, logits, fused_linear_aux


@config.override_method(
    "Gemma4ForCausalLM.forward",
    description="Adapt Gemma 4 text causal-LM loss to VeOmni's fused-loss output contract",
)
def gemma4_for_causal_lm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4CausalLMOutputWithLogProbs:
    r"""
    per_layer_inputs (`torch.Tensor`, *optional*):
        Pre-computed Gemma 4 per-layer embeddings.
    """
    outputs: Gemma4TextModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    if labels is None:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
    loss, logits, fused_linear_aux = _gemma4_loss(
        self,
        hidden_states,
        labels,
        self.vocab_size,
        self.config.final_logit_softcapping,
        **kwargs,
    )
    return Gemma4CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        shared_kv_states=outputs.shared_kv_states,
    )


@config.override_method(
    "Gemma4ForConditionalGeneration.forward",
    description="Adapt Gemma 4 multimodal causal-LM loss to VeOmni's fused-loss output contract",
)
def gemma4_for_conditional_generation_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    per_layer_inputs: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4CausalLMOutputWithLogProbs:
    r"""
    input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`):
        The attention mask for the input audio.
    image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
        2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
    video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
        2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
    per_layer_inputs (`torch.Tensor`, *optional*):
        Pre-computed Gemma 4 per-layer embeddings.
    """
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    if labels is None:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
    text_config = self.config.get_text_config()
    loss, logits, fused_linear_aux = _gemma4_loss(
        self,
        hidden_states,
        labels,
        text_config.vocab_size,
        text_config.final_logit_softcapping,
        **kwargs,
    )
    return Gemma4CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        shared_kv_states=outputs.shared_kv_states,
    )
