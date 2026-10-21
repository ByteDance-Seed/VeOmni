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
"""
Patch configuration for Gemma 3 VeOmni NPU modeling path (text + VLM).

Regen command:
patchgen veomni.models.transformers.gemma3.gemma3_npu_patch_gen_config -o veomni/models/transformers/gemma3/generated --diff

This inherits the GPU patch in
veomni/models/transformers/gemma3/gemma3_gpu_patch_gen_config.py. The local
``VeomniOp`` handles select NPU kernels through the shared op registry, while
this config adds the VLM-specific model, mask, checkpoint, and loss patches.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_gemma3_npu.py".
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast, Gemma3ModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.models.transformers.gemma3.gemma3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_gemma3_npu.py"
config.description = "Gemma 3 (text + VLM) with VeomniOp NPU replacements"
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.utils.constants", names=["IGNORE_INDEX"])


@config.add_helper
def get_gemma3_position_ids(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.LongTensor]:
    position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
    return {"position_ids": position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)}


@config.add_helper
def create_gemma3_masks(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    token_type_ids: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    attention_impl: str,
) -> dict:
    """Build Gemma 3 full/sliding masks while retaining packed-sample boundaries."""
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    if cu_seq_lens_q is None:
        previous_impl = config._attn_implementation
        config._attn_implementation = attention_impl
        try:
            if token_type_ids is None:
                return create_masks_for_generate(**mask_kwargs)
            block_sequence_ids = get_block_sequence_ids_for_mask(token_type_ids, device=inputs_embeds.device)
            return create_masks_for_vision_model(block_sequence_ids=block_sequence_ids, **mask_kwargs)
        finally:
            config._attn_implementation = previous_impl

    if past_key_values is not None:
        raise ValueError("Packed Gemma 3 VLM attention does not support past_key_values.")

    q_len = inputs_embeds.shape[1]
    packed_kwargs = {
        "impl": attention_impl,
        "device": inputs_embeds.device,
        "batch_size": inputs_embeds.shape[0],
        "dtype": inputs_embeds.dtype,
        "cu_seqlens": cu_seq_lens_q,
    }
    if attention_mask is not None:
        packed_kwargs["attention_mask"] = attention_mask
    if token_type_ids is not None:
        block_sequence_ids = get_block_sequence_ids_for_mask(token_type_ids, device=inputs_embeds.device)
        packed_kwargs["or_mask_function"] = blockwise_overlay(block_sequence_ids)

    return {
        "full_attention": packed_causal_mask(q_len, q_len, **packed_kwargs),
        "sliding_attention": sliding_window_mask(
            q_len,
            q_len,
            sliding_window=config.sliding_window,
            **packed_kwargs,
        ),
    }


@config.add_helper_after("Gemma3CausalLMOutputWithPast")
@dataclass
class Gemma3CausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Gemma3CausalLMOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Image features returned by the vision encoder after projection.
    fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
        Per-token tensors produced by the fused-linear loss path. This is
        ``None`` on the plain loss path.
    """


@config.override_method(
    "Gemma3Model.dummy_vision_forward",
    description="Run minimal vision and projector forwards for text-only distributed microbatches",
)
def gemma3_model_dummy_vision_forward(self):
    patch_size = self.config.vision_config.patch_size
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size

    vision_parameter = next(self.vision_tower.parameters())
    pixel_values = torch.zeros(
        (1, self.config.vision_config.num_channels, patch_height, patch_width),
        dtype=vision_parameter.dtype,
        device=vision_parameter.device,
    )
    vision_outputs = self.vision_tower(
        pixel_values=pixel_values,
        interpolate_pos_encoding=True,
        return_dict=True,
    )
    dependency = vision_outputs.last_hidden_state.mean()
    if vision_outputs.pooler_output is not None:
        dependency = dependency + vision_outputs.pooler_output.mean()

    projector_input = vision_outputs.last_hidden_state.new_zeros(
        (
            1,
            self.multi_modal_projector.patches_per_image**2,
            self.config.vision_config.hidden_size,
        )
    )
    return dependency + self.multi_modal_projector(projector_input).mean()


@config.override_method(
    "Gemma3Model.__init__",
    description="Construct the generated text tower instead of the upstream AutoModel class",
)
def gemma3_model_init_npu(self, config):
    super().__init__(config)
    self.vision_tower = AutoModel.from_config(config=config.vision_config)
    self.multi_modal_projector = Gemma3MultiModalProjector(config)
    self.vocab_size = config.text_config.vocab_size

    # AutoModel resolves to the upstream class, bypassing the NPU patches.
    self.language_model = Gemma3TextModel._from_config(config.text_config)
    self.post_init()


@config.override_method(
    "Gemma3Model.forward",
    description="Preserve packed-sample boundaries in Gemma 3 VLM attention masks",
)
def gemma3_model_forward_npu(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    **lm_kwargs: Unpack[TransformersKwargs],
) -> tuple | Gemma3ModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if input_ids is not None and self.config.image_token_id >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values, return_dict=True).pooler_output
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    elif get_parallel_state().fsdp_enabled:
        dummy_dependency = self.dummy_vision_forward().to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + dummy_dependency * 0.0

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        causal_mask_mapping = create_gemma3_masks(
            config=self.config.get_text_config(),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            cu_seq_lens_q=lm_kwargs.get("cu_seq_lens_q"),
            attention_impl=self.language_model.layers[0].self_attn.veomni_attn.impl,
        )

    outputs = self.language_model(
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        return_dict=True,
        **lm_kwargs,
    )

    return Gemma3ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


# Patch the multimodal forward to use the same local cross-entropy VeomniOp as
# the text-only GPU config.


@config.override_method(
    "Gemma3ForConditionalGeneration.__init__",
    description="Map legacy Gemma 3 VLM checkpoint names to the current nested model layout",
)
def gemma3_for_conditional_generation_init_npu(self, config):
    super().__init__(config)
    self.model = Gemma3Model(config)
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self._checkpoint_conversion_mapping = {
        r"^language_model\.model\.": "model.language_model.",
        r"^vision_tower\.vision_model\.": "model.vision_tower.",
        r"^multi_modal_projector\.": "model.multi_modal_projector.",
    }
    impl = resolve_op_impl("cross_entropy_loss_implementation")
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()


@config.override_method(
    "Gemma3ForConditionalGeneration.forward",
    description="Use VeOmni fused cross-entropy in the VLM forward path",
)
def gemma3_for_conditional_generation_forward_npu(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **lm_kwargs: Unpack[TransformersKwargs],
) -> Gemma3CausalLMOutputWithLogProbs:
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=labels,
        return_dict=True,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss_labels = labels
        if attention_mask is not None and hidden_states.shape[1] > 1:
            # Match the upstream VLM loss: padding positions in the shifted
            # causal targets do not contribute, including longer PEFT masks.
            shift_attention_mask = attention_mask[:, -(hidden_states.shape[1] - 1) :].to(labels.device)
            loss_labels = labels.clone()
            loss_labels[..., 1:] = loss_labels[..., 1:].masked_fill(
                shift_attention_mask == 0,
                IGNORE_INDEX,
            )

        if self.config.text_config.final_logit_softcapping is not None:
            logits = self.lm_head(hidden_states)
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=loss_labels,
                vocab_size=self.config.text_config.vocab_size,
                **lm_kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
        else:
            loss, logits, fused_linear_aux = self.loss_function(
                logits=None,
                labels=loss_labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **lm_kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping

    return Gemma3CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


@config.override_method(
    "Gemma3ForConditionalGeneration.get_position_id_func",
    description="Expose standard one-dimensional position IDs to the VeOmni VLM data pipeline",
)
def gemma3_get_position_id_func(self):
    return get_gemma3_position_ids


@config.override_method(
    "Gemma3ForConditionalGeneration.get_extra_collate_infos",
    description="Declare Gemma 3 image and token-type packing rules",
)
def gemma3_get_extra_collate_infos(self):
    if get_parallel_state().sp_enabled:
        raise ValueError("Gemma 3 VLM does not support sequence parallelism; set cp_size=1 and ulysses_size=1.")
    return {
        "pixel_values": (0, False, None, None),
        "token_type_ids": (-1, True, 0, 1),
    }
