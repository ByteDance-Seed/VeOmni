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

This mirrors the GPU patch in
veomni/models/transformers/gemma3/gemma3_gpu_patch_gen_config.py and adds
OpSlot guards for NPU fused RMSNorm and RoPE kernels, plus a VLM-aware
patch for Gemma3ForConditionalGeneration.forward that uses fused CE.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_gemma3_npu.py".
"""

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.models.transformers.gemma3.gemma3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.models.transformers.gemma3.gemma3_gpu_patch_gen_config import (
    gemma3_forcausallm_forward_patched,
    gemma3_textmodel_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin


config = PatchConfig(
    source_module="transformers.models.gemma3.modeling_gemma3",
    target_file="patched_modeling_gemma3_npu.py",
    description="Gemma 3 (text + VLM) with VeOmni NPU fused-operator replacements",
)

# Mirror additional imports + post-import helpers + dropped names from the GPU
# config so the generated file is self-contained (same masking-utils imports,
# same CausalLMOutputWithLogProbs import, same veomni_causal_lm_loss OpSlot).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)
config.add_import("veomni.utils.constants", names=["IGNORE_INDEX"])


@config.add_helper
def get_gemma3_position_ids(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.LongTensor]:
    position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
    return {"position_ids": position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)}


# NPU-specific OpSlot declarations (RMSNorm + RoPE) on top of the GPU config's
# cross-entropy-loss OpSlot.
config.add_post_import_block(
    """
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    """
)


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


# Gemma 3 uses (1.0 + weight) scaling (weight zero-initialised), and the eps
# attribute is ``self.eps`` (not ``self.variance_epsilon``).  Pass ``1.0 +
# self.weight`` so the NPU ``npu_rms_norm`` kernel reproduces the Gemma
# contract exactly.


@config.override_method(
    "Gemma3RMSNorm.forward",
    description="OpSlot guard for NPU fused RMSNorm (Gemma 1.0+weight formulation)",
)
def gemma3_rmsnorm_forward_npu(self, x: torch.Tensor) -> torch.Tensor:
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(x, 1.0 + self.weight, self.eps)
    # Original HF code below, unchanged.
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


@config.replace_function(
    "apply_rotary_pos_emb",
    description="OpSlot guard for NPU fused RoPE",
)
def apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # Original HF code below, unchanged.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


# Reuse the GPU patch verbatim. The masking-utils wrappers handle both
# FlexAttention (BlockMask) and SDPA/eager (tensor mask) backends.


config.override_method(
    "Gemma3TextModel.forward",
    replacement=gemma3_textmodel_forward_patched,
    description="Pass packed-sequence boundaries into VeOmni FlexAttention mask preparation",
)


# Reuse the GPU patch verbatim. The veomni_causal_lm_loss OpSlot dispatches to
# the NPU chunk-loss kernel when bound.


config.override_method(
    "Gemma3ForCausalLM.forward",
    replacement=gemma3_forcausallm_forward_patched,
    description="Adapt Gemma 3 causal-LM loss to VeOmni's fused-loss output contract",
)


# Patch the multimodal (VLM) forward to use VeOmni's fused-CE OpSlot instead
# of the upstream ``nn.CrossEntropyLoss``.  This is the VLM-specific addition
# over the text-only GPU config (which only patches Gemma3ForCausalLM).


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

        if veomni_causal_lm_loss.use_non_eager_impl:
            if self.config.text_config.final_logit_softcapping is not None:
                raise ValueError(
                    "Gemma 3 fused-linear loss does not support final_logit_softcapping; "
                    "use cross_entropy_loss_implementation='eager'."
                )
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=None,
                labels=loss_labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **lm_kwargs,
            )
        else:
            logits = self.lm_head(hidden_states).float()
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
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
    return {
        "pixel_values": (0, False, None, None),
        "token_type_ids": (-1, True, 0, 1),
    }
