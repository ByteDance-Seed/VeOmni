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
"""
Patch configuration for SeedOss GPU LigerKernel replacements.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.seed_oss.seed_oss_gpu_patch_gen_config -o veomni/models/transformers/seed_oss/generated

This mirrors the runtime GPU patch in
veomni/models/transformers/seed_oss/gpu_patch.py and the fused-CE patch in
veomni/models/transformers/seed_oss/modeling_seed_oss.py.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_seed_oss_gpu.py".
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external


config = PatchConfig(
    source_module="transformers.models.seed_oss.modeling_seed_oss",
    target_file="patched_modeling_seed_oss_gpu.py",
    description="SeedOss with LigerKernel GPU replacements + VeOmni fused-CE patch",
)


config.patches.append(
    create_patch_from_external(
        target="SeedOssRMSNorm",
        replacement_module="liger_kernel.transformers.rms_norm",
        replacement_name="LigerRMSNorm",
        description="Use LigerKernel RMSNorm",
    )
)

config.patches.append(
    create_patch_from_external(
        target="SeedOssMLP",
        replacement_module="liger_kernel.transformers.swiglu",
        replacement_name="LigerSwiGLUMLP",
        description="Use LigerKernel SwiGLU MLP",
    )
)


@config.replace_function("apply_rotary_pos_emb", description="Use LigerKernel rotary embedding")
def apply_rotary_pos_emb_liger(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    return liger_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids=position_ids,
        unsqueeze_dim=unsqueeze_dim,
    )


# ================================================================
# Patch: SeedOssForCausalLM.forward
# 1. Support fused cross_entropy loss function (hidden_states passed into
#    self.loss_function, logits computed only on the no-labels branch).
# ================================================================
@config.override_method(
    "SeedOssForCausalLM.forward",
    description="Support fused cross entropy path in SeedOssForCausalLM.forward",
)
def seed_oss_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    # --- Patch.1: Support fused cross_entropy loss function ---
    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    # --- Patch.1: Support fused cross_entropy loss function ---

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
