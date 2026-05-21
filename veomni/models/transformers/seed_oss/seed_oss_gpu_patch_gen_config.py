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

Patches:
- ``SeedOssRMSNorm`` -> ``LigerRMSNorm`` (functional fused RMSNorm).
- ``SeedOssMLP`` -> ``LigerSwiGLUMLP`` (functional fused SwiGLU MLP).
- ``apply_rotary_pos_emb`` -> Liger ``liger_rotary_pos_emb``.
- ``SeedOssForCausalLM.forward``: OpSlot guard for fused cross-entropy loss
  (falls through to the eager HF loss path when no fused kernel is bound) and
  returns the unified ``CausalLMOutputWithLogProbs`` dataclass so callers can
  surface per-token log-probs / entropy alongside the loss.

This file itself is not runnable — it is the declarative source of truth for
the runnable explicitly-patched modeling file
"generated/patched_modeling_seed_oss_gpu.py".
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.ops.dispatch import OpSlot
from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs


# ── OpSlot declarations ──────────────────────────────────────────────────────
# Mirrors the ``add_post_import_block`` content below so the patch function
# bodies type-check and get IDE completion in this file. The actual runtime
# slots used by the generated modeling are the ones declared in the post-import
# block (a separate module scope), and are bound by ``_bind_veomni_ops()`` in
# ``veomni/models/auto.py`` at model-build time.
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")


config = PatchConfig(
    source_module="transformers.models.seed_oss.modeling_seed_oss",
    target_file="patched_modeling_seed_oss_gpu.py",
    description="SeedOss with LigerKernel GPU replacements + VeOmni fused-CE patch",
)

# Surface ``CausalLMOutputWithLogProbs`` in the generated file so the patched
# ``forward`` can return per-token log-probs in the unified output dataclass.
config.add_import("veomni.utils.model_outputs", names=["CausalLMOutputWithLogProbs"])

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    """
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
# 1. OpSlot guard for fused cross-entropy loss; falls back to the eager
#    HF loss path when no fused kernel is bound. Returns the unified
#    ``CausalLMOutputWithLogProbs`` so callers can read per-token
#    log-probs / entropy alongside the loss.
# ================================================================
@config.override_method(
    "SeedOssForCausalLM.forward",
    description="OpSlot guard for fused cross entropy in SeedOssForCausalLM.forward",
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

    loss = None
    logits = None
    log_probs = None
    entropy = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, log_probs, entropy = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, log_probs, entropy = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if log_probs is not None:
                # log_probs path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
