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
Patch configuration for DeepseekV3 GPU patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config -o veomni/models/transformers/deepseek_v3/generated --diff

v5 patches:
1. DeepseekV3NaiveMoe fused expert forward (eager + fused branches) — drops
   upstream ``@use_experts_implementation`` which otherwise routes around our
   fused MoE kernel.
2. DeepseekV3TopkRouter.forward restores ``torch.autocast(enabled=False)``
   around the fp32 router F.linear, required for VeRL actor/rollout parity.
3. DeepseekV3ForCausalLM.forward fused cross-entropy path.
4. Register get_parallel_plan on DeepseekV3ForCausalLM.

Liger kernels (RMSNorm / SwiGLU MLP / rotary) and batch-invariant RMSNorm +
deterministic Triton RoPE are applied at runtime from ``__init__.py`` based on
``VEOMNI_USE_LIGER_KERNEL`` — not baked into the generated file.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.ops import fused_moe_forward
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
    target_file="patched_modeling_deepseek_v3_gpu.py",
    description="DeepseekV3 with LigerKernel GPU replacements and VeOmni fused-MoE / fused-loss patches",
)

config.add_import("veomni.ops", names=["fused_moe_forward"])


# NOTE: Liger replacements (RMSNorm / SwiGLU MLP / apply_rotary_pos_emb) are
# intentionally NOT applied to the v5 generated file. DeepseekV3 runs on the
# deterministic Triton RoPE + batch-invariant RMSNorm kernels wired at runtime
# from ``__init__.py`` so actor/rollout numerics stay aligned; LigerSwiGLUMLP
# additionally rejects the ``intermediate_size`` kwarg used by shared_experts.


# ================================================================
# Patch: DeepseekV3NaiveMoe
# 1. Drop upstream ``@use_experts_implementation`` decorator — it dispatches
#    to ``grouped_mm`` / HF fused paths and bypasses VeOmni's fused MoE.
# 2. Add ``_moe_implementation`` branching: ``eager`` loops over hit experts,
#    ``fused`` calls ``fused_moe_forward`` with stacked ``gate_up_proj``.
# Layout matches v5 upstream (direct, no transpose):
#   gate_up_proj [E, 2*I, H],  down_proj [E, H, I]
# ================================================================
@config.replace_class(
    "DeepseekV3NaiveMoe",
    description="Use v5 gate_up_proj expert layout with explicit VeOmni fused-MoE path",
)
class PatchedDeepseekV3NaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        if self._moe_implementation == "eager":
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        elif self._moe_implementation == "fused":
            top_k_weights = top_k_weights.to(final_hidden_states.dtype)
            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights,
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=None,
                fc1_2_weight=None,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=self.gate_up_proj,
            )
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


# ================================================================
# Patch: DeepseekV3TopkRouter.forward
# 1. Wrap the router F.linear in ``torch.autocast(enabled=False)`` so the
#    explicit fp32 cast isn't silently reverted by an outer autocast context.
#    Required for VeRL actor/rollout numerical parity.
# ================================================================
@config.override_method(
    "DeepseekV3TopkRouter.forward",
    description="Disable autocast around fp32 router linear for VeRL actor/rollout parity",
)
def deepseek_v3_topk_router_forward_patched(self, hidden_states):
    hidden_states = hidden_states.view(-1, self.config.hidden_size)
    # --- Patch.1 ---
    # Disable autocast to ensure fp32 computation — autocast overrides
    # explicit .type(torch.float32) in F.linear, causing precision mismatch
    # between actor (autocast bf16) and rollout (no autocast, native fp32).
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
    # --- Patch.1 ---
    return router_logits


# ================================================================
# Patch: DeepseekV3ForCausalLM.forward
# 1. Route loss through ``self.loss_function(..., hidden_states=..., weights=...)``
#    so fused CE can avoid materializing the full logits tensor.
# ================================================================
@config.override_method(
    "DeepseekV3ForCausalLM.forward",
    description="Support fused cross entropy path in DeepseekV3ForCausalLM.forward",
)
def deepseek_v3_forcausallm_forward_patched(
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
    outputs: BaseModelOutputWithPast = self.model(
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

    # --- Patch.1 ---
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
    # --- Patch.1 ---

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ================================================================
# Patch: DeepseekV3ForCausalLM.get_parallel_plan
# 1. Register VeOmni EP parallel plan on the v5 generated class.
# ================================================================
@config.override_method(
    "DeepseekV3ForCausalLM.get_parallel_plan",
    description="Register DeepseekV3 expert parallel plan for v5 generated modeling",
)
def deepseek_v3_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


# Silence unused import warnings for symbols referenced only in type hints.
_ = (Callable,)
