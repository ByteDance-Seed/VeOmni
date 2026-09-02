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
# See the License for the specific language governing limitations
# under the License.
"""
Patch configuration for DeepseekV3 GPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config -o veomni/models_kernel/transformers/deepseek_v3/generated --diff

RMS, apply-RoPE, shared-expert SwiGLU, routed experts, and CausalLM always
call local VeomniKernel handles. Deterministic freqs use the local
``triton_bmm`` when ``rotary_pos_emb_implementation`` is ``triton``.
"""

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import (
    empty_bias,
    linear_bias,
    resolve_kernel_impl,
    resolve_moe_impl,
)
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs
from veomni.utils.moe_monitor import record_router_indices


config = PatchConfig(
    source_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
    target_file="patched_modeling_deepseek_v3_gpu.py",
    description="DeepseekV3 with VeomniKernel RMS / RoPE / SwiGLU / MoE / fused loss",
)

config.add_import("functools", names=["partial"])
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["empty_bias", "linear_bias", "resolve_kernel_impl", "resolve_moe_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)
config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)


@config.override_method(
    "DeepseekV3RMSNorm.__init__",
    description="Construct a local rms_norm VeomniKernel",
)
def deepseek_v3_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniKernel("rms_norm", "standard", resolve_kernel_impl("rms_norm_implementation"))


@config.override_method(
    "DeepseekV3RMSNorm.forward",
    description="Always call the local rms_norm VeomniKernel",
)
def deepseek_v3_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


@config.override_method(
    "DeepseekV3RotaryEmbedding.forward",
    description="Use local triton_bmm for deterministic freqs when rotary impl is triton",
)
@torch.no_grad()
def deepseek_v3_rotary_embedding_forward_patched(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        if resolve_kernel_impl("rotary_pos_emb_implementation") == "triton":
            from veomni.models_kernel.transformers.deepseek_v3.triton_bmm import triton_bmm

            freqs = triton_bmm(
                inv_freq_expanded.float().contiguous(),
                position_ids_expanded.float().contiguous(),
            ).transpose(1, 2)
        else:
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@config.replace_function("apply_rotary_pos_emb", description="Always call rope full VeomniKernel")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    impl = resolve_kernel_impl("rotary_pos_emb_implementation")
    rope = VeomniKernel("rope", "full", "eager" if impl == "triton" else impl)
    return rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


rotate_half = None  # noqa: E305  resolved from the generated modeling file


@config.override_method(
    "DeepseekV3MLP.__init__",
    description="Construct a local swiglu_mlp VeomniKernel",
)
def deepseek_v3_mlp_init_patched(self, config, intermediate_size=None):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]
    self.veomni_swiglu_mlp = VeomniKernel("swiglu_mlp", "standard", resolve_kernel_impl("swiglu_mlp_implementation"))


@config.override_method(
    "DeepseekV3MLP.forward",
    description="Always call the local swiglu_mlp VeomniKernel",
)
def deepseek_v3_mlp_forward_patched(self, x):
    return self.veomni_swiglu_mlp(
        x,
        self.gate_proj.weight,
        linear_bias(self.gate_proj),
        self.up_proj.weight,
        linear_bias(self.up_proj),
        self.down_proj.weight,
        linear_bias(self.down_proj),
    )


@config.replace_class(
    "DeepseekV3NaiveMoe", description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights"
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
        self.veomni_moe = VeomniKernel("moe_experts", "standard", resolve_moe_impl())

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        unused = empty_bias(self.gate_up_proj)
        return self.veomni_moe(
            hidden_states,
            top_k_weights,
            top_k_index,
            unused,
            unused,
            self.down_proj,
            self.gate_up_proj,
            num_experts=self.num_experts,
        )


@config.override_method(
    "DeepseekV3TopkRouter.forward",
    description="Disable autocast around fp32 router linear for VeRL actor/rollout parity",
)
def deepseek_v3_topk_router_forward_patched(self, hidden_states):
    hidden_states = hidden_states.view(-1, self.config.hidden_size)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
    return router_logits


@config.override_method(
    "DeepseekV3MoE.forward",
    description="Report top-k indices to the MoE load-balance monitor",
)
def deepseek_v3_moe_forward_patched(self, hidden_states):
    residuals = hidden_states
    orig_shape = hidden_states.shape
    router_logits = self.gate(hidden_states)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    record_router_indices(self.gate, topk_indices)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states


@config.override_method(
    "DeepseekV3ForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
def deepseek_v3_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = DeepseekV3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.post_init()


@config.override_method(
    "DeepseekV3ForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
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

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@config.override_method(
    "DeepseekV3ForCausalLM.get_parallel_plan",
    description="Register DeepseekV3 expert parallel plan for v5 generated modeling",
)
def deepseek_v3_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


_ = (Callable, Optional)
