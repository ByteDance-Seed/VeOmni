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
Patch configuration for DeepseekV3 GPU VeomniOp replacements.

Regen command:
patchgen veomni.models.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config -o veomni/models/transformers/deepseek_v3/generated --diff

RMS, apply-RoPE, shared-expert SwiGLU, routed experts, and CausalLM always
call local VeomniOp handles. Deterministic freqs use the local
``triton_bmm`` when ``rotary_pos_emb_implementation`` is ``triton``.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.models.utils.op_utils import (
    merged_experts_act_fn_forward,
    resolve_op_impl,
)
from veomni.ops import VeomniOp
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs
from veomni.utils.moe_monitor import record_router_indices


config = PatchConfig(
    source_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
    target_file="patched_modeling_deepseek_v3_gpu.py",
    description="DeepseekV3 with VeomniOp RMS / RoPE / SwiGLU / MoE / fused loss",
)

config.add_import("functools", names=["partial"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import(
    "veomni.models.utils.op_utils",
    names=[
        "merged_experts_act_fn_forward",
        "resolve_op_impl",
    ],
)
config.add_import(
    "veomni.models.loss_utils",
    names=["ForCausalLMLoss"],
)
config.exclude_from_output("apply_rotary_pos_emb", "apply_rotary_pos_emb_interleave", "rotate_half")
config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)

maybe_autocast = None  # noqa: E305  resolved from the generated modeling file
FlashAttentionKwargs = None  # noqa: E305


@config.add_helper
def _deepseek_v3_rope_op(rope_interleave: bool = False) -> VeomniOp:
    if rope_interleave:
        return VeomniOp("rope", "interleave", "eager")
    impl = resolve_op_impl("rotary_pos_emb_implementation")
    return VeomniOp("rope", "full", "eager" if impl == "triton" else impl)


@config.override_method(
    "DeepseekV3RMSNorm.__init__",
    description="Construct a local rms_norm VeomniOp",
)
def deepseek_v3_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation"))


@config.override_method(
    "DeepseekV3RMSNorm.forward",
    description="Always call the local rms_norm VeomniOp",
)
def deepseek_v3_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


@config.modify_init("DeepseekV3RotaryEmbedding", description="Capture rotary freq impl at construct time")
def deepseek_v3_rotary_embedding_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope_use_triton = resolve_op_impl("rotary_pos_emb_implementation") == "triton"


@config.override_method(
    "DeepseekV3RotaryEmbedding.forward",
    description="Use construct-time triton_bmm choice for deterministic freqs",
)
@torch.no_grad()
def deepseek_v3_rotary_embedding_forward_patched(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
        if self.veomni_rope_use_triton:
            from veomni.models.transformers.deepseek_v3.triton_bmm import triton_bmm

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


@config.modify_init("DeepseekV3Attention", description="Bind instance-local rope and attention VeomniOps")
def deepseek_v3_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = _deepseek_v3_rope_op(self.config.rope_interleave)
    self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)


@config.override_method(
    "DeepseekV3Attention.forward",
    description="Always call the local rope and attention VeomniOps",
)
def deepseek_v3_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_nope = self.kv_a_layernorm(kv_nope)
    kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    q_rot, k_rot = self.veomni_rope(q_rot, k_rot, cos, sin)

    if past_key_values is not None:
        kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states, value_states = self.expand_kv(kv_nope, k_rot)
    attn_output, attn_weights = self.veomni_attn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@config.override_method(
    "DeepseekV3MLP.__init__",
    description="Construct a local swiglu_mlp VeomniOp",
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
    self.veomni_swiglu_mlp = VeomniOp("swiglu_mlp", "standard", resolve_op_impl("swiglu_mlp_implementation"))


@config.override_method(
    "DeepseekV3MLP.forward",
    description="Call swiglu_mlp for silu/swish, otherwise self.act_fn",
)
def deepseek_v3_mlp_forward_patched(self, x):
    if self.config.hidden_act in {"silu", "swish"}:
        return self.veomni_swiglu_mlp(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias if self.gate_proj.bias is not None else self.gate_proj.weight.new_empty(0),
            self.up_proj.weight,
            self.up_proj.bias if self.up_proj.bias is not None else self.up_proj.weight.new_empty(0),
            self.down_proj.weight,
            self.down_proj.bias if self.down_proj.bias is not None else self.down_proj.weight.new_empty(0),
        )
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@config.replace_class("DeepseekV3Experts", description="Always call moe_experts VeomniOp on v5 gate_up_proj weights")
class PatchedDeepseekV3Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.use_swiglu_mlp = config.hidden_act in {"silu", "swish"}
        self.veomni_moe = VeomniOp("moe_experts", "standard", resolve_op_impl("moe_implementation"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_swiglu_mlp:
            return merged_experts_act_fn_forward(
                hidden_states,
                top_k_index,
                top_k_weights,
                self.gate_up_proj,
                self.down_proj,
                self.act_fn,
                self.num_experts,
            )
        unused = self.gate_up_proj.new_empty(0)
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
    hidden_states = hidden_states.view(-1, self.hidden_dim)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores + self.e_score_correction_bias
    group_scores = (
        scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, self.num_group, self.num_experts // self.num_group)
        .reshape(-1, self.num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices


@config.override_method(
    "DeepseekV3MoE.forward",
    description="Report top-k indices to the MoE load-balance monitor",
)
def deepseek_v3_moe_forward_patched(self, hidden_states):
    residuals = hidden_states
    orig_shape = hidden_states.shape
    _, topk_weights, topk_indices = self.gate(hidden_states)
    record_router_indices(self.gate, topk_indices)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states


@config.override_method(
    "DeepseekV3ForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniOp",
)
def deepseek_v3_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = DeepseekV3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_op_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()


@config.override_method(
    "DeepseekV3ForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniOp)",
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
    r"""
    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices depicting the position of input tokens in the sequence. This is
        retained explicitly for callers that pass it positionally.
    """
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
