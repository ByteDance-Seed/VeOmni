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
Patch configuration for Qwen3_5Moe NPU/SP patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_5_moe/generated --diff

Patches applied:
1. Fused MoE expert replacement (merged gate_up_proj layout).
2. Device-agnostic GatedDeltaNet init and varlen FLA forward.
3. DecoderLayer forward with cu_seq_lens_q passthrough.
4. Fused loss + aux_loss in ForConditionalGeneration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config import (
    qwen3_5_gated_deltanet_forward_patched,
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
)
from veomni.ops import fused_moe_forward
from veomni.patchgen.patch_spec import PatchConfig


logger = logging.get_logger(__name__)


config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_npu.py",
    description="Qwen3_5Moe with mojo_opset NPU replacements, fused MoE, and VeOmni SP/fused loss patches",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops", names=["fused_moe_forward"])
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # Modification: We are not using https://github.com/Dao-AILab/causal-conv1d now
    # we are using the triton impl of causal_conv1d from fla.
    # TODO: Evaluate Tridao's impl in the future.
    from functools import wraps

    from mojo_opset import MojoRMSNormFunction, mojo_rope
    from mojo_opset import mojo_causal_conv1d as causal_conv1d_fn
    from mojo_opset_ext import mojo_chunk_gated_delta_rule as chunk_gated_delta_rule
    def causal_conv1d_fn_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.pop("seq_idx")
            kwargs.pop("backend")
            return func(*args, **kwargs)

        return wrapper

    causal_conv1d_fn = causal_conv1d_fn_wrapper(causal_conv1d_fn)

    FusedRMSNormGated = None
    fused_recurrent_gated_delta_rule = None
    causal_conv1d_update = None
    """
)


@config.replace_class(
    "Qwen3_5MoeRMSNorm",
    description="Use eager Qwen3Next-style RMSNorm (1+weight centered formulation) for NPU patchgen",
)
class PatchedQwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        try:
            from mojo_opset import MojoRMSNormFunction

            self._mojo_rms = MojoRMSNormFunction
        except ImportError:
            logger.warning("MojoRMSNormFunction not found, fallback to eager implementation")
            self._mojo_rms = None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self._mojo_rms is None:
            output = self._norm(x.float())
            # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
            # See https://github.com/huggingface/transformers/pull/29402
            output = output * (1.0 + self.weight.float())
            return output.type_as(x)
        else:
            return self._mojo_rms.apply(x, 1.0 + self.weight.float(), self.eps)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# NOTE: apply_rotary_pos_emb is NOT replaced with LigerKernel rotary because
# Qwen3_5Moe uses partial_rotary_factor=0.25 with mrope_interleaved=True.
# The HF implementation correctly handles partial rotary (applying RoPE only
# to the first `rotary_dim` dims and passing through the rest), while
# liger_rotary_pos_emb applies RoPE to the full head_dim, producing incorrect
# results and NaN in attention output.


# ── Propagate _moe_implementation from top-level config to text_config ────────


@config.override_method(
    "Qwen3_5MoeModel.__init__",
    description="Propagate _moe_implementation from top-level config to text_config",
)
def qwen3_5_moe_model_init_patched(self, config):
    # Propagate _moe_implementation so SparseMoeBlock picks up the correct mode.
    moe_implementation = getattr(config, "_moe_implementation", "eager")
    config.text_config._moe_implementation = moe_implementation

    super().__init__(config)
    self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
    self.language_model = Qwen3_5MoeTextModel._from_config(config.text_config)
    self.rope_deltas = None  # cache rope_deltas here

    # Initialize weights and apply final processing
    self.post_init()


# ── SparseMoeBlock forward (avoid in-place op on autograd Function output) ────


@config.override_method(
    "Qwen3_5MoeSparseMoeBlock.forward",
    description="Avoid in-place += on custom autograd Function output",
)
def qwen3_5_moe_sparse_moe_block_forward_patched(
    self, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

    # Modification: use out-of-place add instead of `expert_output += shared_expert_output`
    # to avoid "Output of MergedFc1TritonFusedMoeExpertFunctionBackward is a view and is
    # being modified inplace" RuntimeError from PyTorch autograd.
    expert_output = expert_output + shared_expert_output
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
    return expert_output


# ── MoE Expert replacement (merged gate_up_proj layout) ─────────────────────────


@config.replace_class(
    "Qwen3_5MoeExperts",
    description="Remove @use_experts_implementation decorator and add VeOmni fused MoE dispatch path",
)
class PatchedQwen3_5MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the @use_experts_implementation decorator
    (which routes to grouped_mm and bypasses our fused MoE path) and to add
    VeOmni fused MoE dispatch via _moe_implementation config flag.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # Keep original merged parameter name for checkpoint compatibility.
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
        # Modification: dispatch to fused MoE when _moe_implementation is set.
        # NPU fused MoE kernels are more stable with explicit split+contiguous fc1 weights.
        if self._moe_implementation == "fused":
            intermediate_dim = self.intermediate_dim
            fc1_1_weight = self.gate_up_proj[:, :intermediate_dim, :].contiguous()
            fc1_2_weight = self.gate_up_proj[:, intermediate_dim:, :].contiguous()
            selected_experts = top_k_index.to(torch.int32)
            routing_weights = top_k_weights.to(final_hidden_states.dtype)
            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_states,
                fc1_1_weight=fc1_1_weight,
                fc1_2_weight=fc1_2_weight,
                fc2_weight=self.down_proj,
                fc1_1_2_weight=None,
            )
        elif self._moe_implementation == "eager":
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
                intermediate_dim = self.intermediate_dim
                gate_w = self.gate_up_proj[expert_idx, :intermediate_dim, :]
                up_w = self.gate_up_proj[expert_idx, intermediate_dim:, :]
                gate = nn.functional.linear(current_state, gate_w)
                up = nn.functional.linear(current_state, up_w)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


# ── GatedDeltaNet patches (shared with qwen3_5 via name_map) ─────────────────

_NAME_MAP = {"Qwen3_5": "Qwen3_5Moe"}

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.__init__",
    replacement=qwen3_5_gated_deltanet_init_patched,
    name_map=_NAME_MAP,
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet._get_local_conv1d_weight",
    replacement=qwen3_5_gated_deltanet_get_local_conv1d_weight,
    name_map=_NAME_MAP,
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)

config.override_method(
    "Qwen3_5MoeGatedDeltaNet.forward",
    replacement=qwen3_5_gated_deltanet_forward_patched,
    name_map=_NAME_MAP,
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5MoeGatedDeltaNet.forward",
)


# ── DecoderLayer forward ────────────────────────────────────────────────────────


@config.override_method(
    "Qwen3_5MoeDecoderLayer.forward",
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5MoeDecoderLayer.forward",
)
def qwen3_5_moe_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.FloatTensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Modification: read varlen metadata from kwargs and enforce it for linear-attention varlen kernels.
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    assert cu_seq_lens_q is not None, (
        "cu_seq_lens_q must be provided to support varlen Flash Linear Attention, varlen Conv1D,"
        "and to remove the full Flash Attention CPU-GPU sync."
    )

    # Token Mixer
    if self.layer_type == "linear_attention":
        # Modification: pass cu_seq_lens_q through to Qwen3_5MoeGatedDeltaNet.forward.
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
        )
    elif self.layer_type == "full_attention":
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # For the MoE layers, we need to unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states
    return hidden_states


# ── ForConditionalGeneration forward (fused loss + aux_loss, no vision) ──────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.forward",
    description="Support fused cross entropy path in Qwen3_5MoeForConditionalGeneration.forward",
)
def qwen3_5_moe_forconditional_generation_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:
    # Modification: VeOmni currently supports text-only Qwen3_5Moe.
    if pixel_values is not None or pixel_values_videos is not None:
        raise ValueError(
            "Qwen3_5MoeForConditionalGeneration currently supports text-only inputs in VeOmni; "
            "`pixel_values` and `pixel_values_videos` are not supported yet."
        )

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    return Qwen3_5MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
    )


# ── Expert parallel plan ─────────────────────────────────────────────────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_parallel_plan",
    description="Register Qwen3_5Moe expert parallel plan for v5 generated modeling",
)
def qwen3_5_moe_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
