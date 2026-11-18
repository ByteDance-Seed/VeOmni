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
Patch configuration for GLM-5.3-Flash (``glm5_next``) GPU VeomniOp replacements.

Regen command:
patchgen veomni.models.transformers.glm5_next.glm5_next_gpu_patch_gen_config -o veomni/models/transformers/glm5_next/generated --diff

Eager-first: KDA / DSA / mHC stay on the Hugging Face math path. Routed experts,
dense SwiGLU, RMSNorm, fused CE, and vision attention bind local ``VeomniOp``
handles. EP is wired via ``get_parallel_plan``. Text Ulysses SP all-gathers the
collapsed token stream before each KDA/DSA block (correct, non-fused). Vision
SP uses VeomniOp attention's internal Ulysses path plus SP-sliced rotary
tables; multimodal embeddings are gathered before ``masked_scatter``.
"""

from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.vision_utils import get_vision_attention_seqlens, get_vision_position_ids

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad_and_slice
from veomni.models.loss_utils import ForCausalLMLoss, load_balancing_loss
from veomni.models.utils.moe_utils import merged_experts_act_fn_forward
from veomni.ops import VeomniOp
from veomni.ops.kernels.swiglu_mlp import has_plain_linear_projections
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs
from veomni.utils.moe_monitor import record_router_indices


config = PatchConfig(
    source_module="transformers.models.glm5_next.modeling_glm5_next",
    target_file="patched_modeling_glm5_next_gpu.py",
    description="GLM-5.3-Flash with VeomniOp RMS / SwiGLU / MoE / fused loss, EP + Ulysses SP + VLM metadata",
)

config.drop_import_names("PreTrainedModel")
config.add_import("veomni.models.modeling_base", names=["VeOmniPreTrainedModel as PreTrainedModel"])
config.add_import("functools", names=["partial"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import("veomni.ops.kernels.swiglu_mlp", names=["has_plain_linear_projections"])
config.add_import("veomni.models.utils.moe_utils", names=["merged_experts_act_fn_forward"])
config.add_import(
    "veomni.models.loss_utils",
    names=["ForCausalLMLoss", "load_balancing_loss"],
)
config.add_import("veomni.utils.moe_monitor", names=["record_router_indices"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.add_import(
    "veomni.distributed.parallel_state",
    names=["get_parallel_state"],
)
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"],
)
config.add_import("veomni.utils.device", names=["IS_NPU_AVAILABLE"])
config.drop_import_names("MoeCausalLMOutputWithPast")
# Keep ``rotate_half`` — ``apply_rotary_pos_emb_vision`` depends on it.
config.add_post_import_block("""
import torch.distributed as dist

from veomni.distributed.sequence_parallel import sp_pad_and_slice
from veomni.utils.device import IS_NPU_AVAILABLE
""")


# ── RMSNorm ──────────────────────────────────────────────────────────────


@config.override_method(
    "Glm5NextTextRMSNorm.__init__",
    description="Construct a local rms_norm VeomniOp",
)
def glm5_next_text_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6, config=None) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    impl = "eager" if config is None else config.ops_config.rms_norm_implementation
    self.veomni_rms_norm = VeomniOp("rms_norm", "standard", impl)


@config.override_method(
    "Glm5NextTextRMSNorm.forward",
    description="Always call the local rms_norm VeomniOp",
)
def glm5_next_text_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


@config.override_method(
    "Glm5NextRMSNorm.__init__",
    description="Construct a local rms_norm VeomniOp for vision",
)
def glm5_next_vision_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6, config=None) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    impl = "eager" if config is None else config.ops_config.rms_norm_implementation
    self.veomni_rms_norm = VeomniOp("rms_norm", "standard", impl)


@config.override_method(
    "Glm5NextRMSNorm.forward",
    description="Always call the local rms_norm VeomniOp for vision",
)
def glm5_next_vision_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


# ── Dense MLP (text + vision) with swiglu_limit ──────────────────────────


@config.override_method(
    "Glm5NextTextMLP.__init__",
    description="Construct a local swiglu_mlp VeomniOp with swiglu_limit",
)
def glm5_next_text_mlp_init_patched(self, config, intermediate_size=None):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]
    self.limit = config.swiglu_limit
    self.veomni_swiglu_mlp = VeomniOp("swiglu_mlp", "standard", config.ops_config.swiglu_mlp_implementation)


@config.override_method(
    "Glm5NextTextMLP.forward",
    description="Call swiglu_mlp with swiglu_limit for silu/swish",
)
def glm5_next_text_mlp_forward_patched(self, x):
    if self.config.hidden_act in {"silu", "swish"} and has_plain_linear_projections(self):
        return self.veomni_swiglu_mlp(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias if self.gate_proj.bias is not None else self.gate_proj.weight.new_empty(0),
            self.up_proj.weight,
            self.up_proj.bias if self.up_proj.bias is not None else self.up_proj.weight.new_empty(0),
            self.down_proj.weight,
            self.down_proj.bias if self.down_proj.bias is not None else self.down_proj.weight.new_empty(0),
            swiglu_limit=self.limit,
        )
    gate = self.gate_proj(x).float().clamp(max=self.limit)
    up = self.up_proj(x).float().clamp(min=-self.limit, max=self.limit)
    return self.down_proj((self.act_fn(gate) * up).to(x.dtype))


@config.override_method(
    "Glm5NextVisionMLP.__init__",
    description="Construct a local swiglu_mlp VeomniOp for vision with swiglu_limit",
)
def glm5_next_vision_mlp_init_patched(self, config, bias: bool = True):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
    self.act_fn = ACT2FN[config.hidden_act]
    self.limit = config.swiglu_limit
    self.veomni_swiglu_mlp = VeomniOp("swiglu_mlp", "standard", config.ops_config.swiglu_mlp_implementation)


@config.override_method(
    "Glm5NextVisionMLP.forward",
    description="Call swiglu_mlp with swiglu_limit for vision",
)
def glm5_next_vision_mlp_forward_patched(self, hidden_state):
    if self.config.hidden_act in {"silu", "swish"} and has_plain_linear_projections(self):
        return self.veomni_swiglu_mlp(
            hidden_state,
            self.gate_proj.weight,
            self.gate_proj.bias if self.gate_proj.bias is not None else self.gate_proj.weight.new_empty(0),
            self.up_proj.weight,
            self.up_proj.bias if self.up_proj.bias is not None else self.up_proj.weight.new_empty(0),
            self.down_proj.weight,
            self.down_proj.bias if self.down_proj.bias is not None else self.down_proj.weight.new_empty(0),
            swiglu_limit=self.limit,
        )
    gate = self.gate_proj(hidden_state).float().clamp(max=self.limit)
    up = self.up_proj(hidden_state).float().clamp(min=-self.limit, max=self.limit)
    return self.down_proj((self.act_fn(gate) * up).to(hidden_state.dtype))


# ── Pass config into RMSNorm constructors ────────────────────────────────


@config.override_method(
    "Glm5NextTextDecoderLayer.__init__",
    description="Pass config to Glm5NextTextRMSNorm; keep HF attention/mHC/MoE wiring",
)
def glm5_next_text_decoder_layer_init_patched(self, config, layer_idx: int):
    nn.Module.__init__(self)
    self.block_type = config.layer_types[layer_idx]
    self.hidden_size = config.hidden_size
    self.self_attn = (
        Glm5NextTextLinearAttention(config, layer_idx)  # noqa: F821
        if self.block_type == "linear_attention"
        else Glm5NextTextAttention(config, layer_idx)  # noqa: F821
    )
    self.mlp = (
        Glm5NextTextMoE(config)  # noqa: F821
        if config.mlp_layer_types[layer_idx] == "sparse"
        else Glm5NextTextMLP(config)  # noqa: F821
    )
    self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps, config=config)  # noqa: F821
    self.post_attention_layernorm = Glm5NextTextRMSNorm(  # noqa: F821
        config.hidden_size, config.rms_norm_eps, config=config
    )
    self.attn_hc = Glm5NextTextHyperConnection(config)  # noqa: F821
    self.ffn_hc = Glm5NextTextHyperConnection(config)  # noqa: F821


@config.override_method(
    "Glm5NextTextModel.__init__",
    description="Pass config to final Glm5NextTextRMSNorm",
)
def glm5_next_text_model_init_patched(self, config):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [Glm5NextTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]  # noqa: F821
    )
    self.norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)  # noqa: F821
    self.gradient_checkpointing = False
    self.hc_head = Glm5NextTextHyperHead()  # noqa: F821
    self.post_init()


# ── DecoderLayer: Ulysses gather-before-attn / slice-after ────────────────


@config.override_method(
    "Glm5NextTextDecoderLayer.forward",
    description="Ulysses: all-gather collapsed tokens before attn, slice after; keep HF KDA/DSA/mHC",
)
def glm5_next_text_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = hidden_states.dtype
    parallel_state = get_parallel_state()  # noqa: F821
    ulysses_enabled = bool(getattr(parallel_state, "ulysses_enabled", False))

    residual = hidden_states
    post, comb, hidden_states = self.attn_hc(hidden_states)
    hidden_states = self.input_layernorm(hidden_states)
    topk_indices = None

    if ulysses_enabled:
        # Gather the full sequence for non-fused KDA / DSA, then restore the
        # local shard after attention. EP still applies inside the MoE block.
        hidden_states = gather_outputs(hidden_states, gather_dim=1)  # noqa: F821
        if attention_mask is not None and attention_mask.dim() >= 2:
            attention_mask = gather_outputs(attention_mask, gather_dim=-1)  # noqa: F821
        if position_ids is not None:
            position_ids = gather_outputs(position_ids, gather_dim=-1)  # noqa: F821
        if prev_topk_indices is not None:
            prev_topk_indices = gather_outputs(prev_topk_indices, gather_dim=1)  # noqa: F821

    if self.block_type == "linear_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    else:
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )

    if ulysses_enabled:
        hidden_states = slice_input_tensor(hidden_states, dim=1)  # noqa: F821
        if topk_indices is not None:
            topk_indices = slice_input_tensor(topk_indices, dim=1)  # noqa: F821

    hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )

    residual = hidden_states
    post, comb, hidden_states = self.ffn_hc(hidden_states)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual
    )

    return hidden_states, topk_indices


# ── MoE experts + router monitor ─────────────────────────────────────────


@config.replace_class(
    "Glm5NextTextExperts",
    description="Always call moe_experts VeomniOp on v5 gate_up_proj with swiglu_limit",
)
class PatchedGlm5NextTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.use_swiglu_mlp = config.hidden_act in {"silu", "swish"}
        self.limit = config.swiglu_limit
        self.veomni_moe = VeomniOp("moe_experts", "standard", config.ops_config.moe_implementation)

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
                swiglu_limit=self.limit,
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
            swiglu_limit=self.limit,
        )


@config.override_method(
    "Glm5NextTextMoE.forward",
    description="Report top-k indices to the MoE load-balance monitor",
)
def glm5_next_text_moe_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residuals = hidden_states
    orig_shape = hidden_states.shape
    _, topk_weights, topk_indices = self.gate(hidden_states)
    record_router_indices(self.gate, topk_indices)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states


# ── VLM helpers: 1D position ids + multimodal metadata ───────────────────


@config.add_helper
def get_glm5_next_position_ids(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, torch.LongTensor]:
    """GLM-5.3-Flash has no M-RoPE; expose standard 1D position ids to the VLM pipeline."""
    del attention_mask, kwargs
    position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
    return {"position_ids": position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)}


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad):
    """Derive ``multimodal_metadata`` for the GLM-5.3-Flash ViT (Qwen-VL-family formula).

    Module-level so ``get_metadata_collate_func`` can hand it to VeOmni's
    collator as a picklable callable. Temporal unroll: each ``(t, h, w)``
    expands to ``t`` cu steps of ``h * w`` — matches HF
    ``get_vision_cu_seqlens(..., merge_temporal=False)`` and the flattened
    video grid built inside ``get_video_features``.
    """
    md = {}
    for modality, grid_key, pad_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        grid = batch.get(grid_key)
        if grid is None:
            continue
        grid_list = grid.tolist() if torch.is_tensor(grid) else grid
        if not grid_list:
            continue
        md[f"{modality}_grid_thw_list"] = grid_list
        cu = [0]
        max_hw = 0
        for t, h, w in grid_list:
            hw = h * w
            max_hw = max(max_hw, hw)
            for _ in range(t):
                cu.append(cu[-1] + hw)
        pad = sp_pad.get(pad_key, 0)
        if pad > 0:
            cu.append(cu[-1] + pad)
            max_hw = max(max_hw, pad)
        md[f"vit_{modality}_cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cpu")
        md[f"vit_{modality}_max_seqlen"] = max_hw
    if md:
        batch["multimodal_metadata"] = md


# ── Vision: pass config into RMSNorm + VeomniOp attention ────────────────


# Patch: Glm5NextVisionAttention.__init__
# 1. pass config into q/k RMSNorm so vision can select fused rms_norm
# 2. bind local rope + attention VeomniOps (attention carries Ulysses SP)
# 3. map text-DSA eager/sdpa to veomni_sdpa — vision tokens are SP-sharded
@config.override_method(
    "Glm5NextVisionAttention.__init__",
    description="Pass config to vision RMSNorm; bind SP-aware vision attention VeomniOp",
)
def glm5_next_vision_attention_init_patched(self, config) -> None:
    nn.Module.__init__(self)
    self.dim = config.hidden_size
    self.num_heads = config.num_heads
    self.head_dim = self.dim // self.num_heads
    self.num_key_value_groups = 1
    self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
    self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
    self.scaling = self.head_dim**-0.5
    self.config = config
    self.attention_dropout = config.attention_dropout
    self.is_causal = False
    # --- Patch.1 ---
    self.q_norm = Glm5NextRMSNorm(self.head_dim, eps=config.rms_norm_eps, config=config)  # noqa: F821
    self.k_norm = Glm5NextRMSNorm(self.head_dim, eps=config.rms_norm_eps, config=config)  # noqa: F821
    # --- Patch.1 ---
    # --- Patch.2 ---
    self.veomni_rope = VeomniOp("rope", "full", config.ops_config.rotary_pos_emb_vision_implementation)
    # --- Patch.2 ---
    # --- Patch.3 ---
    # Text DSA keeps gather-before-attn + raw eager/sdpa. Vision patches are
    # already sequence-sharded, so raw eager raises under Ulysses; use a
    # VeOmni attention that performs the exchange.
    attn_impl = config.ops_config.attn_implementation
    if attn_impl in (None, "eager", "sdpa"):
        attn_impl = "veomni_sdpa"
    self.veomni_attn = VeomniOp("attention", "standard", attn_impl)
    # --- Patch.3 ---


# Patch: Glm5NextVisionAttention.forward
# 1. route through local rope + attention VeomniOps (Ulysses-aware)
# 2. accept precomputed max_seqlen to avoid per-layer cu_seqlens sync
@config.override_method(
    "Glm5NextVisionAttention.forward",
    description="Call local rope/attention VeomniOps; consume precomputed max_seqlen",
)
def glm5_next_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_seqlen: int | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)
    cos, sin = position_embeddings
    # --- Patch.1 ---
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin)
    # --- Patch.1 ---
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    # --- Patch.2 ---
    attn_output, _ = self.veomni_attn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )
    # --- Patch.2 ---
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


# Patch: Glm5NextVisionBlock.__init__
# 1. pass config into norm1/norm2 RMSNorm constructors
@config.override_method(
    "Glm5NextVisionBlock.__init__",
    description="Pass config to vision block RMSNorm constructors",
)
def glm5_next_vision_block_init_patched(self, config) -> None:
    nn.Module.__init__(self)
    # --- Patch.1 ---
    self.norm1 = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)  # noqa: F821
    self.norm2 = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)  # noqa: F821
    # --- Patch.1 ---
    self.attn = Glm5NextVisionAttention(config)  # noqa: F821
    self.mlp = Glm5NextVisionMLP(config, bias=config.attention_bias)  # noqa: F821


# Patch: Glm5NextVisionBlock.forward
# 1. thread precomputed max_seqlen into attention
@config.override_method(
    "Glm5NextVisionBlock.forward",
    description="Propagate precomputed max_seqlen to vision attention",
)
def glm5_next_vision_block_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Args:
        hidden_states (`torch.Tensor`):
            Input to the block.
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths for packed vision tokens.
        max_seqlen (`int`, *optional*):
            Max segment length for varlen attention (precomputed to avoid per-layer sync).
        position_embeddings (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
            Tuple of `(cos, sin)` rotary embeddings.
    """
    residual = hidden_states
    hidden_states = self.norm1(hidden_states)
    hidden_states = self.attn(
        hidden_states,
        cu_seqlens=cu_seqlens,
        # --- Patch.1 ---
        max_seqlen=max_seqlen,
        # --- Patch.1 ---
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


# Patch: Glm5NextVisionModel.__init__
# 1. pass config into post_layernorm RMSNorm
@config.override_method(
    "Glm5NextVisionModel.__init__",
    description="Pass config to vision post_layernorm RMSNorm",
)
def glm5_next_vision_model_init_patched(self, config) -> None:
    super().__init__(config)
    self.spatial_merge_size = config.spatial_merge_size
    self.patch_size = config.patch_size
    self.patch_embed = Glm5NextVisionPatchEmbed(config)  # noqa: F821
    head_dim = config.hidden_size // config.num_heads
    self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)  # noqa: F821
    self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])  # noqa: F821
    self.merger = Glm5NextVisionPatchMerger(  # noqa: F821
        dim=config.out_hidden_size,
        context_dim=config.projection_intermediate_size,
        hidden_act=config.hidden_act,
        swiglu_limit=config.swiglu_limit,
    )
    self.downsample = nn.Conv2d(
        in_channels=config.hidden_size,
        out_channels=config.out_hidden_size,
        kernel_size=config.spatial_merge_size,
        stride=config.spatial_merge_size,
    )
    # --- Patch.1 ---
    self.post_layernorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps, config=config)  # noqa: F821
    # --- Patch.1 ---
    self.gradient_checkpointing = False
    self.post_init()


# Patch: Glm5NextVisionModel.forward
# 1. consume vit_metadata (cu_seqlens / max_seqlen) with runtime fallback
# 2. SP-slice rotary tables to match per-rank pixel shards (pad_scale=4)
# 3. extend cu_seqlens with SP-pad tail when metadata was not precomputed
# 4. move cu_seqlens to CPU on NPU (FA2 varlen requirement)
@config.override_method(
    "Glm5NextVisionModel.forward",
    description="VeOmni SP + precomputed ViT metadata for GLM-5.3-Flash vision",
)
def glm5_next_vision_forward_patched(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> BaseModelOutputWithPooling:
    r"""
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            Flattened vision patch features.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image/video.
    """
    # --- Patch.1 ---
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
    precomputed_max_seqlen = vit_metadata.get("max_seqlen")
    # --- Patch.1 ---

    # Position ids always come from the grid actually passed in (image raw, or
    # video flattened by get_video_features) — metadata grid_thw_list stays
    # unflattened for video and must not override this path.
    position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)

    if precomputed_cu_seqlens is not None:
        cu_seqlens = precomputed_cu_seqlens.to(
            hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
        max_seqlen = precomputed_max_seqlen
    else:
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

    # Host-side patch count from the grid actually passed in (image raw or
    # video flattened). Do not use cu_seqlens[-1]: precomputed cu may already
    # include an SP-pad tail that is not content tokens.
    grid_list = grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw
    total_seq_len = sum(t * h * w for t, h, w in grid_list)

    hidden_states = self.patch_embed(hidden_states)
    rotary_emb = self.rotary_pos_emb(position_ids)
    emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    seq_len, _ = hidden_states.size()
    sp_pad_seq_len = 0
    # --- Patch.2 ---
    if get_parallel_state().sp_enabled:
        cos, sin = position_embeddings
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)
        sp_size = get_parallel_state().sp_size
        sp_pad_seq_len = seq_len * sp_size - total_seq_len
        # --- Patch.3 ---
        if sp_pad_seq_len > 0 and precomputed_cu_seqlens is None:
            new_cumsum = cu_seqlens[-1] + sp_pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
            if max_seqlen is not None:
                max_seqlen = max(max_seqlen, sp_pad_seq_len)
        # --- Patch.3 ---
    # --- Patch.2 ---

    if precomputed_max_seqlen is not None:
        max_seqlen = precomputed_max_seqlen
    elif max_seqlen is None:
        max_seqlen = max((int(c2 - c1) for c1, c2 in zip(cu_seqlens[:-1], cu_seqlens[1:])), default=0)

    # --- Patch.4 ---
    if IS_NPU_AVAILABLE:
        cu_seqlens = cu_seqlens.cpu()
    # --- Patch.4 ---

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.post_layernorm(hidden_states)
    hidden_states = hidden_states.view(-1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1])
    hidden_states = hidden_states.permute(0, 3, 1, 2)
    hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
    merged_hidden_states = self.merger(hidden_states)
    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


# Patch: Glm5NextVisionModel.dummy_forward (new)
# 1. FSDP ranks without pixel_values still run the visual encoder
# 2. SP-aware grid (height scaled by sp_size) + pad_scale=4 pixel slice
# 3. host-side vit_metadata so dummy forward skips sync-heavy fallback
@config.override_method(
    "Glm5NextVisionModel.dummy_forward",
    description="Provide dummy vision forward for FSDP with SP-aware shapes",
)
def glm5_next_vision_dummy_forward_patched(self):
    patch_size = self.config.patch_size
    temporal_patch_size = self.config.temporal_patch_size
    in_channels = self.config.in_channels
    merge_size = self.spatial_merge_size
    t = 1
    h_base = 2 * merge_size
    w = 2 * merge_size
    # --- Patch.2 ---
    if get_parallel_state().sp_enabled:
        h = h_base * get_parallel_state().sp_size
    else:
        h = h_base
    # --- Patch.2 ---
    num_patches = t * h * w
    pixel_row_size = in_channels * temporal_patch_size * patch_size * patch_size
    # Live parameter dtype — never cache across calls under FSDP2+MP.
    dtype = self.patch_embed.proj.weight.dtype
    device = self.patch_embed.proj.weight.device
    pixel_values = torch.zeros((num_patches, pixel_row_size), dtype=dtype, device=device)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.int32, device=device)
    # --- Patch.2 ---
    if get_parallel_state().sp_enabled:
        pixel_values = sp_pad_and_slice(pixel_values, dim=0, pad_value=0, pad_scale=4)
    # --- Patch.2 ---
    # --- Patch.3 ---
    cu = [0]
    for _ in range(t):
        cu.append(cu[-1] + h * w)
    vit_metadata = {
        "grid_thw_list": [[t, h, w]],
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cpu"),
        "max_seqlen": h * w,
    }
    # --- Patch.3 ---
    return self(hidden_states=pixel_values, grid_thw=grid_thw, vit_metadata=vit_metadata)


# Patch: Glm5NextModel.get_image_features / get_video_features
# 1. keep pooler_output flat (skip per-image torch.split) so SP gather works
@config.override_method(
    "Glm5NextModel.get_image_features",
    description="Return flat image embeds for VeOmni SP-aware masked_scatter",
)
def glm5_next_model_get_image_features_patched(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPooling:
    pixel_values = pixel_values.type(self.visual.dtype)
    return self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)


@config.override_method(
    "Glm5NextModel.get_video_features",
    description="Return flat video embeds; keep HF temporal flatten of grid_thw",
)
def glm5_next_model_get_video_features_patched(
    self,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPooling:
    t = video_grid_thw[:, 0]
    hw = video_grid_thw[:, 1:]
    flattened_hw = torch.repeat_interleave(hw, t, dim=0)
    prefix_ones = video_grid_thw.new_ones(flattened_hw.shape[0], 1)
    flattened_video_grid_thw = torch.cat([prefix_ones, flattened_hw], dim=1)
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    return self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw, **kwargs)


# Patch: Glm5NextModel.get_placeholder_mask
# 1. return raw bool masks without feature-count validation (SP seq lens differ)
@config.override_method(
    "Glm5NextModel.get_placeholder_mask",
    description="Return raw image/video placeholder bool masks for VeOmni SP scatter",
)
def glm5_next_model_get_placeholder_mask_patched(
    self,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor | None = None,
    image_features: torch.FloatTensor | None = None,
    video_features: torch.FloatTensor | None = None,
):
    del inputs_embeds, image_features, video_features
    special_mm_mask = input_ids == self.config.image_token_id
    in_video_span = (input_ids == self.config.video_start_token_id).cumsum(-1) > (
        input_ids == self.config.video_end_token_id
    ).cumsum(-1)
    return special_mm_mask & ~in_video_span, special_mm_mask & in_video_span


# Patch: Glm5NextModel.forward
# 1. SP gather embeds + vision features, scatter, then slice back
# 2. prefer collator image_mask/video_mask; gather input_ids fallback under SP
# 3. thread vit_metadata from multimodal_metadata
# 4. FSDP dummy_forward when a rank has no pixels
# 5. fail fast if position_ids is None under SP (no on-the-fly mrope here)
@config.override_method(
    "Glm5NextModel.forward",
    description="VeOmni SP + dummy-forward + multimodal_metadata for GLM-5.3-Flash",
)
def glm5_next_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.2 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
    del mm_token_type_ids  # accepted for HF processor compatibility; unused (no M-RoPE)
    if video_mask is None and image_mask is None and input_ids is not None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for _ in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids_full = torch.cat(input_ids_list, dim=1)
        else:
            input_ids_full = input_ids
        image_mask, video_mask = self.get_placeholder_mask(input_ids_full)
    # --- Patch.2 ---

    # --- Patch.3 ---
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    image_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_image_max_seqlen"),
        }
    }
    video_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": multimodal_metadata.get("video_grid_thw_list"),
            "cu_seqlens": multimodal_metadata.get("vit_video_cu_seqlens"),
            "max_seqlen": multimodal_metadata.get("vit_video_max_seqlen"),
        }
    }
    # --- Patch.3 ---

    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=get_parallel_state().sp_group)
    # --- Patch.1 ---

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw, **image_vit_kwargs).pooler_output
        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
        # --- Patch.1 ---
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.4 ---

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **video_vit_kwargs).pooler_output
        # --- Patch.1 ---
        if get_parallel_state().sp_enabled:
            video_embeds = gather_outputs(video_embeds, gather_dim=0, group=get_parallel_state().sp_group)
        # --- Patch.1 ---
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.4 ---
        fake_vision = self.visual.dummy_forward()
        fake_embeds = fake_vision.pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.4 ---

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=get_parallel_state().sp_group)
    # --- Patch.1 ---

    # --- Patch.5 ---
    if position_ids is None and get_parallel_state().sp_enabled:
        raise RuntimeError(
            "Glm5NextModel.forward: position_ids is None while sequence parallel "
            "is enabled; position_ids must be precomputed via get_position_id_func "
            "in the VeOmni data pipeline."
        )
    # --- Patch.5 ---

    kwargs.update(flash_attn_kwargs)
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    return MoeModelOutputWithPast(  # noqa: F821
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


# ── Parallel plans ───────────────────────────────────────────────────────


@config.override_method(
    "Glm5NextForConditionalGeneration.get_parallel_plan",
    description="Register GLM-5.3-Flash expert parallel plan for the VLM wrapper",
)
def glm5_next_forcg_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


@config.override_method(
    "Glm5NextTextModel.get_parallel_plan",
    description="Register GLM-5.3-Flash expert parallel plan for standalone text",
)
def glm5_next_text_get_parallel_plan_patched(self):
    from ..parallel_plan import get_text_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


@config.override_method(
    "Glm5NextForConditionalGeneration.get_position_id_func",
    description="Expose standard 1D position ids to the VeOmni VLM data pipeline",
)
def glm5_next_get_position_id_func_patched(self):
    return get_glm5_next_position_ids  # noqa: F821 defined via add_helper


@config.override_method(
    "Glm5NextForConditionalGeneration.get_metadata_collate_func",
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)
def glm5_next_get_metadata_collate_func_patched(self):
    return collate_multimodal_metadata  # noqa: F821 defined via add_helper


# ── ForConditionalGeneration: fused CE + aux loss ────────────────────────


@config.override_method(
    "Glm5NextForConditionalGeneration.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniOps",
)
def glm5_next_forcg_init_patched(self, config):
    super().__init__(config)
    self.model = Glm5NextModel(config)  # noqa: F821
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
    self.num_experts = config.text_config.num_local_experts
    self.num_experts_per_tok = config.text_config.num_experts_per_tok
    impl = config.ops_config.cross_entropy_loss_implementation
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.veomni_lb = VeomniOp(
        "load_balancing_loss",
        "standard",
        config.ops_config.load_balancing_loss_implementation,
    )
    self.load_balancing_loss = partial(load_balancing_loss, op=self.veomni_lb)
    self.post_init()


@config.override_method(
    "Glm5NextForConditionalGeneration.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniOps; forward VLM kwargs",
)
def glm5_next_forcg_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    output_router_logits: bool | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    mm_encoder_outputs: dict | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    pixel_values (`torch.Tensor`, *optional*):
        Image pixel values.
    pixel_values_videos (`torch.FloatTensor`, *optional*):
        Video pixel values.
    image_grid_thw (`torch.LongTensor`, *optional*):
        Image grid metadata.
    video_grid_thw (`torch.LongTensor`, *optional*):
        Video grid metadata.
    output_router_logits (`bool`, *optional*):
        Whether to return router logits for the aux load-balancing loss.
    mm_token_type_ids (`torch.IntTensor`, *optional*):
        Multimodal token type ids (accepted for HF processor compatibility).
    logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
        If an `int`, compute logits for only the last `logits_to_keep` tokens.
    mm_encoder_outputs (`dict`, *optional*):
        Optional precomputed multimodal encoder outputs.
    """
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
    )

    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    multimodal_metadata = kwargs.pop("multimodal_metadata", None)

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
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        mm_encoder_outputs=mm_encoder_outputs,
        mm_token_type_ids=mm_token_type_ids,
        image_mask=image_mask,
        video_mask=video_mask,
        multimodal_metadata=multimodal_metadata,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
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
    if output_router_logits:
        aux_loss = self.load_balancing_loss(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if isinstance(loss, torch.Tensor) and isinstance(aux_loss, torch.Tensor):
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )
