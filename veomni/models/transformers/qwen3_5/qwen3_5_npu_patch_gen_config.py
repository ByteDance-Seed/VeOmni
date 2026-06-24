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
Patch configuration for Qwen3_5 NPU/SP patched modeling generation.

Regen command:
patchgen veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config -o veomni/models/transformers/qwen3_5/generated --diff

Language-model focused patches from qwen3_next example:
1. Device-agnostic GatedDeltaNet init and varlen FLA forward.
2. DecoderLayer forward with cu_seq_lens_q passthrough.
3. Use VeOmni fused loss path in Qwen3_5ForConditionalGeneration.forward.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    apply_mask_to_padding_states,
)

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    _Qwen3_5FakeForPosID,
    collate_multimodal_metadata,
    get_position_id,
    mm_token_type_ids_from_input_ids,
    qwen3_5_decoder_layer_forward_patched,
    qwen3_5_forcausallm_forward_patched,
    qwen3_5_forconditional_generation_forward_patched,
    qwen3_5_forconditional_generation_get_metadata_collate_func,
    qwen3_5_forconditional_generation_get_position_id_func,
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
    qwen3_5_model_forward,
    qwen3_5_model_get_image_features,
    qwen3_5_model_get_placeholder_mask,
    qwen3_5_rmsnorm_forward_patched,
    qwen3_5_text_model_update_linear_attn_mask,
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
    qwen3_5_vision_model_rot_pos_emb,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (  # noqa: F401  consumed by in-config dataclass + emitted forward
    FusedLinearAuxOutput,
    FusedLinearAuxOutputMixin,
)


config = PatchConfig(
    source_module="transformers.models.qwen3_5.modeling_qwen3_5",
    target_file="patched_modeling_qwen3_5_npu.py",
    description="Qwen3_5 with VeOmni language-model SP and fused loss patches",
)

config.add_import("copy", names=["copy"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)

# gather_outputs / slice_input_tensor live in veomni.distributed.sequence_parallel.data
# (re-exported by the package __init__), not in .ulysses.
config.add_import(
    "veomni.distributed.sequence_parallel", names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"]
)
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward`` (re-used
# from the GPU config) can return per-token log-probs in the unified output
# dataclass.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)  # noqa: F401
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # NPU has no fla/flash_qla backend registered today; selecting a non-eager
    # linear-attention impl raises at OpSlot.bind() time, which is desirable —
    # a silent fallback would mask the misconfiguration. These None
    # placeholders preserve the upstream HF top-level
    # `is_fast_path_available = all((causal_conv1d_fn, ...))` (resolves to
    # False — legacy warning) and let the `<fla_name> or <torch_fallback>`
    # assignments in __init__ resolve to torch.
    FusedRMSNormGated = None
    causal_conv1d_fn = None
    causal_conv1d_update = None
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None
    """
)

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "partial")
    veomni_apply_rotary_pos_emb_vision = OpSlot("rotary_pos_emb_vision", "full")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_rms_norm_gated = OpSlot("rms_norm_gated", "standard")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")
    """
)


# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
torch_chunk_gated_delta_rule = None  # noqa: F811 — also imported above for the forward patch
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None
gather_outputs = None
slice_input_tensor = None
veomni_rms_norm_gated = None  # OpSlot, declared in post-import block above
veomni_causal_conv1d = None  # OpSlot, declared in post-import block above
veomni_chunk_gated_delta_rule = None  # OpSlot, declared in post-import block above

# This NPU config reuses qwen3_5_vision_model_forward (Patch.5) from the GPU
# config but does NOT register the Qwen3_5VisionAttention.forward consumer —
# NPU vision attention runs the upstream HF body which recomputes max_seqlen
# itself. Setting the sentinel to False suppresses Patch.5's host sync /
# kwarg leak into `attention_interface(**kwargs)` on NPU.
config.add_post_import_block("_VEOMNI_VISION_ATTENTION_PATCHED = False")


# Register the multimodal helpers used by the reused get_position_id_func /
# get_metadata_collate_func / Model.forward bodies. Defined in
# qwen3_5_gpu_patch_gen_config.py (imported above) and referenced by name
# in the reused function bodies, so the NPU generated file must emit them.
# (qwen3_5_npu doesn't wholesale `config.helpers.extend(gpu_config.helpers)`
# the way qwen3_vl_npu does; it picks functions à la carte, so each helper
# has to be registered explicitly here. `mm_token_type_ids_from_input_ids`
# in particular is called from `get_position_id` and the Model.forward
# multimodal-RoPE path — both required since transformers v5.)
config.add_helper(mm_token_type_ids_from_input_ids)
config.add_helper(get_position_id)
config.add_helper(collate_multimodal_metadata)
config.add_helper(_Qwen3_5FakeForPosID)


config.override_method(
    "Qwen3_5RMSNorm.forward",
    replacement=qwen3_5_rmsnorm_forward_patched,
    description="Use fused rmsnorm to impl zero-centered rmsnorm (1+weight centered formulation)",
)


@config.replace_function(
    "apply_rotary_pos_emb",
    description="Use fused rope to impl partial rotary postion embedding",
)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


@config.replace_function(
    "apply_rotary_pos_emb_vision", description="Use fused rope to impl rotary postion embedding in vit"
)
def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if veomni_apply_rotary_pos_emb_vision.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb_vision(q, k, cos, sin)
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


config.override_method(
    "Qwen3_5GatedDeltaNet.__init__",
    replacement=qwen3_5_gated_deltanet_init_patched,
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)


config.override_method(
    "Qwen3_5GatedDeltaNet._get_local_conv1d_weight",
    replacement=qwen3_5_gated_deltanet_get_local_conv1d_weight,
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)


@config.override_method(
    "Qwen3_5GatedDeltaNet.forward",
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5GatedDeltaNet.forward",
)
def qwen3_5_gated_deltanet_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    # Modification: plumb varlen sequence metadata to FLA kernels.
    cu_seq_lens_q: torch.Tensor | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )

    # getting projected states from cache if it exists
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Modification: Ulysses SP all-to-all for linear attention heads.
    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled:
        ulysses_group = get_parallel_state().ulysses_group
        ulysses_size = get_parallel_state().ulysses_size
        ulysses_rank = get_parallel_state().ulysses_rank
        assert self.num_k_heads % ulysses_size == 0 and self.num_v_heads % ulysses_size == 0, (
            f"SP size ({ulysses_size}) must divide num_k_heads ({self.num_k_heads}) "
            f"and num_v_heads ({self.num_v_heads}) for gated deltanet LASP"
        )

        local_num_k_heads = self.num_k_heads // ulysses_size
        local_num_v_heads = self.num_v_heads // ulysses_size
        local_key_dim = self.head_k_dim * local_num_k_heads
        local_value_dim = self.head_v_dim * local_num_v_heads

        # Reshape mixed_qkv to head layout for all-to-all: [B, S_local, D] -> split+reshape to heads
        q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # All-to-all: gather full sequence, scatter heads -> [B, S_full, local_heads, head_dim]
        q_proj = gather_seq_scatter_heads(q_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        k_proj = gather_seq_scatter_heads(k_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        v_proj = gather_seq_scatter_heads(v_proj, seq_dim=1, head_dim=2, group=ulysses_group)

        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2, group=ulysses_group)
        a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2, group=ulysses_group)

        # Flatten heads back to channels and concat for conv1d: [B, S_full, local_dim]
        q_proj = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], -1)
        k_proj = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], -1)
        v_proj = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], -1)
        mixed_qkv = torch.cat((q_proj, k_proj, v_proj), dim=-1)
    else:
        local_num_k_heads = self.num_k_heads
        local_num_v_heads = self.num_v_heads
        local_key_dim = self.key_dim
        local_value_dim = self.value_dim

    if use_precomputed_states:
        # Modification: keep this disabled until FLA causal_conv1d_update decode path is validated.
        raise NotImplementedError("use_precomputed_states=True is not supported yet for causal_conv1d_update now.")
    else:
        if cache_params is not None:
            mixed_qkv_t = mixed_qkv.transpose(1, 2)
            conv_state = F.pad(mixed_qkv_t, (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            # Modification: shard conv1d weights per Ulysses rank to match head-sharded channels.
            if ulysses_enabled:
                conv_weight = self._get_local_conv1d_weight(
                    ulysses_rank=ulysses_rank,
                    local_key_dim=local_key_dim,
                    local_value_dim=local_value_dim,
                )
            else:
                conv_weight = self.conv1d.weight.squeeze(1)
            # mixed_qkv is [B, S, D] — FLA causal_conv1d expects [B, S, D].
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=conv_weight,
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
                backend="triton",
                cu_seqlens=cu_seq_lens_q.npu(),
            )[0]
        else:
            raise NotImplementedError("This path is not supported yet because it can't process varlen now.")

    query, key, value = torch.split(
        mixed_qkv,
        [
            local_key_dim,
            local_key_dim,
            local_value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(query.shape[0], query.shape[1], local_num_k_heads, self.head_k_dim)
    key = key.reshape(key.shape[0], key.shape[1], local_num_k_heads, self.head_k_dim)
    value = value.reshape(value.shape[0], value.shape[1], local_num_v_heads, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    # Modification: slice A_log/dt_bias for local V-heads under Ulysses SP.
    if ulysses_enabled:
        v_head_offset = ulysses_rank * local_num_v_heads
        v_head_slice = slice(v_head_offset, v_head_offset + local_num_v_heads)
        g = -self.A_log[v_head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[v_head_slice])
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        # Modification: instance-local guard (see GPU patch comment).
        if self.chunk_gated_delta_rule is torch_chunk_gated_delta_rule:
            raise RuntimeError(
                "Varlen Qwen3.5 GatedDeltaNet training is GPU-only — NPU has no fla/flash_qla "
                "backend registered today. On GPU, set chunk_gated_delta_rule_implementation='fla' "
                "(and install flash-linear-attention) or 'flash_qla' (ships under the gpu extra, "
                "Hopper sm90 only) in OpsImplementationConfig."
            )
        else:
            # Modification: use direct args and pass cu_seqlens for varlen FLA attention.
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seq_lens_q.npu(),
            )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    # Update cache
    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    # Modification: gather attention output back to sequence-sharded layout before gated norm.
    if ulysses_enabled:
        core_attn_out = gather_heads_scatter_seq(
            core_attn_out, head_dim=2, seq_dim=1, group=get_parallel_state().ulysses_group
        )

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


config.override_method(
    "Qwen3_5DecoderLayer.forward",
    replacement=qwen3_5_decoder_layer_forward_patched,
    name_map={"GPU": "NPU"},
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5DecoderLayer.forward",
)


config.override_method(
    "Qwen3_5TextModel._update_linear_attn_mask",
    replacement=qwen3_5_text_model_update_linear_attn_mask,
    description="Avoid host-device sync: decide linear-attention padding-mask zeroing without reading GPU scalars.",
)


config.override_method(
    "Qwen3_5Model.get_image_features",
    replacement=qwen3_5_model_get_image_features,
    description="Remove unnecessary split operation to maintain contiguous memory layout.",
)


config.override_method(
    "Qwen3_5Model.get_placeholder_mask",
    replacement=qwen3_5_model_get_placeholder_mask,
    description="Extract multimodal placeholder masks from input_ids using self-defined placeholder IDs.",
)


config.override_method(
    "Qwen3_5VisionModel.rot_pos_emb",
    replacement=qwen3_5_vision_model_rot_pos_emb,
    description="Accept pre-materialized grid_thw metadata to avoid redundant host sync in vision RoPE setup.",
)


config.override_method(
    "Qwen3_5VisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_5_vision_model_fast_pos_embed_interpolate,
    description="Optimized bilinear interpolation for high-resolution vision embeddings, adapted from vLLM.",
)


config.override_method(
    "Qwen3_5VisionModel.forward",
    replacement=qwen3_5_vision_model_forward,
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens.",
)


config.override_method(
    "Qwen3_5VisionModel.dummy_forward",
    replacement=qwen3_5_vision_model_dummy_forward,
    description="Add dummy_forward to prevent FSDP reduce-scatter hang on uneven multimodal batches.",
)


config.override_method(
    "Qwen3_5Model.forward",
    replacement=qwen3_5_model_forward,
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-NPU sync avoidance via pre-computed metadata."
    ),
)


config.override_method(
    "Qwen3_5ForCausalLM.forward",
    replacement=qwen3_5_forcausallm_forward_patched,
    description="Support fused cross entropy path in Qwen3_5ForCausalLM.forward",
)


config.override_method(
    "Qwen3_5ForConditionalGeneration.get_position_id_func",
    replacement=qwen3_5_forconditional_generation_get_position_id_func,
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)


config.override_method(
    "Qwen3_5ForConditionalGeneration.get_metadata_collate_func",
    replacement=qwen3_5_forconditional_generation_get_metadata_collate_func,
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)


config.override_method(
    "Qwen3_5ForConditionalGeneration.forward",
    replacement=qwen3_5_forconditional_generation_forward_patched,
    description="Support fused cross entropy path in Qwen3_5ForConditionalGeneration.forward",
)


# Mirrors the GPU config's helper-after; see qwen3_5_gpu_patch_gen_config.py
# for why @auto_docstring is intentionally skipped here.
@config.add_helper_after("Qwen3_5CausalLMOutputWithPast")
@dataclass
class Qwen3_5CausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen3_5CausalLMOutputWithPast):
    """``Qwen3_5CausalLMOutputWithPast`` + ``fused_linear_aux`` payload.

    fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
        Per-token tensors produced by the fused-linear loss path
        (``log_probs`` / ``entropy``; plus ``distillation_losses`` /
        ``student_mass`` / ``teacher_mass`` on the top-k distillation path).
        ``None`` on the plain loss path; populated when ``return_log_probs=True``.
    """
