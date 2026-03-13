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
Patch configuration for Qwen3_5 GPU/SP patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config -o veomni/models/transformers/qwen3_5/generated --diff

Language-model focused patches from qwen3_next example:
1. Device-agnostic GatedDeltaNet init and varlen FLA forward.
2. DecoderLayer forward with cu_seq_lens_q passthrough.
3. Use VeOmni fused loss path in Qwen3_5ForConditionalGeneration.forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Config,
    Qwen3_5DynamicCache,
    Qwen3_5RMSNormGated,
    apply_mask_to_padding_states,
    torch_chunk_gated_delta_rule,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from veomni.distributed.parallel_state import get_parallel_state
from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external
from veomni.utils.device import get_device_id


logger = logging.get_logger(__name__)


config = PatchConfig(
    source_module="transformers.models.qwen3_5.modeling_qwen3_5",
    target_file="patched_modeling_qwen3_5_gpu.py",
    description="Qwen3_5 with VeOmni language-model SP and fused loss patches",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)
config.patches.append(
    create_patch_from_external(
        target="Qwen3_5RMSNorm",
        replacement_module="liger_kernel.transformers.rms_norm",
        replacement_name="LigerRMSNormForQwen3Next",
        description="Use LigerKernel RMSNorm for Qwen3Next (1+weight centered formulation)",
    )
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
    try:
        from fla.modules import FusedRMSNormGated
        from fla.modules.convolution import causal_conv1d as causal_conv1d_fn
        from fla.modules.convolution import causal_conv1d_update
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except ImportError:
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
        FusedRMSNormGated = None
        causal_conv1d_update, causal_conv1d_fn = None, None
        logging.get_logger(__name__).warning(
            "Failed to import FLA modules: fallback to eager implementation. "
            "This case can't support rmpad_with_pos_ids=True!"
        )
    """
)


# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
FusedRMSNormGated = None
Qwen3_5GatedDeltaNet = None
causal_conv1d_fn = None
causal_conv1d_update = None
torch_causal_conv1d_update = None
chunk_gated_delta_rule = None
torch_chunk_gated_delta_rule = None  # noqa: F811 — also imported above for the forward patch
fused_recurrent_gated_delta_rule = None
torch_recurrent_gated_delta_rule = None
is_fast_path_available = None
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None


@config.override_method(
    "Qwen3_5GatedDeltaNet.__init__",
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)
def qwen3_5_gated_deltanet_init_patched(self, config: Qwen3_5Config, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.num_v_heads = config.linear_num_value_heads
    self.num_k_heads = config.linear_num_key_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads

    self.conv_kernel_size = config.linear_conv_kernel_dim
    self.layer_idx = layer_idx
    self.activation = config.hidden_act
    self.act = ACT2FN[config.hidden_act]
    self.layer_norm_epsilon = config.rms_norm_eps

    # QKV
    self.conv_dim = self.key_dim * 2 + self.value_dim
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=False,
        kernel_size=self.conv_kernel_size,
        groups=self.conv_dim,
        padding=self.conv_kernel_size - 1,
    )

    # time step projection (discretization)
    # instantiate once and copy inv_dt in init_weights of PretrainedModel
    self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

    A = torch.empty(self.num_v_heads).uniform_(0, 16)
    self.A_log = nn.Parameter(torch.log(A))

    self.norm = (
        Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        if FusedRMSNormGated is None
        else FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            # Modification: use device-agnostic get_device_id() instead of hardcoded device
            device=get_device_id(),
            dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
        )
    )

    self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    self.causal_conv1d_fn = causal_conv1d_fn
    self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
    self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
    self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

    if not is_fast_path_available:
        logger.warning_once(
            "The fast path is not available because one of the required library is not installed. Falling back to "
            "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
            " https://github.com/Dao-AILab/causal-conv1d"
        )

    self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
    self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
    self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
    self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)


@config.override_method(
    "Qwen3_5GatedDeltaNet._get_local_conv1d_weight",
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)
def qwen3_5_gated_deltanet_get_local_conv1d_weight(
    self, ulysses_rank: int, local_key_dim: int, local_value_dim: int
) -> torch.Tensor:
    # Modification: shard depthwise conv1d weights to match head-sharded mixed_qkv channels.
    w_full = self.conv1d.weight.squeeze(1)
    assert w_full.shape[0] == self.key_dim * 2 + self.value_dim, (
        f"conv1d weight dim ({w_full.shape[0]}) must match "
        f"(2 * key_dim + value_dim) ({self.key_dim * 2 + self.value_dim})"
    )
    k_off = ulysses_rank * local_key_dim
    v_off = ulysses_rank * local_value_dim
    w_q = w_full[k_off : k_off + local_key_dim]
    w_k = w_full[self.key_dim + k_off : self.key_dim + k_off + local_key_dim]
    w_v = w_full[2 * self.key_dim + v_off : 2 * self.key_dim + v_off + local_value_dim]
    return torch.cat([w_q, w_k, w_v], dim=0)


@config.override_method(
    "Qwen3_5GatedDeltaNet.forward",
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5GatedDeltaNet.forward",
)
def qwen3_5_gated_deltanet_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cache_params: Qwen3_5DynamicCache | None = None,
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
                cu_seqlens=cu_seq_lens_q,
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
        if self.chunk_gated_delta_rule is torch_chunk_gated_delta_rule:
            raise RuntimeError(
                "Varlen training requires FLA. Install flash-linear-attention so "
                "chunk_gated_delta_rule supports cu_seqlens."
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
                cu_seqlens=cu_seq_lens_q,
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


@config.override_method(
    "Qwen3_5DecoderLayer.forward",
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5DecoderLayer.forward",
)
def qwen3_5_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
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
        # Modification: pass cu_seq_lens_q through to Qwen3_5GatedDeltaNet.forward.
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
    hidden_states = residual + hidden_states
    return hidden_states


@config.override_method(
    "Qwen3_5ForConditionalGeneration.forward",
    description="Support fused cross entropy path in Qwen3_5ForConditionalGeneration.forward",
)
def qwen3_5_forconditional_generation_forward_patched(
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
) -> tuple | Qwen3_5CausalLMOutputWithPast:
    # Modification: VeOmni currently supports text-only Qwen3_5.
    # TODO(veomni): add vision input support for pixel_values/pixel_values_videos.
    if pixel_values is not None or pixel_values_videos is not None:
        raise ValueError(
            "Qwen3_5ForConditionalGeneration currently supports text-only inputs in VeOmni; "
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

    return Qwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
