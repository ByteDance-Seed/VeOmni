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
"""Patch configuration for the Qwen4-Exp GPU integration.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

The initial Ulysses path uses full-sequence QSA masks as a numerical reference;
it is correctness-oriented and intentionally fails closed for unsupported
cache topologies; CP2 is an explicit candidate. MTP is outside the training model and is
filtered by ``checkpoint_tensor_converter.py``.
"""

import math
from copy import copy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpTextModel,
    Qwen4ExpVisionModel,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    causal_conv1d_fn,
    causal_conv1d_update,
    load_balancing_loss_func,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, torch_compilable_check

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.models.transformers.qwen4_exp.packed_utils import compact_qsa_select, gather_qsa_selected_indices
from veomni.ops.kernels.attention.ulysses import prepare_ulysses_qkv, restore_ulysses_output
from veomni.ops.kernels.qwen4_exp import qsa_attn_npu_fused
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin
from veomni.utils.seqlen_pos_transform_utils import culen2pos, pos2culen


config = PatchConfig(
    source_module="transformers.models.qwen4_exp.modeling_qwen4_exp",
    target_file="patched_modeling_qwen4_exp_gpu.py",
    description="Qwen4-Exp initial GPU VLM-SFT integration with explicit PLE/QSA limits",
)

config.add_import("copy", names=["copy"])
config.add_import("veomni.utils.seqlen_pos_transform_utils", names=["culen2pos", "pos2culen"])
config.add_import("dataclasses", names=["dataclass"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.moe.comm", names=["all_to_all"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor"],
)
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_heads_scatter_seq", "gather_seq_scatter_heads"],
)
config.add_import(
    "veomni.ops.kernels.attention.ulysses",
    names=["prepare_ulysses_qkv", "restore_ulysses_output"],
)
config.add_import(
    "veomni.models.transformers.qwen4_exp.packed_utils",
    names=["compact_qsa_select", "gather_qsa_selected_indices"],
)
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.add_import("veomni.utils.model_outputs", names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"])
config.add_post_import_block(
    """
    # Bound by ``_bind_veomni_ops`` before model construction. Qwen4-Exp
    # runs eager QSA by default with optional CANN fused attention. GatedDeltaNet
    # binds the same kernels as Qwen3.5.
    from veomni.ops.dispatch import OpsConfigSlot, OpSlot
    from veomni.ops.kernels.qwen4_exp import qsa_attn_npu_fused
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    veomni_rms_norm_gated = OpSlot("rms_norm_gated", "standard")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")
    veomni_qsa_attention_implementation = OpsConfigSlot("qsa_attention_implementation")
    veomni_gdn_cp_implementation = OpsConfigSlot("qwen4_gdn_cp_implementation")
    """
)


# OpSlots are declared in the generated module's post-import block.
veomni_rms_norm_gated = None
veomni_causal_conv1d = None
veomni_chunk_gated_delta_rule = None
veomni_qsa_attention_implementation = None
veomni_gdn_cp_implementation = None


# ================================================================
# Patch: Qwen4ExpTextModel.reverse_embedding
# 1. Preserve the upstream recovery path while making exception chaining
#    explicit so generated code passes the repository's B904 lint gate.
# ================================================================
@config.override_method(
    "Qwen4ExpTextModel.reverse_embedding",
    description="Make the upstream reverse-embedding error path ruff-compliant",
)
def qwen4_exp_text_model_reverse_embedding_patched(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        input_ids = (
            (inputs_embeds[:, :, None, :] == self.embed_tokens.weight[None, None, :, :]).all(dim=3).nonzero()[:, 2]
        )
        try:
            input_ids = input_ids.view(inputs_embeds.shape[:2])
        except RuntimeError:
            # --- Patch.1 ---
            raise RuntimeError(
                "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and "
                "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Qwen4-Exp needs "
                "to reverse the embedding for PLE, provide exact embedding-table values or pass `ple_input_ids`."
            ) from None
            # --- Patch.1 ---
    return input_ids


# ================================================================
# Patch: Qwen4ExpModel.__init__
# 1. Build the generated local text/vision classes instead of AutoModel, so
#    VeOmni patches are retained inside the VLM wrapper.
# 2. Propagate the selected MoE backend into the nested text config.
# ================================================================
@config.override_method(
    "Qwen4ExpModel.__init__",
    description="Build local patched submodels and propagate the VeOmni MoE implementation",
)
def qwen4_exp_model_init_patched(self, config):
    # --- Patch.2 ---
    config.text_config._moe_implementation = getattr(config, "_moe_implementation", "eager")
    # --- Patch.2 ---

    super().__init__(config)
    # --- Patch.1 ---
    self.visual = Qwen4ExpVisionModel._from_config(config.vision_config)
    self.language_model = Qwen4ExpTextModel._from_config(config.text_config)
    # --- Patch.1 ---
    self.rope_deltas = None
    self.post_init()


@config.add_helper
def qwen4_exp_gdn_gather_sequence(tensor, seq_dim, head_dim, group):
    """Put sequence/head first to reuse the existing all_to_all_single path."""
    if (seq_dim, head_dim) != (1, 2):
        raise ValueError("GDN exchange expects [B,S,H,...].")
    output = gather_seq_scatter_heads(tensor.movedim(0, 2), seq_dim=0, head_dim=1, group=group)
    return output.movedim(2, 0).contiguous()


@config.add_helper
def qwen4_exp_gdn_scatter_sequence(tensor, seq_dim, head_dim, group):
    """Inverse headwise exchange; preserve the same autograd collective."""
    if (seq_dim, head_dim) != (1, 2):
        raise ValueError("GDN exchange expects [B,S,H,...].")
    output = gather_heads_scatter_seq(tensor.movedim(0, 2), seq_dim=0, head_dim=1, group=group)
    return output.movedim(2, 0).contiguous()


# ================================================================
# Patch: Qwen4ExpTextGatedDeltaNet
# 1. Freeze the configured GDN kernels on each model instance.
# 2. Exchange local sequence ownership for local head ownership under Ulysses.
# 3. Slice depthwise-convolution and recurrent parameters by local head range.
# 4. Restore local-sequence/full-head layout before the output gate.
# ================================================================
@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.__init__",
    description="Bind instance-local GDN kernels for Qwen4-Exp Ulysses",
)
def qwen4_exp_gated_deltanet_init_patched(self, config, layer_idx):
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
    self.layer_norm_epsilon = config.rms_norm_eps

    self.conv_dim = self.key_dim * 2 + self.value_dim
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=False,
        kernel_size=self.conv_kernel_size,
        groups=self.conv_dim,
        padding=self.conv_kernel_size - 1,
    )

    self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

    A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
    self.A_log = nn.Parameter(torch.log(A))
    self.norm = Qwen4ExpTextRMSNormGated(
        self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type or config.hidden_act
    )
    self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    self.layer_type = config.layer_types[layer_idx]

    self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
    self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
    self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
    self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    self.veomni_gdn_cp_implementation = veomni_gdn_cp_implementation.value
    if self.veomni_gdn_cp_implementation == "eager":
        self.veomni_gdn_cp_implementation = "headwise"
    if self.veomni_gdn_cp_implementation not in {"headwise"}:
        raise ValueError("Unknown Qwen4-Exp GDN CP implementation")
    self.veomni_causal_conv1d_fn = veomni_causal_conv1d.bound_kernel()
    self.veomni_chunk_gated_delta_rule = veomni_chunk_gated_delta_rule.bound_kernel()
    if veomni_rms_norm_gated.use_non_eager_impl:
        from veomni.utils.device import get_device_id

        self.norm = veomni_rms_norm_gated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=config.output_gate_type or config.hidden_act,
            device=get_device_id(),
            dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
        )


@config.add_helper
def _qwen4_exp_validate_packed_seq_lens(
    packed_seq_lens: tuple[int, ...] | list[int] | None,
    batch_size: int,
    sequence_length: int,
) -> tuple[int, ...]:
    """Validate the collator-provided segment lengths used by stateful token mixers."""
    if packed_seq_lens is None:
        raise ValueError("Qwen4-Exp VeOmni training requires packed sequence lengths.")
    if batch_size != 1:
        raise ValueError("Qwen4-Exp packed sequence metadata currently requires batch_size=1.")

    lengths = tuple(int(length) for length in packed_seq_lens)
    if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != sequence_length:
        raise ValueError(
            "Qwen4-Exp packed sequence lengths must be positive and cover the complete sequence; "
            f"got {lengths} for sequence_length={sequence_length}."
        )
    return lengths


@config.add_helper
def qwen4_exp_gdn_non_sp_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    linear_attn_cu_seq_lens_q: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    # --- Patch.3 ---
    if linear_attn_cu_seq_lens_q is None:
        raise ValueError("Qwen4-Exp GDN requires linear_attn_cu_seq_lens_q for VeOmni varlen training.")
    packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
        linear_attn_cu_seq_lens_q.diff().tolist(), batch_size, seq_len
    )
    if cache_params is not None:
        raise ValueError("Qwen4-Exp varlen GDN training does not support recurrent cache state.")
    # --- Patch.3 ---

    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if self.veomni_causal_conv1d_fn is not None:
        mixed_qkv = self.veomni_causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
            backend="triton",
            cu_seqlens=linear_attn_cu_seq_lens_q,
        )[0]
    else:
        mixed_qkv = torch.cat(
            [
                causal_conv1d_fn(
                    segment.transpose(1, 2),
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
                for segment in mixed_qkv.split(packed_seq_lens, dim=1)
            ],
            dim=1,
        )

    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim).contiguous()
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim).contiguous()
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if self.veomni_chunk_gated_delta_rule is not None:
        core_attn_out, _ = self.veomni_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=linear_attn_cu_seq_lens_q,
        )
    else:
        outputs = []
        for q_segment, k_segment, v_segment, g_segment, beta_segment in zip(
            query.split(packed_seq_lens, dim=1),
            key.split(packed_seq_lens, dim=1),
            value.split(packed_seq_lens, dim=1),
            g.split(packed_seq_lens, dim=1),
            beta.split(packed_seq_lens, dim=1),
            strict=True,
        ):
            segment_output, _ = torch_chunk_gated_delta_rule(
                q_segment,
                k_segment,
                v_segment,
                g=g_segment,
                beta=beta_segment,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            outputs.append(segment_output)
        core_attn_out = torch.cat(outputs, dim=1)

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


@config.add_helper
def qwen4_exp_gdn_pad_value_heads(states, num_key_heads):
    """Append a zero value slot within each three-value GDN head group."""
    grouped = states.unflatten(2, (num_key_heads, 3))
    zeros = grouped.new_zeros(*grouped.shape[:3], 1, *grouped.shape[4:])
    return torch.cat((grouped, zeros), dim=3).flatten(2, 3)


@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.forward",
    description="Run Qwen4-Exp GatedDeltaNet in full-sequence/local-head Ulysses layout",
)
def qwen4_exp_gated_deltanet_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    linear_attn_cu_seq_lens_q: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    parallel_state = get_parallel_state()
    if linear_attn_cu_seq_lens_q is None:
        linear_attn_cu_seq_lens_q = kwargs.pop("cu_seq_lens_q", None)
    if not parallel_state.sp_enabled:
        return qwen4_exp_gdn_non_sp_forward(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            linear_attn_cu_seq_lens_q=linear_attn_cu_seq_lens_q,
            **kwargs,
        )
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    if linear_attn_cu_seq_lens_q is None:
        raise ValueError("Qwen4-Exp GDN requires global packed boundaries under SP.")
    _qwen4_exp_validate_packed_seq_lens(
        linear_attn_cu_seq_lens_q.diff().tolist(), batch_size, seq_len * parallel_state.sp_size
    )
    ulysses_enabled = parallel_state.sp_enabled

    if ulysses_enabled and cache_params is not None:
        raise NotImplementedError("Qwen4-Exp GatedDeltaNet does not support KV/recurrent cache state under Ulysses.")

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if ulysses_enabled:
        ulysses_size = parallel_state.sp_size
        padded_gdn_heads = ulysses_size == 2 * self.num_k_heads and self.num_v_heads == 3 * self.num_k_heads
        if not padded_gdn_heads and (self.num_k_heads % ulysses_size != 0 or self.num_v_heads % ulysses_size != 0):
            raise ValueError(
                f"ulysses_size ({ulysses_size}) must divide Qwen4-Exp GatedDeltaNet key heads "
                f"({self.num_k_heads}) and value heads ({self.num_v_heads})."
            )
        if self.veomni_causal_conv1d_fn is None or self.veomni_chunk_gated_delta_rule is None:
            raise RuntimeError(
                "Qwen4-Exp GatedDeltaNet Ulysses requires non-eager causal_conv1d and "
                "chunk_gated_delta_rule implementations."
            )

        ulysses_group = parallel_state.sp_group
        ulysses_rank = parallel_state.sp_rank
        local_num_k_heads = 1 if padded_gdn_heads else self.num_k_heads // ulysses_size
        local_num_v_heads = 2 if padded_gdn_heads else self.num_v_heads // ulysses_size
        local_key_dim = local_num_k_heads * self.head_k_dim
        local_value_dim = local_num_v_heads * self.head_v_dim

        q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        if padded_gdn_heads:
            # Split each real 1K/3V group over two ranks as 1K/2V and 1K/(1V+zero).
            # Autograd sums the two Q/K uses back into the original parameters.
            q_proj = q_proj.repeat_interleave(2, dim=2)
            k_proj = k_proj.repeat_interleave(2, dim=2)
            v_proj = qwen4_exp_gdn_pad_value_heads(v_proj, self.num_k_heads)
            b = qwen4_exp_gdn_pad_value_heads(b, self.num_k_heads)
            a = qwen4_exp_gdn_pad_value_heads(a, self.num_k_heads)
        q_proj = qwen4_exp_gdn_gather_sequence(q_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        k_proj = qwen4_exp_gdn_gather_sequence(k_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        v_proj = qwen4_exp_gdn_gather_sequence(v_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        b = qwen4_exp_gdn_gather_sequence(b, seq_dim=1, head_dim=2, group=ulysses_group)
        a = qwen4_exp_gdn_gather_sequence(a, seq_dim=1, head_dim=2, group=ulysses_group)
        mixed_qkv = torch.cat(
            (
                q_proj.flatten(2),
                k_proj.flatten(2),
                v_proj.flatten(2),
            ),
            dim=-1,
        )

        full_weight = self.conv1d.weight.squeeze(1)
        key_offset = (ulysses_rank // 2 if padded_gdn_heads else ulysses_rank) * local_key_dim
        value_head_start = (
            3 * (ulysses_rank // 2) + 2 * (ulysses_rank % 2) if padded_gdn_heads else ulysses_rank * local_num_v_heads
        )
        valid_value_heads = 2 - ulysses_rank % 2 if padded_gdn_heads else local_num_v_heads
        value_offset = value_head_start * self.head_v_dim
        value_weight = full_weight[
            2 * self.key_dim + value_offset : 2 * self.key_dim + value_offset + valid_value_heads * self.head_v_dim
        ]
        value_weight = F.pad(value_weight, (0, 0, 0, (local_num_v_heads - valid_value_heads) * self.head_v_dim))
        conv_weight = torch.cat(
            (
                full_weight[key_offset : key_offset + local_key_dim],
                full_weight[self.key_dim + key_offset : self.key_dim + key_offset + local_key_dim],
                value_weight,
            ),
            dim=0,
        )
        mixed_qkv = self.veomni_causal_conv1d_fn(
            x=mixed_qkv,
            weight=conv_weight,
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
            backend="triton",
            cu_seqlens=linear_attn_cu_seq_lens_q,
        )[0]
    else:
        local_num_k_heads = self.num_k_heads
        local_num_v_heads = self.num_v_heads
        local_key_dim = self.key_dim
        local_value_dim = self.value_dim
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
    # FlashQLA requires the sequence stride to match each unpacked tensor's
    # own head width. ``torch.split`` otherwise leaves Q/K/V as views whose
    # sequence stride is the complete packed-QKV width.
    query = query.reshape(batch_size, -1, local_num_k_heads, self.head_k_dim).contiguous()
    key = key.reshape(batch_size, -1, local_num_k_heads, self.head_k_dim).contiguous()
    value = value.reshape(batch_size, -1, local_num_v_heads, self.head_v_dim).contiguous()
    beta = b.sigmoid()

    if ulysses_enabled:
        value_head_slice = slice(value_head_start, value_head_start + valid_value_heads)
        pad_heads = local_num_v_heads - valid_value_heads
        local_a_log = F.pad(self.A_log[value_head_slice], (0, pad_heads))
        local_dt_bias = F.pad(self.dt_bias[value_head_slice], (0, pad_heads))
        g = -local_a_log.float().exp() * F.softplus(a.float() + local_dt_bias)
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if local_num_v_heads // local_num_k_heads > 1:
        query = query.repeat_interleave(local_num_v_heads // local_num_k_heads, dim=2)
        key = key.repeat_interleave(local_num_v_heads // local_num_k_heads, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
    if ulysses_enabled:
        core_attn_out, last_recurrent_state = self.veomni_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=linear_attn_cu_seq_lens_q,
        )
    elif use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            **kwargs,
        )
    else:
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            **kwargs,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    if ulysses_enabled:
        core_attn_out = qwen4_exp_gdn_scatter_sequence(
            core_attn_out,
            head_dim=2,
            seq_dim=1,
            group=parallel_state.sp_group,
        )

        if padded_gdn_heads:
            core_attn_out = core_attn_out.unflatten(2, (self.num_k_heads, 4))[:, :, :, :3].flatten(2, 3).contiguous()

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


# ================================================================
# Patch: Qwen4ExpTextQSAIndexer.forward
# Use local-query/global-block compact selection. The eager attention backend
# expands the returned indices to a dense selected-token mask after exchange.
# ================================================================
@config.override_method(
    "Qwen4ExpTextQSAIndexer.forward",
    description="Select compact global QSA token indices under Ulysses",
)
def qwen4_exp_qsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    del attention_mask
    parallel_state = get_parallel_state()
    if past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp compact QSA does not support a KV/indexer cache.")
    selected_local = compact_qsa_select(
        self,
        hidden_states,
        position_embeddings,
        cu_seq_lens_q,
        group=parallel_state.sp_group,
        rank=parallel_state.sp_rank if parallel_state.sp_enabled else 0,
        world_size=parallel_state.sp_size,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    return gather_qsa_selected_indices(
        selected_local,
        group=parallel_state.ulysses_group,
        world_size=parallel_state.ulysses_size,
    )


@config.add_helper
def qwen4_exp_gather_cp_kv(states: torch.Tensor, parallel_state) -> torch.Tensor:
    """Restore canonical tokens after U head exchange and CP KV all-gather.

    Input is [B,H,U*L,D]. CP gathering produces [CP,U,L] token blocks;
    the collator and flattened SP mesh use [U,CP,L]. Queries stay CP-local.
    gather_outputs sums KV gradients from the disjoint CP query partitions.
    """
    cp_size = parallel_state.cp_size
    ulysses_size = parallel_state.ulysses_size
    if states.shape[2] % ulysses_size:
        raise ValueError("Qwen4-Exp CP KV length must divide into equal Ulysses shards.")
    batch, heads, length, dim = states.shape
    gathered = gather_outputs(states, gather_dim=2, group=parallel_state.cp_group)
    return (
        gathered.reshape(batch, heads, cp_size, ulysses_size, length // ulysses_size, dim)
        .transpose(2, 3)
        .reshape(batch, heads, cp_size * length, dim)
    )


# ================================================================
# Patch: Qwen4ExpTextAttention.forward
# 1. Keep compact selections in global token coordinates.
# 2. Exchange main Q/K/V into full-sequence/local-head layout.
# 3. Expand compact indices to a dense mask in the eager QSA reference.
# ================================================================
@config.override_method(
    "Qwen4ExpTextAttention.forward",
    description="Run dense-mask eager QSA with global selection and Ulysses QKV exchange",
)
def qwen4_exp_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    parallel_state = get_parallel_state()
    if parallel_state.cp_enabled and attention_mask is not None:
        raise NotImplementedError(
            "Qwen4-Exp CP2 uses compact QSA selections; explicit attention masks are unsupported."
        )
    if self.training and self.attention_dropout != 0:
        raise ValueError("Qwen4-Exp compact QSA currently requires attention_dropout=0 during training.")
    if parallel_state.sp_enabled:
        if past_key_values is not None:
            raise NotImplementedError("Qwen4-Exp QSA does not support a KV cache under Ulysses.")
        ulysses_size = parallel_state.ulysses_size
        query_head_count = self.q_proj.out_features // (2 * self.head_dim)
        key_value_head_count = self.k_proj.out_features // self.head_dim
        if query_head_count % ulysses_size != 0:
            raise ValueError(
                f"Qwen4-Exp QSA query heads ({query_head_count}) must be divisible by ulysses_size ({ulysses_size})."
            )
        if key_value_head_count % ulysses_size != 0 and ulysses_size % key_value_head_count != 0:
            raise ValueError(
                f"Qwen4-Exp QSA KV heads ({key_value_head_count}) and ulysses_size ({ulysses_size}) "
                "must divide one another."
            )
        local_seq_len = hidden_states.shape[1]
        global_seq_len = position_embeddings[0].shape[1]
        if global_seq_len != local_seq_len * parallel_state.sp_size:
            raise ValueError(
                "Qwen4-Exp QSA position embeddings must cover the complete Ulysses sequence; "
                f"got global={global_seq_len}, local={local_seq_len}, ulysses_size={ulysses_size}."
            )

    selection = self.indexer(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
    )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    if parallel_state.sp_enabled:
        position_start = parallel_state.sp_rank * local_seq_len
        local_position_embeddings = tuple(
            tensor[:, position_start : position_start + local_seq_len, :] for tensor in position_embeddings
        )
    else:
        local_position_embeddings = tuple(tensor[:, -hidden_states.shape[1] :, :] for tensor in position_embeddings)

    cos, sin = local_position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            self.layer_idx,
        )
        query_states = query_states.transpose(1, 2)
    elif parallel_state.ulysses_enabled:
        query_states, key_states, value_states, _ = prepare_ulysses_qkv(
            query_states,
            key_states,
            value_states,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    if parallel_state.cp_enabled:
        key_states = qwen4_exp_gather_cp_kv(key_states, parallel_state)
        value_states = qwen4_exp_gather_cp_kv(value_states, parallel_state)
    # Non-streamed implementations consume global KV and compact indices.
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        selected_indices=selection,
    )

    if parallel_state.ulysses_enabled:
        attn_output = restore_ulysses_output(attn_output, group=parallel_state.ulysses_group)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output * torch.sigmoid(gate))
    return attn_output, attn_weights


# ================================================================
# Patch: eager_attention_forward
# 1. Expand compact QSA selections into a dense mask for the reference path.
# 2. Preserve the Transformers eager contract for every non-QSA caller.
# ================================================================
@config.replace_function("eager_attention_forward", description="Optional dense QSA dispatch")
def qwen4_exp_eager_attention_forward_patched(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run standard eager attention, optionally constrained by compact QSA indices.

    Q/K/V use ``[B, H, S, D]`` and indices use ``[B, S, K]``. Invalid slots
    are ``-1``. Query heads may be a multiple of KV heads (GQA/MQA).

    The compact-selection implementation
    uses SDPA with a dense selection mask. A fused device backend can
    avoid the full score tensor, but this is not a sparse compute backend.
    """

    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Match Transformers' GQA expansion without ``repeat_interleave``."""
        if n_rep == 1:
            return hidden_states
        batch_size, kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None].expand(batch_size, kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch_size, kv_heads * n_rep, seq_len, head_dim)

    selected_indices = kwargs.pop("selected_indices", None)
    if selected_indices is None:
        # --- Patch.2 ---
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attention_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_weights = F.dropout(attention_weights, p=dropout, training=module.training)
        attention_output = torch.matmul(attention_weights, value_states)
        # --- Patch.2 ---
        return attention_output.transpose(1, 2).contiguous(), attention_weights

    # --- Patch.1 ---
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or selected_indices.ndim != 3:
        raise ValueError("QSA expects query/key/value [B,H,S,D] and selected_indices [B,S,K].")
    if dropout != 0:
        raise ValueError("QSA eager attention currently requires dropout=0.")
    batch_size, query_heads, query_len, head_dim = query.shape
    if key.shape[:1] != (batch_size,) or value.shape[:1] != (batch_size,):
        raise ValueError("QSA query, key, and value batch sizes must match.")
    if key.shape != value.shape:
        raise ValueError(f"QSA key/value shapes must match; got key={key.shape}, value={value.shape}.")
    if selected_indices.shape[:2] != (batch_size, query_len):
        raise ValueError(
            "QSA selected indices must match the query batch and sequence dimensions; "
            f"got indices={selected_indices.shape}, query={query.shape}."
        )
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"QSA selected indices must be int32 or int64; got {selected_indices.dtype}.")
    kv_heads, kv_len = key.shape[1:3]
    if kv_len == 0:
        raise ValueError("QSA requires at least one KV token.")
    if query_heads % kv_heads != 0:
        raise ValueError(f"QSA query heads ({query_heads}) must be divisible by KV heads ({kv_heads}).")
    if key.shape[-1] != head_dim:
        raise ValueError(f"QSA query/key head dimensions must match; got {head_dim} and {key.shape[-1]}.")
    invalid_indices = (selected_indices < -1) | (selected_indices >= kv_len)
    if bool(invalid_indices.any()):
        raise ValueError("QSA selected indices must be -1 or valid global KV token indices.")

    qsa_implementation = veomni_qsa_attention_implementation.value
    if qsa_implementation not in {"eager", "npu_fused"}:
        raise ValueError(
            f"Unknown qsa_attention_implementation={qsa_implementation!r}; expected 'eager' or 'npu_fused'."
        )
    if qsa_implementation == "npu_fused":
        if attention_mask is not None:
            raise ValueError("QSA npu_fused requires masking encoded in selected_indices")
        chunk_size = kwargs.pop("query_chunk_size", 2048)
        return qsa_attn_npu_fused(query, key, value, selected_indices, scaling, chunk_size), None

    from torch.utils.checkpoint import checkpoint

    query_chunk_size = kwargs.pop("query_chunk_size", 2048 if query.device.type == "npu" else 1024)
    if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
        raise ValueError("QSA query_chunk_size must be a positive integer.")
    if query_len == 0:
        return query.transpose(1, 2).contiguous(), None
    repeats = query_heads // kv_heads
    key_states = _repeat_kv(key, repeats)
    value_states = _repeat_kv(value, repeats)

    def attend_chunk(q, k, v, indices, bias):
        # Only this query block owns a dense mask. The sentinel column absorbs
        # invalid slots without overwriting valid duplicate selections.
        selected = torch.zeros((*indices.shape[:-1], kv_len + 1), dtype=torch.bool, device=indices.device)
        scatter_indices = torch.where(indices >= 0, indices, kv_len)
        allowed = selected.scatter_(-1, scatter_indices.long(), True)[..., :kv_len][:, None]
        sdpa_mask = allowed
        if bias is not None:
            if bias.is_floating_point():
                sdpa_mask = bias.masked_fill(~allowed, float("-inf"))
            else:
                sdpa_mask = allowed & bias
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False, scale=scaling
        )

    outputs = []
    for start in range(0, query_len, query_chunk_size):
        stop = min(start + query_chunk_size, query_len)
        bias = attention_mask
        if bias is not None and bias.ndim >= 2 and bias.shape[-2] != 1:
            bias = bias[..., start:stop, :]
        args = (query[:, :, start:stop], key_states, value_states, selected_indices[:, start:stop], bias)
        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in args):
            # Recompute masks inside backward; saving every block's mask would
            # accumulate quadratic memory. No SP collective is inside this scope.
            output = checkpoint(attend_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            output = attend_chunk(*args)
        outputs.append(output)
    attention_output = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
    # --- Patch.1 ---
    return attention_output, None


# ================================================================
# Patch: Qwen4ExpTextExperts
# 1. Drop HF's use_experts_implementation decorator so VeOmni owns dispatch.
# 2. Retain the upstream fused checkpoint layout and eager implementation.
# ================================================================
@config.replace_class(
    "Qwen4ExpTextExperts",
    description="Use the VeOmni MoE OpSlot while preserving Qwen4-Exp fused expert weights",
)
class PatchedQwen4ExpTextExperts(nn.Module):
    """Qwen4-Exp expert tensors with optional VeOmni fused dispatch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # --- Patch.1 ---
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        # --- Patch.1 ---

        # --- Patch.2 ---
        final_hidden_states = torch.zeros_like(hidden_states)
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
        return final_hidden_states
        # --- Patch.2 ---


# ================================================================
# Patch: Qwen4ExpTextNGramEmbedding
# 1. Preserve the checkpoint-native 128-table layout instead of concatenating
#    the complete ~95 GiB PLE parameter.
# 2. Pad each independent table on dim 0 so it can be row-sharded by the
#    generic ExtraParallel/FSDP2 streaming loader.
# 3. Route lookup requests to the owning PLE rank with autograd-aware all-to-all
#    so ranks may train on different data-parallel samples.
# 4. Keep each table persistently sharded over PLE rows and complementary
#    PLE-FSDP columns; route requests over the flattened 2D mesh instead of
#    all-gathering parameters.
# 5. Cast lookup results to the requested compute dtype before communicating
#    them, while retaining FP32 master parameters.
# ================================================================
@config.add_helper
class _Qwen4ExpScaleGradient(torch.autograd.Function):
    """Leave lookup values unchanged and average their backward contribution."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, divisor: float) -> torch.Tensor:
        ctx.divisor = divisor
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output / ctx.divisor, None


@config.replace_class(
    "Qwen4ExpTextNGramEmbedding",
    description="Use checkpoint-native row-sharded PLE tables with distributed lookup",
)
class PatchedQwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(self, config, embedding_dim: int, layer_idx: int, ple_layer_index: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.layer_multipliers = nn.Buffer(
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed)
        )
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))

        # --- Patch.1 / Patch.2 ---
        vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / vocab_divisor) * vocab_divisor
        if padded_vocab_size % config.split_ngram_parts != 0:
            raise ValueError(
                "Qwen4-Exp PLE padded vocabulary must divide evenly across split_ngram_parts; "
                f"got padded_vocab_size={padded_vocab_size}, split_ngram_parts={config.split_ngram_parts}."
            )
        self.split_ngram_parts = config.split_ngram_parts
        self.rows_per_checkpoint_shard = padded_vocab_size // self.split_ngram_parts
        self.padded_rows_per_shard = math.ceil(self.rows_per_checkpoint_shard / vocab_divisor) * vocab_divisor
        self.ngram_embedding = nn.ModuleDict(
            {
                f"shard_{shard_idx}": nn.Embedding(
                    self.padded_rows_per_shard,
                    head_dim_per_ngram,
                    dtype=torch.float32,
                )
                for shard_idx in range(self.split_ngram_parts)
            }
        )
        # --- Patch.1 / Patch.2 ---

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def _lookup_local_rows(
        self,
        shard_ids: torch.Tensor,
        row_ids: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        first_embedding = self.ngram_embedding["shard_0"]
        first_weight = first_embedding.weight
        if hasattr(first_weight, "to_local"):
            first_weight = first_weight.to_local()
        output = first_weight.new_zeros(
            (shard_ids.numel(), first_weight.shape[1]),
            dtype=output_dtype or first_weight.dtype,
        )
        for shard_idx, embedding in enumerate(self.ngram_embedding.values()):
            positions = torch.where(shard_ids == shard_idx)[0]
            weight = embedding.weight
            if hasattr(weight, "to_local"):
                weight = weight.to_local()
            values = nn.functional.embedding(row_ids[positions], weight).to(output.dtype)
            output = output.index_copy(0, positions, values)
        return output

    def _distributed_lookup(
        self,
        shard_ids: torch.Tensor,
        row_ids: torch.Tensor,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parallel_state = get_parallel_state()
        ple_size = parallel_state.extra_parallel_sizes.get("ple", 1)
        if ple_size == 1:
            return self._lookup_local_rows(shard_ids, row_ids, output_dtype=output_dtype)

        # --- Patch.3 ---
        first_embedding = self.ngram_embedding["shard_0"]
        first_weight = first_embedding.weight
        persistent_2d = hasattr(first_weight, "placements") and len(first_weight.placements) == 2
        if persistent_2d:
            ple_mesh = parallel_state.extra_parallel_fsdp_device_mesh["ple"]
            ple_fsdp_size = ple_mesh.size(0)
            group = parallel_state.extra_parallel_flat_group("ple")
            group_size = ple_size * ple_fsdp_size
        else:
            # Compatibility path for an old row-only plan whose local row
            # partition is still managed by FSDP2.
            ple_fsdp_size = 1
            group = parallel_state.extra_parallel_group("ple")
            group_size = ple_size

        if not dist.is_initialized() or dist.get_world_size(group) != group_size:
            raise RuntimeError("Qwen4-Exp PLE parallel lookup requires an initialized 'ple' process group.")
        local_rows = self.padded_rows_per_shard // ple_size
        local_weight = first_weight.to_local() if persistent_2d else first_weight
        expected_local_cols = first_embedding.embedding_dim // ple_fsdp_size
        if tuple(local_weight.shape) != (local_rows, expected_local_cols):
            raise RuntimeError(
                "Qwen4-Exp PLE parameters do not have the expected local row/column shard; "
                f"got {tuple(local_weight.shape)}, expected {(local_rows, expected_local_cols)}."
            )

        owners = torch.div(row_ids, local_rows, rounding_mode="floor")
        local_row_ids = row_ids - owners * local_rows
        if persistent_2d:
            # Each logical request needs one slice from every column owner.
            # Repetition order is [request0-col0..F, request1-col0..F, ...],
            # which lets the inverse permutation reconstruct [K, F, E/F].
            col_owners = torch.arange(ple_fsdp_size, device=row_ids.device).repeat(row_ids.numel())
            routed_owners = owners.repeat_interleave(ple_fsdp_size)
            routed_shard_ids = shard_ids.repeat_interleave(ple_fsdp_size)
            routed_local_row_ids = local_row_ids.repeat_interleave(ple_fsdp_size)
            rank_table = row_ids.new_tensor(parallel_state.extra_parallel_2d_rank_table("ple"))
            destinations = rank_table[col_owners, routed_owners]
        else:
            destinations = owners
            routed_shard_ids = shard_ids
            routed_local_row_ids = local_row_ids

        order = torch.argsort(destinations)
        send_counts_tensor = torch.bincount(destinations, minlength=group_size).to(dtype=torch.int64)
        recv_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=group)
        send_counts = send_counts_tensor.tolist()
        recv_counts = recv_counts_tensor.tolist()

        requests = torch.stack((routed_shard_ids[order], routed_local_row_ids[order]), dim=-1)
        received_requests = requests.new_empty((sum(recv_counts), 2))
        dist.all_to_all_single(
            received_requests,
            requests,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        # --- Patch.5 ---
        local_output = self._lookup_local_rows(
            received_requests[:, 0],
            received_requests[:, 1],
            output_dtype=output_dtype,
        )
        # --- Patch.5 ---
        returned_output = all_to_all(group, local_output, send_counts, recv_counts)

        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        returned_output = returned_output[inverse_order]
        if persistent_2d:
            # FSDP2 ignores PLE weights, so its reduce-scatter no longer
            # averages their gradients. Every source rank's contribution is
            # routed to the unique 2D owner; average those contributions once
            # in this lookup's backward path.
            returned_output = _Qwen4ExpScaleGradient.apply(
                returned_output, float(parallel_state.extra_parallel_gradient_divide_factor("ple"))
            )
            returned_output = returned_output.view(shard_ids.numel(), ple_fsdp_size, expected_local_cols).flatten(1)
        return returned_output
        # --- Patch.3 ---

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        output_dtype: torch.dtype | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.long()
        parallel_state = get_parallel_state()
        if cu_seq_lens_q is None:
            raise ValueError("Qwen4-Exp PLE requires global packed boundaries.")
        _qwen4_exp_validate_packed_seq_lens(
            cu_seq_lens_q.diff().tolist(), input_ids.shape[0], input_ids.shape[1] * parallel_state.sp_size
        )
        if past_key_values is not None:
            raise ValueError("Qwen4-Exp varlen PLE does not support cache state.")
        if parallel_state.sp_enabled and past_key_values is not None:
            raise NotImplementedError("Qwen4-Exp PLE n-gram history does not support cache state under Ulysses.")
        if parallel_state.sp_enabled and self.context_len > 0:
            if input_ids.shape[1] < self.context_len:
                raise ValueError(
                    f"The local Ulysses sequence length ({input_ids.shape[1]}) must be at least the PLE "
                    f"n-gram halo ({self.context_len})."
                )
            local_tail = input_ids[:, -self.context_len :].contiguous()
            gathered_tails = [torch.empty_like(local_tail) for _ in range(parallel_state.sp_size)]
            dist.all_gather(gathered_tails, local_tail, group=parallel_state.sp_group)
            previous_context = (
                input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
                if parallel_state.sp_rank == 0
                else gathered_tails[parallel_state.sp_rank - 1]
            )
        elif past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            if (
                not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                and input_ids.shape[1] < self.context_len
            ):
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]
        local_start = parallel_state.sp_rank * input_ids.shape[1] if parallel_state.sp_enabled else 0
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + local_start
        segment_ids = torch.bucketize(positions, cu_seq_lens_q[1:].to(positions.device), right=True)
        segment_starts = cu_seq_lens_q.to(positions.device)[segment_ids]
        shifted_tokens = [
            torch.where(
                (positions - shift >= segment_starts).unsqueeze(0),
                tokens[:, -input_ids.shape[1] :],
                self.eos_token_id,
            )
            for shift, tokens in enumerate(shifted_tokens)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        original_shape = ngram_ids.shape
        flat_ids = ngram_ids.reshape(-1)
        shard_ids = torch.div(flat_ids, self.rows_per_checkpoint_shard, rounding_mode="floor")
        row_ids = torch.remainder(flat_ids, self.rows_per_checkpoint_shard)
        # --- Patch.5 ---
        embeddings = self._distributed_lookup(shard_ids, row_ids, output_dtype=output_dtype)
        # --- Patch.5 ---
        return embeddings.view(*original_shape, -1).flatten(-2)


# ================================================================
# Patch: Qwen4ExpTextPLELayer._short_conv
# 1. Add a differentiable left halo for the dilated depthwise convolution.
# 2. Keep the original cache/padding path unchanged when Ulysses is disabled.
# ================================================================
@config.override_method(
    "Qwen4ExpTextPLELayer._short_conv",
    description="Exchange PLE halos and reset convolution at packed boundaries",
)
def qwen4_exp_text_ple_layer_short_conv_patched(
    self,
    hidden_states: torch.Tensor,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    parallel_state = get_parallel_state()
    if cu_seq_lens_q is None:
        raise ValueError("Qwen4-Exp PLE convolution requires global packed boundaries.")
    local_length = hidden_states.shape[1]
    _qwen4_exp_validate_packed_seq_lens(
        cu_seq_lens_q.diff().tolist(), hidden_states.shape[0], local_length * parallel_state.sp_size
    )
    if past_key_values is not None:
        raise ValueError("Qwen4-Exp varlen PLE convolution does not support cache state.")
    halo_length = self.short_conv_state_len
    if parallel_state.sp_enabled and local_length < halo_length:
        raise ValueError("Local SP sequence must be at least the PLE convolution halo length.")
    if parallel_state.sp_enabled and halo_length:
        tails = gather_outputs(
            hidden_states[:, -halo_length:].contiguous(), gather_dim=1, group=parallel_state.sp_group
        )
        start = max(parallel_state.sp_rank - 1, 0) * halo_length
        halo = tails[:, start : start + halo_length]
        if parallel_state.sp_rank == 0:
            halo = halo * 0
    else:
        halo = hidden_states.new_zeros(hidden_states.shape[0], halo_length, hidden_states.shape[2])
    history = torch.cat((halo, hidden_states), dim=1)
    local_start = parallel_state.sp_rank * local_length if parallel_state.sp_enabled else 0
    outputs = []
    boundaries = cu_seq_lens_q.tolist()
    for segment_start, segment_end in zip(boundaries, boundaries[1:]):
        query_start = max(segment_start, local_start)
        query_end = min(segment_end, local_start + local_length)
        if query_end <= query_start:
            continue
        history_start = max(segment_start, query_start - halo_length)
        offset = history_start - local_start + halo_length
        segment = history[:, offset : query_end - local_start + halo_length].transpose(1, 2)
        segment = F.pad(segment, (halo_length - (query_start - history_start), 0))
        outputs.append(F.silu(self.conv1d(segment)).transpose(1, 2))
    return torch.cat(outputs, dim=1)


# ================================================================
# Patch: Qwen4ExpTextPLELayer.forward
# 1. Keep FP32 PLE master weights while casting sparse lookup results to the
#    activation dtype before the result all-to-all and downstream projections.
# ================================================================
@config.override_method(
    "Qwen4ExpTextPLELayer.forward",
    description="Use mixed-precision PLE lookup values and reset state at packed boundaries",
)
def qwen4_exp_text_ple_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    past_key_values: Cache | None,
    conv_mask: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    # --- Patch.1 ---
    embeddings = self.ple_embedding(
        input_ids,
        past_key_values,
        output_dtype=hidden_states.dtype,
        cu_seq_lens_q=cu_seq_lens_q,
    )
    # --- Patch.1 ---
    key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
    value = self.value_proj(embeddings)
    query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = self.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    if conv_mask is not None:
        gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
        gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
    output = gated_value + self._short_conv(
        gated_value_normed,
        past_key_values,
        cu_seq_lens_q=cu_seq_lens_q,
    )
    return output


@config.override_method(
    "Qwen4ExpTextDecoderLayer.forward",
    description="Pass packed sequence boundaries to Qwen4-Exp GDN and PLE",
)
def qwen4_exp_text_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    conv_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    ple_input_ids: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.FloatTensor:
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(
            hidden_states,
            ple_input_ids,
            past_key_values,
            conv_mask=conv_mask,
            cu_seq_lens_q=cu_seq_lens_q,
        )

    hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        linear_attn_cu_seq_lens_q = kwargs.pop("linear_attn_cu_seq_lens_q", cu_seq_lens_q)
        hidden_states = self.linear_attn(
            hidden_states,
            cache_params=past_key_values,
            attention_mask=conv_mask,
            linear_attn_cu_seq_lens_q=linear_attn_cu_seq_lens_q,
            **kwargs,
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)

    hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
    hidden_states = self.mlp(hidden_states)
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


# ================================================================
# Patch: Qwen4ExpTextModel.forward
# 1. Defer the quadratic QSA mask to the attention backend while retaining
#    full-sequence M-RoPE embeddings under Ulysses.
# 2. Keep hidden states, PLE ids, and recurrent padding masks sequence-local.
# 3. Reject cache and context-parallel combinations before collectives.
# ================================================================
@config.override_method(
    "Qwen4ExpTextModel.forward",
    description="Coordinate full-sequence QSA metadata with local GDN/PLE tensors under Ulysses",
)
def qwen4_exp_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    ple_input_ids: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    r"""
    ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Original token ids used by Per-Layer Embedding (PLE). This is only needed when PLE is enabled and
        `inputs_embeds` are passed instead of `input_ids`.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    parallel_state = get_parallel_state()
    if self.training and self.config.attention_dropout != 0:
        raise ValueError("Qwen4-Exp compact QSA currently requires attention_dropout=0 during training.")
    if parallel_state.cp_enabled and (parallel_state.cp_size not in (2, 4) or not parallel_state.ulysses_enabled):
        raise NotImplementedError("Qwen4-Exp CP candidate requires cp_size in (2, 4) and ulysses_size>1.")
    if parallel_state.sp_enabled and (use_cache or past_key_values is not None):
        raise NotImplementedError("Qwen4-Exp does not support cache prefill or decode under Ulysses.")
    if self.config.ple_layer_ids and ple_input_ids is None:
        ple_input_ids = input_ids if input_ids is not None else self.reverse_embedding(inputs_embeds)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        if parallel_state.sp_enabled:
            raise ValueError("Qwen4-Exp Ulysses requires collator-provided globalizable position_ids.")
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if parallel_state.sp_enabled:
        position_ids = gather_outputs(
            position_ids,
            gather_dim=-1,
            group=parallel_state.sp_group,
        )

    if position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    elif position_ids.shape[0] == 1:
        text_position_ids = position_ids[0]
        position_ids = position_ids.expand(3, -1, -1)
    else:
        text_position_ids = None

    if past_key_values is not None:
        if hasattr(past_key_values, "position_ids"):
            position_ids = torch.cat((past_key_values.position_ids, position_ids), dim=-1)
        past_key_values.position_ids = position_ids

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        if parallel_state.sp_enabled:
            mask_seq_len = inputs_embeds.shape[1] * parallel_state.sp_size
            mask_inputs = inputs_embeds.new_empty((inputs_embeds.shape[0], mask_seq_len, 1))
        else:
            mask_inputs = inputs_embeds
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": mask_inputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
            "allow_is_causal_skip": False,
        }
        causal_mask_mapping = {
            "full_attention": None,
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }

    full_attention_mask = None
    conv_mask = causal_mask_mapping.get("linear_attention")
    if parallel_state.sp_enabled:
        if conv_mask is not None:
            conv_mask = slice_input_tensor(
                conv_mask,
                dim=-1,
                padding=False,
                group=parallel_state.sp_group,
            )

    if self.config.ple_layer_ids and conv_mask is not None:
        eos_token_id = self.config.eos_token_id
        eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos_token_id)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=full_attention_mask,
            conv_mask=conv_mask,
            past_key_values=past_key_values,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

    hidden_states = self.hyper_connection_mixer(hidden_states)
    return Qwen4ExpModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ================================================================
# Patch: Qwen4ExpVisionModel.dummy_forward
# 1. Touch the vision tower on text-only FSDP ranks.
# 2. Derive shapes and dtype from the live model instead of hardcoding them.
# ================================================================
@config.override_method(
    "Qwen4ExpVisionModel.dummy_forward",
    description="Add a config-derived dummy vision forward for rank-asymmetric FSDP batches",
)
def qwen4_exp_vision_model_dummy_forward(self):
    # --- Patch.1 / Patch.2 ---
    merge_size = self.spatial_merge_size
    t, h, w = 1, merge_size, merge_size
    config = self.config
    flattened_patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
    dtype = self.patch_embed.proj.weight.dtype
    device = self.patch_embed.proj.weight.device
    pixel_values = torch.zeros((t * h * w, flattened_patch_size), dtype=dtype, device=device)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=device)
    return self(hidden_states=pixel_values, grid_thw=grid_thw)
    # --- Patch.1 / Patch.2 ---


@config.add_helper
def qwen4_exp_mm_token_type_ids(input_ids, config):
    """Build Qwen4 multimodal token types from VeOmni's placeholder ids."""
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == config.image_token_id] = 1
    mm_token_type_ids[input_ids == config.video_token_id] = 2
    return mm_token_type_ids


@config.add_helper
def qwen4_exp_get_position_id(main_func, self, **kwargs):
    """Picklable wrapper used by preprocessing workers."""
    if kwargs.get("mm_token_type_ids") is None and kwargs.get("input_ids") is not None:
        kwargs["mm_token_type_ids"] = qwen4_exp_mm_token_type_ids(kwargs["input_ids"], self.config)
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


@config.add_helper
def qwen4_exp_collate_metadata(batch, sp_pad):
    """Record the position layout and visual tail padding added before SP slicing."""
    batch["qwen4_exp_position_ids_layout"] = "batch_first"
    batch["multimodal_metadata"] = {
        "image_sp_padding": sp_pad.get("pixel_values", 0),
        "video_sp_padding": sp_pad.get("pixel_values_videos", 0),
    }


@config.add_helper
class _Qwen4ExpFakeForPositionIds(SimpleNamespace):
    """Picklable minimal receiver for Qwen4ExpModel.get_rope_index."""

    def get_vision_position_ids(self, *args, **kwargs):
        return Qwen4ExpModel.get_vision_position_ids(self, *args, **kwargs)


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.get_position_id_func
# 1. Expose M-RoPE preprocessing using VeOmni's negative placeholder ids.
# ================================================================
@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_position_id_func",
    description="Expose a picklable Qwen4-Exp multimodal position-id preprocessor",
)
def qwen4_exp_get_position_id_func_patched(self):
    # --- Patch.1 ---
    fake_config = copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = _Qwen4ExpFakeForPositionIds(config=fake_config)
    return partial(qwen4_exp_get_position_id, Qwen4ExpModel.get_rope_index, fake_model)
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.get_metadata_collate_func
# 1. Mark VeOmni-packed position ids as batch-first so Model.forward can
#    distinguish them from HF's canonical axis-first layout.
# ================================================================
@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_metadata_collate_func",
    description="Expose an explicit layout marker for packed Qwen4-Exp position ids",
)
def qwen4_exp_get_metadata_collate_func_patched(self):
    # --- Patch.1 ---
    return qwen4_exp_collate_metadata
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.get_parallel_plan
# 1. Register checkpoint-native PLE shards under the dedicated ``ple``
#    ExtraParallel mesh for row-sharded streaming load and training.
# ================================================================
@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_parallel_plan",
    description="Register the Qwen4-Exp PLE ExtraParallel plan",
)
def qwen4_exp_get_parallel_plan_patched(self):
    # --- Patch.1 ---
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
    # --- Patch.1 ---


# ================================================================
# Patch: Qwen4ExpModel.forward
# 1. Consume VeOmni's precomputed masks after placeholder ids are zeroed.
# 2. Reconstruct real modality ids specifically for PLE n-gram hashing.
# 3. Touch missing vision modalities on FSDP ranks.
# 4. Perform multimodal scatter in global-sequence layout under Ulysses.
# 5. Accept VeOmni's batch-first precomputed M-RoPE layout.
# ================================================================
@config.override_method(
    "Qwen4ExpModel.forward",
    description="Support VeOmni VLM SFT masks, PLE ids, and global placeholder scatter under Ulysses",
)
def qwen4_exp_model_forward_patched(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen4ExpModelOutputWithPast:
    parallel_state = get_parallel_state()
    if parallel_state.cp_enabled and (parallel_state.cp_size not in (2, 4) or not parallel_state.ulysses_enabled):
        raise NotImplementedError("Qwen4-Exp CP candidate requires cp_size in (2, 4) and ulysses_size>1.")
    if parallel_state.sp_enabled and past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp does not support cache prefill or decode under Ulysses.")
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    local_ple_input_ids = None
    if self.config.text_config.ple_layer_ids:
        local_ple_input_ids = (
            input_ids.clone() if input_ids is not None else self.language_model.reverse_embedding(inputs_embeds)
        )

    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    position_ids_layout = kwargs.pop("qwen4_exp_position_ids_layout", None)
    lm_kwargs = {}
    for key in (
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "linear_attn_cu_seq_lens_q",
        "tail_padding_length",
    ):
        if key in kwargs:
            lm_kwargs[key] = kwargs.pop(key)
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}

    if position_ids_layout not in (None, "batch_first"):
        raise ValueError(f"Unsupported Qwen4-Exp position_ids layout: {position_ids_layout!r}")

    if parallel_state.sp_enabled:
        inputs_embeds = gather_outputs(
            inputs_embeds,
            gather_dim=1,
            group=parallel_state.sp_group,
        )

    if image_mask is None or video_mask is None:
        mask_input_ids = input_ids
        if parallel_state.sp_enabled and input_ids is not None:
            mask_input_ids = gather_outputs(
                input_ids,
                gather_dim=1,
                group=parallel_state.sp_group,
            )
        fallback_image_mask, fallback_video_mask = self.get_placeholder_mask(mask_input_ids, inputs_embeds)
        image_mask = fallback_image_mask.squeeze(-1) if image_mask is None else image_mask
        video_mask = fallback_video_mask.squeeze(-1) if video_mask is None else video_mask
    image_mask = image_mask.bool()
    video_mask = video_mask.bool()

    if pixel_values is not None:
        if parallel_state.sp_enabled:
            pixel_values = gather_outputs(
                pixel_values,
                gather_dim=0,
                group=parallel_state.sp_group,
            )
            image_sp_padding = multimodal_metadata.get("image_sp_padding", 0)
            if image_sp_padding:
                pixel_values = pixel_values[:-image_sp_padding]
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        torch_compilable_check(
            image_mask.sum() * inputs_embeds.shape[-1] == image_embeds.numel(),
            "Image features and image placeholder tokens do not match.",
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---

    if pixel_values_videos is not None:
        if parallel_state.sp_enabled:
            pixel_values_videos = gather_outputs(
                pixel_values_videos,
                gather_dim=0,
                group=parallel_state.sp_group,
            )
            video_sp_padding = multimodal_metadata.get("video_sp_padding", 0)
            if video_sp_padding:
                pixel_values_videos = pixel_values_videos[:-video_sp_padding]
        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
        )
        video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        torch_compilable_check(
            video_mask.sum() * inputs_embeds.shape[-1] == video_embeds.numel(),
            "Video features and video placeholder tokens do not match.",
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask.unsqueeze(-1), video_embeds)
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3 ---
        fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---

    if parallel_state.sp_enabled:
        inputs_embeds = slice_input_tensor(
            inputs_embeds,
            dim=1,
            padding=False,
            group=parallel_state.sp_group,
        )
        image_mask = slice_input_tensor(
            image_mask,
            dim=1,
            padding=False,
            group=parallel_state.sp_group,
        )
        video_mask = slice_input_tensor(
            video_mask,
            dim=1,
            padding=False,
            group=parallel_state.sp_group,
        )

    ple_input_ids = None
    if local_ple_input_ids is not None:
        ple_input_ids = local_ple_input_ids
        ple_input_ids.masked_fill_(image_mask, self.config.image_token_id)
        ple_input_ids.masked_fill_(video_mask, self.config.video_token_id)

    if position_ids is None:
        if parallel_state.sp_enabled:
            raise ValueError("Qwen4-Exp Ulysses requires precomputed position_ids from the data collator.")
        tensor_attention_mask = (
            attention_mask.get("full_attention") if isinstance(attention_mask, dict) else attention_mask
        )
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=tensor_attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    # --- Patch.5 ---
    elif position_ids_layout == "batch_first":
        if (
            position_ids.ndim != 3
            or position_ids.shape[0] != inputs_embeds.shape[0]
            or position_ids.shape[1] not in (3, 4)
        ):
            raise ValueError(
                "Qwen4-Exp batch-first position_ids must have shape (batch, 3|4, sequence) matching input_ids."
            )
        position_ids = position_ids.transpose(0, 1).contiguous()
    # --- Patch.5 ---

    kwargs.update(lm_kwargs)
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    if cu_seq_lens_q is None:
        if parallel_state.sp_enabled:
            raise ValueError("Qwen4-Exp SP requires collator-provided global cu_seq_lens_q.")
        cu_seq_lens_q = pos2culen(position_ids[0])
        kwargs["cu_seq_lens_q"] = cu_seq_lens_q
    kwargs.setdefault("linear_attn_cu_seq_lens_q", cu_seq_lens_q)
    _qwen4_exp_validate_packed_seq_lens(
        cu_seq_lens_q.diff().tolist(), inputs_embeds.shape[0], inputs_embeds.shape[1] * parallel_state.sp_size
    )
    if position_ids.shape[0] == 3:
        text_position_ids = culen2pos(cu_seq_lens_q).to(device=position_ids.device, dtype=position_ids.dtype)
        if parallel_state.sp_enabled:
            local_start = parallel_state.sp_rank * inputs_embeds.shape[1]
            text_position_ids = text_position_ids[:, local_start : local_start + inputs_embeds.shape[1]]
        if tuple(text_position_ids.shape) != tuple(position_ids.shape[1:]):
            raise ValueError("Qwen4-Exp text position ids must match the local M-RoPE positions.")
        position_ids = torch.cat((text_position_ids.unsqueeze(0), position_ids), dim=0)
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        ple_input_ids=ple_input_ids,
        **kwargs,
    )
    return Qwen4ExpModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


@config.add_helper
def qwen4_exp_global_aux_router_logits(
    router_logits: tuple[torch.Tensor, ...] | None,
    batch_size: int,
    local_sequence_length: int,
) -> tuple[torch.Tensor, ...] | None:
    """Gather sequence-local router logits for the replicated global auxiliary loss."""
    parallel_state = get_parallel_state()
    if not parallel_state.sp_enabled or router_logits is None:
        return router_logits

    global_router_logits = []
    expected_local_rows = batch_size * local_sequence_length
    for layer_idx, layer_logits in enumerate(router_logits):
        if layer_logits.shape[0] != expected_local_rows:
            raise ValueError(
                "Qwen4-Exp router logits must cover the complete local Ulysses sequence; "
                f"layer {layer_idx} has {layer_logits.shape[0]} rows, expected {expected_local_rows}."
            )
        layer_logits = layer_logits.reshape(batch_size, local_sequence_length, -1)
        layer_logits = gather_outputs(
            layer_logits,
            gather_dim=1,
            group=parallel_state.sp_group,
        )
        global_router_logits.append(layer_logits.flatten(0, 1))
    return tuple(global_router_logits)


@config.add_helper_after("Qwen4ExpCausalLMOutputWithPast")
@dataclass
class Qwen4ExpCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen4ExpCausalLMOutputWithPast):
    """Qwen4-Exp output extended with VeOmni fused-loss auxiliary tensors.

    Args:
        fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
            Per-token values produced by VeOmni's fused-linear loss path.
    """


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.forward
# 1. Use VeOmni's fused-linear-compatible loss contract for VLM SFT and keep
#    model-only metadata out of loss kwargs.
# 2. Preserve Qwen4 MoE router auxiliary loss without enabling MTP loss.
# ================================================================
@config.override_method(
    "Qwen4ExpForConditionalGeneration.forward",
    description="Use VeOmni fused loss for Qwen4-Exp VLM SFT without MTP loss",
)
def qwen4_exp_for_conditional_generation_forward_patched(
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
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen4ExpCausalLMOutputWithLogProbs:
    # --- Patch.1 ---
    position_ids_layout = kwargs.pop("qwen4_exp_position_ids_layout", None)
    # --- Patch.1 ---
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        qwen4_exp_position_ids_layout=position_ids_layout,
        **kwargs,
    )

    hidden_states = outputs[0]
    batch_size, local_sequence_length = hidden_states.shape[:2]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
    else:
        logits = self.lm_head(hidden_states)
    # --- Patch.1 ---

    # --- Patch.2 ---
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        global_router_logits = qwen4_exp_global_aux_router_logits(
            outputs.router_logits,
            batch_size,
            local_sequence_length,
        )
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                global_router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                global_router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)
    # MTP is intentionally absent: no MTP module is constructed and no MTP
    # objective is added to the SFT loss.
    # --- Patch.2 ---

    return Qwen4ExpCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
        fused_linear_aux=fused_linear_aux,
    )
