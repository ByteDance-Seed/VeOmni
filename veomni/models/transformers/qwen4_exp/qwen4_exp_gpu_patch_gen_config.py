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

The sequence-parallel QSA path keeps query states and compact token selections
sequence-local (`[B, S_local, ...]`) and gathers only compressed indexer keys
plus full-attention K/V across the active sequence-parallel group (`ulysses_group`
or `cp_group`), avoiding QKV head all-to-all and removing QSA head-divisibility
constraints. MTP is outside the training model and is filtered by
``checkpoint_tensor_converter.py``.
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
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpVisionModel,
    apply_mask_to_padding_states,
    causal_conv1d_update,
    torch_recurrent_gated_delta_rule,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.context_parallel.dsa_cp import all_gather_compressed_rows, all_gather_kv
from veomni.distributed.parallel_state import _init_parallel_state, get_parallel_state, use_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads
from veomni.models.loss_utils import ForCausalLMLoss, load_balancing_loss
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
    qwen3_5_vision_model_rot_pos_emb,
)
from veomni.models.transformers.qwen4_exp.packed_ple import (
    correct_packed_ple_convolution,
    local_packed_boundary_indices,
    local_packed_positions,
)
from veomni.models.transformers.qwen4_exp.qsa_metadata import PreparedQSAMetadata, prepare_qsa_metadata
from veomni.models.utils.moe_utils import merged_experts_act_fn_forward
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.ops.kernels.swiglu_mlp import has_plain_linear_projections
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils import logging
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin
from veomni.utils.seqlen_pos_transform_utils import pos2culen


logger = logging.get_logger(__name__)


config = PatchConfig(
    source_module="transformers.models.qwen4_exp.modeling_qwen4_exp",
    target_file="patched_modeling_qwen4_exp_gpu.py",
    description="Qwen4-Exp initial GPU VLM-SFT integration with explicit PLE/QSA limits",
)
config.exclude_from_output("apply_rotary_pos_emb", "apply_rotary_pos_emb_vision", "rotate_half")

config.add_import(
    "veomni.models.transformers.qwen4_exp.packed_ple",
    names=["correct_packed_ple_convolution", "local_packed_boundary_indices", "local_packed_positions"],
)


@config.override_method(
    "Qwen4ExpTextRMSNorm.__init__",
    description="Construct a local rms_norm offset VeomniOp",
)
def qwen4_exp_rmsnorm_init_patched(self, dim: int, group_size: int | None = None, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.eps = eps
    self.weight = nn.Parameter(torch.zeros(dim))
    self.group_size = group_size
    if group_size is not None and dim % group_size != 0:
        raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")
    self.veomni_rms_norm = VeomniOp("rms_norm", "offset", resolve_op_impl("rms_norm_implementation"))


@config.override_method(
    "Qwen4ExpTextRMSNorm.forward",
    description="Call rms_norm offset, passing optional group_size",
)
def qwen4_exp_rmsnorm_forward_patched(self, x):
    return self.veomni_rms_norm(x, self.weight, eps=self.eps, group_size=self.group_size)


@config.override_method(
    "Qwen4ExpTextMLP.__init__",
    description="Construct a local swiglu_mlp VeomniOp",
)
def qwen4_exp_text_mlp_init_patched(self, config, intermediate_size=None):
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
    "Qwen4ExpTextMLP.forward",
    description="Call swiglu_mlp for silu/swish on plain nn.Linear projections, otherwise the projection modules",
)
def qwen4_exp_text_mlp_forward_patched(self, x):
    if self.config.hidden_act in {"silu", "swish"} and has_plain_linear_projections(self):
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


config.add_import(
    "veomni.models.transformers.qwen4_exp.qsa_metadata", names=["PreparedQSAMetadata", "prepare_qsa_metadata"]
)
config.add_import("copy", names=["copy"])
config.add_import("dataclasses", names=["dataclass"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.moe.comm", names=["all_to_all"])
config.add_import(
    "veomni.distributed.context_parallel.dsa_cp",
    names=["all_gather_compressed_rows", "all_gather_kv"],
)
config.add_import(
    "veomni.distributed.parallel_state",
    names=["_init_parallel_state", "get_parallel_state", "use_parallel_state"],
)
config.add_import(
    "veomni.distributed.sequence_parallel",
    names=["gather_outputs", "slice_input_tensor", "sp_pad_and_slice"],
)
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_heads_scatter_seq", "gather_seq_scatter_heads"],
)
config.add_import("veomni.distributed.sequence_parallel.gdn_exchange", names=["coalesced_gather_aligned"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import("veomni.ops.config", names=["resolve_op_impl"])
config.add_import("veomni.ops.kernels.swiglu_mlp", names=["has_plain_linear_projections"])
config.add_import("veomni.models.utils.moe_utils", names=["merged_experts_act_fn_forward"])
config.add_import("veomni.models.loss_utils", names=["ForCausalLMLoss", "load_balancing_loss"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.add_import("veomni.utils.model_outputs", names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"])
config.add_import("veomni.utils.seqlen_pos_transform_utils", names=["pos2culen"])
config.add_import("veomni.utils.moe_router_replay", names=["get_active_replay", "maybe_replay_indices"])
config.add_post_import_block("_VEOMNI_VISION_ATTENTION_PATCHED = True")
config.add_import("veomni.utils", names=["logging"])
config.add_post_import_block("""
from veomni.utils import logging
logger = logging.get_logger(__name__)
""")


# ``all_to_all`` is imported into the generated file; this sentinel exists
# so the PLE replacement class can mention it while the patch config loads.
all_to_all = None

# Names resolved at codegen time from generated imports.
is_flash_attention_requested = None


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
            # Preserve the upstream recovery path while making exception chaining
            # explicit so generated code passes the repository's B904 lint gate.
            raise RuntimeError(
                "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and "
                "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Qwen4-Exp needs "
                "to reverse the embedding for PLE, provide exact embedding-table values or pass `ple_input_ids`."
            ) from None
            # --- Patch.1 ---
    return input_ids


@config.override_method(
    "Qwen4ExpModel.__init__",
    description="Build local patched submodels and propagate the VeOmni MoE implementation",
)
def qwen4_exp_model_init_patched(self, config):
    # --- Patch.1: Propagate the selected MoE backend into the nested text config. ---
    config.text_config._moe_implementation = getattr(config, "_moe_implementation", "eager")
    # --- Patch.1 ---

    super().__init__(config)
    # --- Patch.2: Build local patched submodels instead of AutoModel ---
    # VeOmni patches are retained inside the VLM wrapper. ---
    self.visual = Qwen4ExpVisionModel._from_config(config.vision_config)
    self.language_model = Qwen4ExpTextModel._from_config(config.text_config)
    # --- Patch.2 ---
    self.rope_deltas = None
    self.post_init()


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
def _qwen4_exp_sequence_parallel_state(parallel_state):
    """Resolve the active sequence-parallel group/rank/size across Ulysses and CP."""
    ulysses_enabled = bool(getattr(parallel_state, "ulysses_enabled", False))
    cp_enabled = bool(getattr(parallel_state, "cp_enabled", False))
    if ulysses_enabled and cp_enabled:
        if not getattr(parallel_state, "allow_hybrid_cp", False):
            raise NotImplementedError("Combined Qwen4-Exp Ulysses/CP requires allow_hybrid_cp=True.")
        # The collator slices contiguous sequence shards in unified SP rank
        # order. QSA keeps those local queries and gathers KV over that group;
        # GDN exchanges sequence for heads over the same complete group.
        return True, parallel_state.sp_group, parallel_state.sp_rank, parallel_state.sp_size
    if cp_enabled:
        return True, parallel_state.cp_group, parallel_state.cp_rank, parallel_state.cp_size
    if ulysses_enabled:
        return True, parallel_state.ulysses_group, parallel_state.ulysses_rank, parallel_state.ulysses_size
    return False, None, 0, 1


@config.add_helper
def _qwen4_exp_validate_sp_attention_mask(attention_mask, local_seq_len: int, sp_size: int, sp_rank: int):
    """Validate and locally slice a Qwen4-Exp text mask before any collective."""
    if not isinstance(attention_mask, torch.Tensor):
        return attention_mask
    kv_full_len = local_seq_len * sp_size
    if attention_mask.ndim < 3 or attention_mask.shape[-2] not in (1, local_seq_len, kv_full_len):
        raise ValueError(
            "Qwen4-Exp sequence parallelism expects a 3D/4D attention_mask whose query "
            f"dimension (-2) is 1, the local shard length ({local_seq_len}), or the full "
            f"sequence length ({kv_full_len}); got shape {tuple(attention_mask.shape)}."
        )
    if attention_mask.shape[-2] == kv_full_len:
        return attention_mask.narrow(-2, sp_rank * local_seq_len, local_seq_len)
    return attention_mask


@config.add_helper
def _qwen4_exp_vision_parallel_state(parallel_state):
    """Return a Ulysses-enabled ParallelState for Qwen4-Exp vision modules."""
    if not bool(getattr(parallel_state, "cp_enabled", False)):
        return parallel_state
    return _init_parallel_state(
        dp_size=parallel_state.dp_size,
        dp_replicate_size=parallel_state.dp_replicate_size,
        dp_shard_size=parallel_state.dp_shard_size,
        tp_size=parallel_state.tp_size,
        pp_size=parallel_state.pp_size,
        cp_size=1,
        ulysses_size=parallel_state.sp_size,
        dp_mode=parallel_state.dp_mode,
        device_type=parallel_state.device_type,
        include_sp_in_fsdp=parallel_state.include_sp_in_fsdp,
        async_enabled=parallel_state.async_enabled,
        name=None,
    )


@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.__init__",
    description="Bind instance-local GDN VeomniOps; keep Qwen4ExpTextRMSNormGated for the weight",
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
    # --- Patch.1 ---
    self.veomni_rms_norm_gated = VeomniOp(
        "rms_norm_gated",
        "standard",
        resolve_op_impl("rms_norm_gated_implementation"),
    )
    self.veomni_causal_conv1d = VeomniOp(
        "causal_conv1d",
        "standard",
        resolve_op_impl("causal_conv1d_implementation"),
    )
    self.veomni_chunk_gated_delta_rule = VeomniOp(
        "chunk_gated_delta_rule",
        "standard",
        resolve_op_impl("chunk_gated_delta_rule_implementation"),
    )
    # --- Patch.1 ---


@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.forward",
    description="Run Qwen4-Exp GatedDeltaNet in full-sequence/local-head Ulysses layout",
)
def qwen4_exp_gated_deltanet_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    # --- Patch.1 ---
    # Freeze the configured GDN kernels on each model instance.
    # Exchange local sequence ownership for local head ownership under Ulysses.
    # Slice depthwise-convolution and recurrent parameters by local head range.
    # Restore local-sequence/full-head layout before the output gate.
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    parallel_state = get_parallel_state()
    sp_enabled, sp_group, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    # --- Patch.2: Reuse host-side packed lengths threaded through kwargs. ---
    # The trainer moves cu_seq_lens_q onto the device, so deriving the lengths
    # here would sync GPU->CPU once per layer; the model forward exports the
    # list once and threads it through kwargs instead.
    if cu_seq_lens_q is None:
        packed_seq_lens = None
    else:
        sequence_length = seq_len * sp_size if sp_enabled else seq_len
        if packed_seq_lens is None:
            packed_seq_lens = cu_seq_lens_q.diff().tolist()
        packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(packed_seq_lens, batch_size, sequence_length)
    # --- Patch.2 ---

    if sp_enabled and cache_params is not None:
        raise NotImplementedError(
            "Qwen4-Exp GatedDeltaNet does not support KV/recurrent cache state under Ulysses/CP."
        )

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx, state_idx=0)
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if sp_enabled:
        if self.num_k_heads % sp_size != 0 or self.num_v_heads % sp_size != 0:
            raise ValueError(
                f"sp_size ({sp_size}) must divide Qwen4-Exp GatedDeltaNet key heads "
                f"({self.num_k_heads}) and value heads ({self.num_v_heads})."
            )
        if self.veomni_causal_conv1d.impl == "eager" or self.veomni_chunk_gated_delta_rule.impl == "eager":
            raise RuntimeError(
                "Qwen4-Exp GatedDeltaNet Ulysses/CP requires non-eager causal_conv1d and "
                "chunk_gated_delta_rule implementations."
            )

        local_num_k_heads = self.num_k_heads // sp_size
        local_num_v_heads = self.num_v_heads // sp_size
        local_key_dim = local_num_k_heads * self.head_k_dim
        local_value_dim = local_num_v_heads * self.head_v_dim

        q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        if q_proj.dtype == k_proj.dtype == v_proj.dtype == b.dtype == a.dtype:
            q_proj, k_proj, v_proj, b, a = coalesced_gather_aligned(
                q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim),
                k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim),
                v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim),
                b,
                a,
                sp_group,
            )
        else:
            q_proj = gather_seq_scatter_heads(
                q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim),
                seq_dim=1,
                head_dim=2,
                group=sp_group,
            )
            k_proj = gather_seq_scatter_heads(
                k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim),
                seq_dim=1,
                head_dim=2,
                group=sp_group,
            )
            v_proj = gather_seq_scatter_heads(
                v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim),
                seq_dim=1,
                head_dim=2,
                group=sp_group,
            )
            b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2, group=sp_group)
            a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2, group=sp_group)
        mixed_qkv = torch.cat(
            (
                q_proj.flatten(2),
                k_proj.flatten(2),
                v_proj.flatten(2),
            ),
            dim=-1,
        )

        full_weight = self.conv1d.weight.squeeze(1)
        key_offset = sp_rank * local_key_dim
        value_offset = sp_rank * local_value_dim
        conv_weight = torch.cat(
            (
                full_weight[key_offset : key_offset + local_key_dim],
                full_weight[self.key_dim + key_offset : self.key_dim + key_offset + local_key_dim],
                full_weight[2 * self.key_dim + value_offset : 2 * self.key_dim + value_offset + local_value_dim],
            ),
            dim=0,
        )
        mixed_qkv = self.veomni_causal_conv1d(
            mixed_qkv,
            conv_weight,
            self.conv1d.bias,
            cu_seq_lens_q,
            activation=self.activation,
            seq_idx=None,
            backend="triton",
        )
    else:
        local_num_k_heads = self.num_k_heads
        local_num_v_heads = self.num_v_heads
        local_key_dim = self.key_dim
        local_value_dim = self.value_dim
        if packed_seq_lens is not None:
            if cache_params is not None:
                raise ValueError("Qwen4-Exp packed GDN training does not support recurrent cache state.")
            conv_weight = self.conv1d.weight.squeeze(1)
            if self.veomni_causal_conv1d.impl == "eager":
                # Eager has no varlen path, so run each packed segment on its own.
                mixed_qkv = torch.cat(
                    [
                        self.veomni_causal_conv1d(
                            segment,
                            conv_weight,
                            self.conv1d.bias,
                            None,
                            activation=self.activation,
                            seq_idx=None,
                            backend="triton",
                        )
                        for segment in mixed_qkv.split(packed_seq_lens, dim=1)
                    ],
                    dim=1,
                )
            else:
                mixed_qkv = self.veomni_causal_conv1d(
                    mixed_qkv,
                    conv_weight,
                    self.conv1d.bias,
                    cu_seq_lens_q,
                    activation=self.activation,
                    seq_idx=None,
                    backend="triton",
                )
        elif use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            mixed_qkv = mixed_qkv.transpose(1, 2)
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            ).transpose(1, 2)
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv.transpose(1, 2), self.layer_idx, conv_kernel_size=self.conv_kernel_size
                ).transpose(1, 2)
            mixed_qkv = self.veomni_causal_conv1d(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                None,
                activation=self.activation,
                seq_idx=None,
                backend="triton",
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, -seq_len:]

    query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
    query = query.reshape(batch_size, -1, local_num_k_heads, self.head_k_dim).contiguous()
    key = key.reshape(batch_size, -1, local_num_k_heads, self.head_k_dim).contiguous()
    value = value.reshape(batch_size, -1, local_num_v_heads, self.head_v_dim).contiguous()
    beta = b.sigmoid()

    if sp_enabled:
        value_head_start = sp_rank * local_num_v_heads
        value_head_slice = slice(value_head_start, value_head_start + local_num_v_heads)
        g = -self.A_log[value_head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[value_head_slice])
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1 and not (
        self.veomni_chunk_gated_delta_rule.impl == "flash_qla" and not use_precomputed_states
    ):
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
    if sp_enabled or (packed_seq_lens is not None and self.veomni_chunk_gated_delta_rule.impl != "eager"):
        core_attn_out, last_recurrent_state = self.veomni_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            None,
            cu_seq_lens_q,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
    elif packed_seq_lens is not None:
        # Eager has no varlen path, so run each packed segment on its own.
        outputs = []
        for q_segment, k_segment, v_segment, g_segment, beta_segment in zip(
            query.split(packed_seq_lens, dim=1),
            key.split(packed_seq_lens, dim=1),
            value.split(packed_seq_lens, dim=1),
            g.split(packed_seq_lens, dim=1),
            beta.split(packed_seq_lens, dim=1),
            strict=True,
        ):
            segment_output, _ = self.veomni_chunk_gated_delta_rule(
                q_segment,
                k_segment,
                v_segment,
                g_segment,
                beta_segment,
                None,
                None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            outputs.append(segment_output)
        core_attn_out = torch.cat(outputs, dim=1)
        last_recurrent_state = None
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
        core_attn_out, last_recurrent_state = self.veomni_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            recurrent_state,
            kwargs.pop("cu_seq_lens_q", None),
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    if sp_enabled:
        core_attn_out = gather_heads_scatter_seq(
            core_attn_out,
            head_dim=2,
            seq_dim=1,
            group=sp_group,
        )

    core_attn_out = self.veomni_rms_norm_gated(
        core_attn_out.reshape(-1, self.head_v_dim),
        z.reshape(-1, self.head_v_dim),
        self.norm.weight,
        eps=self.layer_norm_epsilon,
        activation=self.norm.activation,
    )
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)
    # --- Patch.1 ---


@config.modify_init("Qwen4ExpTextQSAIndexer", description="Bind an instance-local partial rope VeomniOp")
def qwen4_exp_text_qsa_indexer_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))


@config.override_method(
    "Qwen4ExpTextQSAIndexer.forward",
    description="Select compact global QSA token indices for local queries under Ulysses/CP",
)
def qwen4_exp_qsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    qsa_metadata: PreparedQSAMetadata | None = None,
) -> torch.Tensor:
    # --- Patch.1: Select compact global QSA token indices for local queries under Ulysses/CP. ---
    # Using sequence-local RoPE inputs and gathering only the compressed keys
    # across the sequence-parallel group while returning local query selections.
    del attention_mask
    parallel_state = get_parallel_state()
    if past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp compact QSA does not support a KV/indexer cache.")

    _, group, rank, world_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    batch_size, local_seq_len, _ = hidden_states.shape
    global_seq_len = local_seq_len * world_size
    if global_seq_len + self.compress_ratio > torch.iinfo(torch.int32).max:
        raise ValueError("Qwen4-Exp compact QSA token positions must fit int32, including compression padding.")
    if self.index_kv_heads != 1:
        raise ValueError(f"Qwen4-Exp compact QSA currently requires one indexer KV head; got {self.index_kv_heads}.")
    if self.compress_ratio > local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA needs each sequence-parallel shard to be at least one compression block wide; "
            f"got local_seq_len={local_seq_len}, compress_ratio={self.compress_ratio}."
        )

    qk = self.index_qk_proj(hidden_states)
    q_width = self.index_n_heads * self.index_head_dim
    q, raw_keys = torch.split(qk, [q_width, self.index_head_dim], dim=-1)
    q = self.q_layernorm(q.reshape(batch_size, local_seq_len, self.index_n_heads, self.index_head_dim))
    local_cos, local_sin = position_embeddings
    if local_cos.shape[1] != local_seq_len or local_sin.shape[1] != local_seq_len:
        raise ValueError(
            "Qwen4-Exp compact QSA position embeddings must match the local sequence length; "
            f"got cos={local_cos.shape[1]}, sin={local_sin.shape[1]}, local_seq_len={local_seq_len}."
        )
    local_start = rank * local_seq_len
    q, _ = self.veomni_rope(q, q, local_cos, local_sin, unsqueeze_dim=2)

    # Reuse tensors prepared once outside the decoder/checkpointed layer loop.
    # Direct component callers retain the same boundary validation and layout.
    if qsa_metadata is None:
        qsa_metadata = prepare_qsa_metadata(
            batch_size,
            local_seq_len,
            self.compress_ratio,
            rank,
            world_size,
            hidden_states.device,
            cu_seq_lens_q,
            packed_seq_lens,
        )
    if qsa_metadata.layout != (batch_size, local_seq_len, self.compress_ratio, rank, world_size):
        raise ValueError("Qwen4-Exp QSA metadata does not match the current sequence-parallel layout.")
    if qsa_metadata.local_segment_ids.device != hidden_states.device:
        raise ValueError("Qwen4-Exp QSA metadata must be on the hidden states' device.")
    local_batch_ids = qsa_metadata.local_batch_ids
    local_block_starts = qsa_metadata.local_block_starts
    block_segment_ids = qsa_metadata.block_segment_ids
    block_starts = qsa_metadata.block_starts
    local_segment_ids = qsa_metadata.local_segment_ids
    local_segment_starts = qsa_metadata.local_segment_starts

    halo = self.compress_ratio - 1
    if halo == 0 or world_size == 1:
        right_halo = raw_keys[:, :0]
    else:
        prefixes = gather_outputs(raw_keys[:, :halo].contiguous(), gather_dim=1, group=group)
        if rank + 1 == world_size:
            right_halo = prefixes[:, :halo] * 0
        else:
            halo_start = (rank + 1) * halo
            right_halo = prefixes[:, halo_start : halo_start + halo]
    extended_keys = torch.cat((raw_keys, right_halo), dim=1)

    if local_batch_ids.numel():
        local_offsets = local_block_starts - local_start
        window_indices = local_offsets[:, None] + torch.arange(self.compress_ratio, device=hidden_states.device)[None]
        pooled_keys = extended_keys[local_batch_ids[:, None], window_indices].float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys)
        # Rotate the owned blocks as one [1, 1, blocks, D] sequence, the rope op's
        # [B, H, S, D] layout, with per-block start positions.
        pooled_keys = pooled_keys[None, None]
        pooled_keys, _ = self.veomni_rope(
            pooled_keys,
            pooled_keys,
            local_cos[local_batch_ids, local_offsets][None],
            local_sin[local_batch_ids, local_offsets][None],
        )
        pooled_keys = pooled_keys[0]
    else:
        # Keep both the projection and RMSNorm parameters in the autograd graph
        # on ranks that own no complete block.
        pooled_keys = self.k_layernorm(extended_keys[:, :0]).reshape(1, 0, self.index_head_dim)

    if world_size > 1:
        pooled_keys = all_gather_compressed_rows(pooled_keys, qsa_metadata.counts, group)
    output_width = self.token_budget + self.compress_ratio - 1
    query_positions = local_start + torch.arange(local_seq_len, device=hidden_states.device)

    if block_starts.numel():
        top_count = min(self.block_topk, block_starts.numel())
        block_offsets = torch.arange(self.compress_ratio, dtype=torch.int32, device=hidden_states.device)
        elements_per_query = max(1, batch_size * block_starts.numel() * self.index_n_heads)
        query_chunk_size = max(1, min(local_seq_len, 16 * 1024 * 1024 // elements_per_query))
        selected_chunks = []
        for query_start in range(0, local_seq_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, local_seq_len)
            scores = torch.einsum(
                "blhd,nd->blnh",
                q[:, query_start:query_end].float(),
                pooled_keys[0].float(),
            )
            scores = torch.relu(scores).sum(dim=-1) / self.index_head_dim**0.5
            visible = (block_segment_ids[None, None] == local_segment_ids[:, query_start:query_end, None]) & (
                block_starts[None, None] + self.compress_ratio - 1
                <= query_positions[None, query_start:query_end, None]
            )
            scores = scores.masked_fill(~visible, float("-inf"))
            top_blocks = scores.topk(top_count, dim=-1).indices
            top_valid = (
                visible.to(torch.int8).gather(-1, top_blocks).bool()
                if hidden_states.device.type == "npu"
                else visible.gather(-1, top_blocks)
            )
            selected_starts = block_starts[top_blocks].to(torch.int32)
            selected_chunk = selected_starts[..., None] + block_offsets
            selected_chunks.append(selected_chunk.masked_fill(~top_valid[..., None], -1).flatten(-2))
        selected_local = torch.cat(selected_chunks, dim=1)
    else:
        selected_local = torch.empty(
            batch_size,
            local_seq_len,
            0,
            dtype=torch.long,
            device=hidden_states.device,
        )

    visible_count = query_positions[None] - local_segment_starts + 1
    tail_count = torch.remainder(visible_count, self.compress_ratio)
    tail_start = (
        local_segment_starts
        + torch.div(visible_count, self.compress_ratio, rounding_mode="floor") * self.compress_ratio
    )
    tail_offsets = torch.arange(self.compress_ratio - 1, device=hidden_states.device)
    tails = tail_start[..., None] + tail_offsets
    tails = tails.to(torch.int32).masked_fill(tail_offsets[None, None] >= tail_count[..., None], -1)

    selected_local = torch.cat((selected_local, tails), dim=-1)
    if selected_local.shape[-1] < output_width:
        selected_local = torch.nn.functional.pad(
            selected_local,
            (0, output_width - selected_local.shape[-1]),
            value=-1,
        )
    selected_local = selected_local[..., :output_width].to(torch.int32).contiguous()
    return selected_local
    # --- Patch.1 ---


@config.modify_init("Qwen4ExpTextAttention", description="Bind instance-local rope and QSA attention VeomniOps")
def qwen4_exp_text_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))
    self.veomni_qsa_attn = VeomniOp("qsa_attention", "standard", resolve_op_impl("qsa_attention_implementation"))


@config.override_method(
    "Qwen4ExpTextAttention.forward",
    description="Call the local rope and QSA attention VeomniOps with local queries and gathered KV",
)
def qwen4_exp_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # --- Patch.1 ---
    parallel_state = get_parallel_state()
    sp_enabled, sp_group, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    if sp_enabled and past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp QSA does not support a KV cache under Ulysses/CP.")
    if sp_enabled:
        # Every rank must reject an unsupported mask shape before the indexer
        # enters its compressed-row or halo collectives.
        attention_mask = _qwen4_exp_validate_sp_attention_mask(
            attention_mask,
            hidden_states.shape[1],
            sp_size,
            sp_rank,
        )
    # --- Patch.1 ---

    # --- Patch.4: Forward the host-side packed lengths so the indexer never
    # syncs on cu_seq_lens_q inside the training hot path. ---
    selection = self.indexer(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
        packed_seq_lens=kwargs.get("packed_seq_lens"),
        qsa_metadata=kwargs.pop("qsa_metadata", None),
    )
    # --- Patch.4 ---
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

    # --- Patch.2: Apply local RoPE before gathering global KV under sequence parallelism. ---
    # Under Ulysses/CP these embeddings already belong to this rank's sequence
    # shard. The trailing slice preserves the non-SP cache continuation path.
    cos, sin = (tensor[:, -hidden_states.shape[1] :, :] for tensor in position_embeddings)
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin, unsqueeze_dim=2)
    # --- Patch.2 ---

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            self.layer_idx,
        )
        query_states = query_states.transpose(1, 2)
    elif sp_enabled:
        query_states = query_states.transpose(1, 2)
        key_states = all_gather_kv(key_states.transpose(1, 2), sp_group)
        value_states = all_gather_kv(value_states.transpose(1, 2), sp_group)
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    # --- Patch.3 ---
    # The compact selection already encodes causality and packed-sample
    # boundaries, so the QSA op takes it instead of a dense mask.
    attn_output, attn_weights = self.veomni_qsa_attn(
        query_states,
        key_states,
        value_states,
        selection,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        attention_mask=attention_mask,
    )
    # --- Patch.3 ---
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output * torch.sigmoid(gate))
    return attn_output, attn_weights


@config.replace_class(
    "Qwen4ExpTextExperts",
    description="Always call moe_experts VeomniOp on v5 gate_up_proj weights",
)
class PatchedQwen4ExpTextExperts(nn.Module):
    """Qwen4-Exp expert tensors with a local moe_experts VeomniOp."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
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
        # --- Patch.1 ---
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
        # --- Patch.1 ---


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
# 6. Reuse host-side packed lengths threaded through kwargs so the PLE lookup
#    never derives them from the on-device cu_seq_lens_q (GPU->CPU sync).
# 7. Sort shard ids once and pay a single host sync for the per-shard split
#    sizes instead of one nonzero sync per shard in every PLE lookup.
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

        # --- Patch.1 / Patch.2: Add VeOmni n-gram embedding and metadata handling. ---
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
        # --- Patch.7: Sort once and take a single host sync for the per-shard
        # split sizes. The previous ``torch.where(shard_ids == shard_idx)[0]``
        # nonzero-synced the match count to the host once per shard per lookup,
        # serialising the host against the device on the PLE hot path. ---
        order = torch.argsort(shard_ids, stable=True)
        shard_counts = torch.bincount(shard_ids, minlength=len(self.ngram_embedding)).tolist()
        shard_start = 0
        for embedding, shard_count in zip(self.ngram_embedding.values(), shard_counts, strict=True):
            shard_end = shard_start + shard_count
            if shard_end > shard_start:
                positions = order[shard_start:shard_end]
                weight = embedding.weight
                if hasattr(weight, "to_local"):
                    weight = weight.to_local()
                values = nn.functional.embedding(row_ids[positions], weight).to(output.dtype)
                output = output.index_copy(0, positions, values)
            shard_start = shard_end
        # --- Patch.7 ---
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

        # --- Patch.3: Apply VeOmni PLE input preprocessing. ---
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
        # --- Patch.5: Use the VeOmni PLE lookup path. ---
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
        packed_seq_lens: list[int] | tuple[int, ...] | None = None,
        packed_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del packed_seq_lens
        if cu_seq_lens_q is not None and packed_positions is None:
            parallel_state = get_parallel_state()
            sp_enabled, _, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
            packed_positions = local_packed_positions(
                cu_seq_lens_q,
                input_ids.shape[0],
                input_ids.shape[1],
                sp_rank=sp_rank if sp_enabled else 0,
                sp_size=sp_size if sp_enabled else 1,
                device=input_ids.device,
            )
        if packed_positions is not None and past_key_values is not None:
            raise NotImplementedError("Packed PLE boundaries cannot be combined with cached n-gram history.")
        input_ids = input_ids.long()
        parallel_state = get_parallel_state()
        sp_enabled, sp_group, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
        if sp_enabled and past_key_values is not None:
            raise NotImplementedError("Qwen4-Exp PLE n-gram history does not support cache state under Ulysses/CP.")
        if sp_enabled and self.context_len > 0:
            if input_ids.shape[1] < self.context_len:
                raise ValueError(
                    f"The local Ulysses/CP sequence length ({input_ids.shape[1]}) must be at least the PLE "
                    f"n-gram halo ({self.context_len})."
                )
            local_tail = input_ids[:, -self.context_len :].contiguous()
            gathered_tails = [torch.empty_like(local_tail) for _ in range(sp_size)]
            dist.all_gather(gathered_tails, local_tail, group=sp_group)
            previous_context = (
                input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
                if sp_rank == 0
                else gathered_tails[sp_rank - 1]
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
        if packed_positions is not None:
            if packed_positions.shape != input_ids.shape:
                raise ValueError("Packed PLE n-gram positions must match the local token IDs.")
            # Reset history at each packed sample boundary, including the SP
            # halo, without gathering or slicing the full global input.
            shifted_tokens = [
                torch.where(
                    packed_positions >= shift,
                    shifted[:, -input_ids.shape[1] :],
                    self.eos_token_id,
                )
                for shift, shifted in enumerate(shifted_tokens)
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
        # --- Patch.5: Load distributed PLE rows through the VeOmni lookup path. ---
        embeddings = self._distributed_lookup(shard_ids, row_ids, output_dtype=output_dtype)
        # --- Patch.5 ---
        return embeddings.view(*original_shape, -1).flatten(-2)


@config.override_method(
    "Qwen4ExpTextPLELayer._short_conv",
    description="Exchange differentiable PLE dilated-convolution halos under Ulysses/CP",
)
def qwen4_exp_text_ple_layer_short_conv_patched(
    self,
    hidden_states: torch.Tensor,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    packed_positions: torch.Tensor | None = None,
    packed_boundary_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    parallel_state = get_parallel_state()
    sp_enabled, sp_group, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    if cu_seq_lens_q is not None and packed_positions is None:
        packed_positions = local_packed_positions(
            cu_seq_lens_q,
            hidden_states.shape[0],
            hidden_states.shape[1],
            sp_rank=sp_rank if sp_enabled else 0,
            sp_size=sp_size if sp_enabled else 1,
            device=hidden_states.device,
        )
        packed_boundary_indices = local_packed_boundary_indices(
            cu_seq_lens_q,
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.short_conv_state_len,
            sp_rank=sp_rank if sp_enabled else 0,
            sp_size=sp_size if sp_enabled else 1,
            device=hidden_states.device,
            packed_seq_lens=packed_seq_lens,
        )
    if packed_positions is not None and past_key_values is not None:
        raise NotImplementedError("Packed PLE boundaries cannot be combined with cached convolution state.")
    if (packed_positions is None) != (packed_boundary_indices is None):
        raise ValueError("Packed PLE positions and boundary indices must be provided together.")
    if not sp_enabled:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        if past_key_values is not None:
            hidden_states = past_key_values.update_conv_state(
                hidden_states, self.layer_idx, state_idx=1, conv_kernel_size=self.short_conv_state_len
            )
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]
        conv_output = self.conv1d(hidden_states)
        if packed_positions is not None:
            conv_output = correct_packed_ple_convolution(
                hidden_states, conv_output, self.conv1d, packed_positions, packed_boundary_indices
            )
        return F.silu(conv_output).transpose(1, 2)

    if past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp PLE dilated convolution does not support cache state under Ulysses/CP.")
    # --- Patch.1: Add a differentiable left halo for the dilated depthwise convolution ---
    halo_length = self.short_conv_state_len
    if halo_length == 0:
        return F.silu(self.conv1d(hidden_states.transpose(1, 2))).transpose(1, 2)
    if hidden_states.shape[1] < halo_length:
        raise ValueError(
            f"The local Ulysses/CP sequence length ({hidden_states.shape[1]}) must be at least the PLE "
            f"convolution halo ({halo_length})."
        )

    local_tail = hidden_states[:, -halo_length:, :].contiguous()
    gathered_tails = gather_outputs(
        local_tail,
        gather_dim=1,
        group=sp_group,
    )
    # --- Patch.1 ---
    # --- Patch.2: Keep the original cache/padding path unchanged when Ulysses/CP is disabled. ---
    if sp_rank == 0:
        left_halo = gathered_tails[:, :halo_length, :] * 0
    else:
        start = (sp_rank - 1) * halo_length
        left_halo = gathered_tails[:, start : start + halo_length, :]
    # --- Patch.2 ---
    conv_input = torch.cat((left_halo, hidden_states), dim=1).transpose(1, 2)
    conv_output = self.conv1d(conv_input)
    if packed_positions is not None:
        conv_output = correct_packed_ple_convolution(
            conv_input, conv_output, self.conv1d, packed_positions, packed_boundary_indices
        )
    return F.silu(conv_output).transpose(1, 2)


@config.override_method(
    "Qwen4ExpTextPLELayer.forward",
    description="Match PLE lookup results to the mixed-precision activation dtype before communication",
)
def qwen4_exp_text_ple_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    past_key_values: Cache | None,
    conv_mask: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    packed_positions: torch.Tensor | None = None,
    packed_boundary_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if packed_positions is not None and past_key_values is not None:
        raise NotImplementedError("Packed PLE boundaries cannot be combined with cached PLE state.")
    # --- Patch.1: Run the PLE layer with VeOmni sequence-parallel handling. ---
    # Keep FP32 PLE master weights while casting sparse lookup results to the
    # activation dtype before the result all-to-all and downstream projections.
    embeddings = self.ple_embedding(
        input_ids,
        past_key_values,
        output_dtype=hidden_states.dtype,
        cu_seq_lens_q=cu_seq_lens_q,
        packed_seq_lens=packed_seq_lens,
        packed_positions=packed_positions,
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
        packed_seq_lens=packed_seq_lens,
        packed_positions=packed_positions,
        packed_boundary_indices=packed_boundary_indices,
    )
    return output


# Patch: Qwen4ExpTextDecoderLayer.forward
# 1. Thread the collator's packed boundaries into the PLE token mixers, so
#    the PLE n-gram history and short convolution reset state at every packed
#    boundary instead of leaking context across segments (upstream
#    DecoderLayer.forward never forwards cu_seq_lens_q to self.ple).
# 2. Forward the host-side packed lengths exported once per forward alongside
#    the device tensor, so the PLE mixers reuse them instead of re-deriving
#    them (GPU->CPU sync) from the on-device cu_seq_lens_q.
# ================================================================
@config.override_method(
    "Qwen4ExpTextDecoderLayer.forward",
    description="Pass packed sequence boundaries to Qwen4-Exp PLE",
)
def qwen4_exp_text_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    conv_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    ple_input_ids: torch.LongTensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
    ple_packed_positions: torch.Tensor | None = None,
    ple_packed_boundary_indices: torch.Tensor | None = None,
    qsa_metadata: PreparedQSAMetadata | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.FloatTensor:
    # --- Patch.1: Thread the packed boundaries into the PLE token mixers. ---
    if self.ple is not None:
        # A cached forward with one segment is ordinary decode/prefill, not a
        # packed batch. Keep the cache path's implicit history while retaining
        # cu_seq_lens_q for the attention layers below.
        cached_single_segment = (
            past_key_values is not None and packed_seq_lens is not None and len(packed_seq_lens) == 1
        )
        hidden_states = hidden_states + self.ple(
            hidden_states,
            ple_input_ids,
            past_key_values,
            conv_mask=conv_mask,
            cu_seq_lens_q=None if cached_single_segment else cu_seq_lens_q,
            packed_seq_lens=packed_seq_lens,
            packed_positions=ple_packed_positions,
            packed_boundary_indices=ple_packed_boundary_indices,
        )
    # --- Patch.1 ---

    hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states,
            cache_params=past_key_values,
            attention_mask=conv_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            packed_seq_lens=packed_seq_lens,
            **kwargs,
        )
    else:
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cu_seq_lens_q=cu_seq_lens_q,
            packed_seq_lens=packed_seq_lens,
            qsa_metadata=qsa_metadata,
            **kwargs,
        )

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)

    hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
    hidden_states = self.mlp(hidden_states)

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)
    return hidden_states


@config.override_method(
    "Qwen4ExpTextModel.forward",
    description="Coordinate global QSA metadata with local RoPE/GDN/PLE tensors under Ulysses",
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

    # --- Patch.1: Prepare VeOmni inputs and sequence-parallel metadata. ---
    # Defer the quadratic QSA mask to the attention backend while retaining
    # sequence-local M-RoPE embeddings under Ulysses/CP.
    parallel_state = get_parallel_state()
    sp_enabled, sp_group, sp_rank, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    if sp_enabled and (use_cache or past_key_values is not None):
        raise NotImplementedError("Qwen4-Exp does not support cache prefill or decode under Ulysses/CP.")
    # --- Patch.1 ---
    # --- Patch.5: Export host-side packed lengths once per forward. ---
    # The trainer moves cu_seq_lens_q onto the device, so every layer deriving
    # the segment lengths from it would sync GPU->CPU once per layer. Pop the
    # list exported by the top-level VLM forward, or derive it here exactly
    # once for direct TextModel callers (external inference, unit tests).
    packed_seq_lens = kwargs.pop("packed_seq_lens", None)
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    if cu_seq_lens_q is not None and packed_seq_lens is None:
        sequence_length = inputs_embeds.shape[1] * sp_size if sp_enabled else inputs_embeds.shape[1]
        packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
            cu_seq_lens_q.diff().tolist(), inputs_embeds.shape[0], sequence_length
        )
    cache_active = bool(use_cache or past_key_values is not None)
    cached_single_segment = cache_active and packed_seq_lens is not None and len(packed_seq_lens) == 1
    if cache_active and self.config.ple_layer_ids and cu_seq_lens_q is not None and not cached_single_segment:
        raise NotImplementedError("Packed PLE boundaries cannot be combined with cached PLE state.")
    # --- Patch.5 ---
    # CODEPATH: @ArthurZucker fix flagging for no reason here
    if self.config.ple_layer_ids and ple_input_ids is None:
        # If we do not have input_ids but have ple, we need to revert the embeddings to find back the ids
        ple_input_ids = input_ids if input_ids is not None else self.reverse_embedding(inputs_embeds)

    # Packed PLE metadata is rank-local and independent of the decoder layer.
    # Build it once here so PLE token mixers never rediscover boundaries or
    # materialize a global segment x reach window on every layer.
    ple_packed_positions = None
    ple_packed_boundary_indices = None
    if self.config.ple_layer_ids and cu_seq_lens_q is not None and not cached_single_segment:
        ple_reach = (self.config.ple_conv_kernel_size - 1) * self.config.ngram_size
        ple_packed_positions = local_packed_positions(
            cu_seq_lens_q,
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            sp_rank=sp_rank if sp_enabled else 0,
            sp_size=sp_size if sp_enabled else 1,
            device=inputs_embeds.device,
        )
        ple_packed_boundary_indices = local_packed_boundary_indices(
            cu_seq_lens_q,
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            ple_reach,
            sp_rank=sp_rank if sp_enabled else 0,
            sp_size=sp_size if sp_enabled else 1,
            device=inputs_embeds.device,
            packed_seq_lens=packed_seq_lens,
        )

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        # --- Patch.2: Pass the VeOmni attention mask contract to decoder layers. ---
        if sp_enabled:
            raise ValueError("Qwen4-Exp Ulysses/CP requires collator-provided sequence-local position_ids.")
        # --- Patch.2 ---
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

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
            previous_positions = past_key_values.position_ids
            position_ids = torch.cat([previous_positions, position_ids], dim=-1)
        past_key_values.position_ids = position_ids

    # --- Patch.3: Preserve the VeOmni sequence-parallel decoder layout. ---
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        if sp_enabled:
            mask_seq_len = inputs_embeds.shape[1] * sp_size
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
    # --- Patch.3 ---

    conv_mask = causal_mask_mapping.get("linear_attention")
    # --- Patch.4: Run the patched decoder layers and restore the output layout. ---
    if sp_enabled:
        if conv_mask is not None:
            conv_mask = slice_input_tensor(
                conv_mask,
                dim=-1,
                padding=False,
                group=sp_group,
            )
    # --- Patch.4 ---

    # CODEPATH: @ArthurZucker fix flagging for no reason here
    if self.config.ple_layer_ids and conv_mask is not None:
        eos_token_id = self.config.eos_token_id
        eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos_token_id)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)

    # Integer metadata is independent of layer weights and hidden-state values.
    # Keep it local to this forward so checkpoint recomputation can reuse it,
    # without retaining device tensors in the cross-batch host cache.
    qsa_metadata = None
    if not cache_active and "full_attention" in getattr(self.config, "layer_types", ()):
        qsa_metadata = prepare_qsa_metadata(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.config.indexer_compress_ratio,
            sp_rank,
            sp_size,
            inputs_embeds.device,
            cu_seq_lens_q,
            packed_seq_lens,
        )

    # --- Patch.4: Run the patched decoder layers and restore the output layout. ---
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            # --- Patch.1: Disable the upstream attention mask for local decoder input. ---
            attention_mask=None,
            # --- Patch.1 ---
            conv_mask=conv_mask,
            past_key_values=past_key_values,
            ple_input_ids=ple_input_ids,
            packed_seq_lens=packed_seq_lens,
            ple_packed_positions=ple_packed_positions,
            ple_packed_boundary_indices=ple_packed_boundary_indices,
            qsa_metadata=qsa_metadata,
            **kwargs,
        )
    # --- Patch.4 ---

    hidden_states = self.hyper_connection_mixer(hidden_states)

    return Qwen4ExpModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


_QWEN3_5_TO_QWEN4_EXP = {"Qwen3_5": "Qwen4Exp"}

config.override_method(
    "Qwen4ExpVisionModel.rot_pos_emb",
    replacement=qwen3_5_vision_model_rot_pos_emb,
    name_map=_QWEN3_5_TO_QWEN4_EXP,
    description="Reuse the Qwen3.5 host-materialized vision rotary-position path",
)
config.override_method(
    "Qwen4ExpVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_5_vision_model_fast_pos_embed_interpolate,
    name_map=_QWEN3_5_TO_QWEN4_EXP,
    description="Reuse the Qwen3.5 sync-free vision position interpolation path",
)
config.override_method(
    "Qwen4ExpVisionModel.forward",
    replacement=qwen3_5_vision_model_forward,
    name_map=_QWEN3_5_TO_QWEN4_EXP,
    description="Reuse the Qwen3.5 SP-aware local-shard vision forward",
)


@config.override_method(
    "Qwen4ExpVisionAttention.forward",
    description="Run Qwen4-Exp vision attention under a Ulysses SP scope with precomputed max seqlen",
)
def qwen4_exp_vision_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    parallel_state = get_parallel_state()
    if not is_flash_attention_requested(self.config):
        sp_enabled, _, _, _ = _qwen4_exp_sequence_parallel_state(parallel_state)
        if sp_enabled or bool(getattr(parallel_state, "sp_enabled", False)):
            raise NotImplementedError(
                "Qwen4-Exp vision sequence parallelism requires FlashAttention; eager and SDPA "
                "receive local visual-token Q/K/V but global cu_seqlens and cannot segment that layout safely."
            )
    # Modification: prefer the int max_seqlen pre-computed once in
    # Qwen4ExpVisionModel.forward (Patch.5). Fall back to the original
    # GPU-side reduction so this method still works when the model forward
    # has not been patched (e.g. external callers, unit tests).
    max_seqlen = kwargs.pop("vision_max_seqlen", None)
    if max_seqlen is None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    with use_parallel_state(_qwen4_exp_vision_parallel_state(parallel_state)):
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

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


@config.modify_init("Qwen4ExpVisionAttention", description="Bind instance-local rope and attention VeomniOps")
def qwen4_exp_vision_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "full", resolve_op_impl("rotary_pos_emb_vision_implementation"))
    self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)


@config.override_method(
    "Qwen4ExpVisionModel.dummy_forward",
    description="Add a config-derived dummy vision forward for rank-asymmetric FSDP batches",
)
def qwen4_exp_vision_model_dummy_forward(self):
    # --- Patch.1: Reuse VeOmni's SP-aware vision helpers. ---
    # 1. Touch the vision tower on text-only FSDP ranks.
    # 2. Derive shapes and dtype from the live model instead of hardcoding them.
    # 3. Under SP, describe the global grid while supplying only this rank's
    #    local patch rows, and avoid runtime metadata synchronization.
    merge_size = self.spatial_merge_size
    parallel_state = get_parallel_state()
    t = 1
    h = merge_size * parallel_state.sp_size if parallel_state.sp_enabled else merge_size
    w = merge_size
    config = self.config
    flattened_patch_size = config.in_channels * config.temporal_patch_size * config.patch_size**2
    dtype = self.patch_embed.proj.weight.dtype
    device = self.patch_embed.proj.weight.device
    local_patch_count = t * h * w // parallel_state.sp_size
    pixel_values = torch.zeros((local_patch_count, flattened_patch_size), dtype=dtype, device=device)
    grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=device)
    attention_seq_len = t * h * w
    vit_metadata = {
        "grid_thw_list": [[t, h, w]],
        "cu_seqlens": torch.tensor([0, attention_seq_len], dtype=torch.int32, device="cpu"),
        "max_seqlen": attention_seq_len,
    }
    return self(hidden_states=pixel_values, grid_thw=grid_thw, vit_metadata=vit_metadata)
    # --- Patch.1 ---


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
    """Precompute Qwen4-Exp position-layout and ViT varlen metadata on CPU."""
    batch["qwen4_exp_position_ids_layout"] = "batch_first"
    multimodal_metadata = {}
    for modality, grid_key, pixel_key in (
        ("image", "image_grid_thw", "pixel_values"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ):
        grid_thw = batch.get(grid_key)
        if grid_thw is None:
            continue
        grid_thw_list = grid_thw.tolist() if torch.is_tensor(grid_thw) else grid_thw
        if not grid_thw_list:
            continue

        cu_seqlens = [0]
        max_seqlen = 0
        for t, h, w in grid_thw_list:
            frame_seqlen = h * w
            max_seqlen = max(max_seqlen, frame_seqlen)
            for _ in range(t):
                cu_seqlens.append(cu_seqlens[-1] + frame_seqlen)

        padding = sp_pad.get(pixel_key, 0)
        if padding > 0:
            cu_seqlens.append(cu_seqlens[-1] + padding)
            max_seqlen = max(max_seqlen, padding)

        multimodal_metadata[f"{modality}_grid_thw_list"] = grid_thw_list
        multimodal_metadata[f"vit_{modality}_cu_seqlens"] = torch.tensor(
            cu_seqlens,
            dtype=torch.int32,
            device="cpu",
        )
        multimodal_metadata[f"vit_{modality}_max_seqlen"] = max_seqlen

    if multimodal_metadata:
        batch["multimodal_metadata"] = multimodal_metadata


@config.add_helper
class _Qwen4ExpFakeForPositionIds(SimpleNamespace):
    """Picklable minimal receiver for Qwen4ExpModel.get_rope_index."""

    def get_vision_position_ids(self, *args, **kwargs):
        return Qwen4ExpModel.get_vision_position_ids(self, *args, **kwargs)


@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_position_id_func",
    description="Expose a picklable Qwen4-Exp multimodal position-id preprocessor",
)
def qwen4_exp_get_position_id_func_patched(self):
    # --- Patch.1: Expose VeOmni multimodal position-id preprocessing. ---
    fake_config = copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = _Qwen4ExpFakeForPositionIds(config=fake_config)
    return partial(qwen4_exp_get_position_id, Qwen4ExpModel.get_rope_index, fake_model)
    # --- Patch.1 ---


@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_metadata_collate_func",
    description="Expose Qwen4-Exp position-layout and ViT metadata collation",
)
def qwen4_exp_get_metadata_collate_func_patched(self):
    # --- Patch.1 : Expose packed position and ViT metadata collation. ---
    # Qwen4ExpForConditionalGeneration.get_metadata_collate_func
    # Mark VeOmni-packed position ids as batch-first so Model.forward can
    # distinguish them from HF's canonical axis-first layout.
    # Precompute image/video ViT varlen metadata after SP padding so the local
    # VisionModel path does not synchronize GPU tensors back to the host.
    return qwen4_exp_collate_metadata
    # --- Patch.1 ---


@config.override_method(
    "Qwen4ExpForConditionalGeneration.get_parallel_plan",
    description="Register the Qwen4-Exp PLE ExtraParallel plan",
)
def qwen4_exp_get_parallel_plan_patched(self):
    # --- Patch.1: Register the VeOmni PLE ExtraParallel plan. ---
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
    # --- Patch.1 ---


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
    sp_enabled, sp_group, _, sp_size = _qwen4_exp_sequence_parallel_state(parallel_state)
    if sp_enabled and past_key_values is not None:
        raise NotImplementedError("Qwen4-Exp does not support cache prefill or decode under Ulysses/CP.")
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
        "tail_padding_length",
    ):
        if key in kwargs:
            lm_kwargs[key] = kwargs.pop(key)
    multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
    # --- Patch.6: Pass precomputed ViT metadata into the vision model. ---
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
    # --- Patch.6 ---

    if position_ids_layout not in (None, "batch_first"):
        raise ValueError(f"Unsupported Qwen4-Exp position_ids layout: {position_ids_layout!r}")

    if sp_enabled:
        inputs_embeds = gather_outputs(
            inputs_embeds,
            gather_dim=1,
            group=sp_group,
        )
    # if None, calculate mask
    if image_mask is None or video_mask is None:
        mask_input_ids = input_ids
        if sp_enabled and input_ids is not None:
            mask_input_ids = gather_outputs(
                input_ids,
                gather_dim=1,
                group=sp_group,
            )
        fallback_image_mask, fallback_video_mask = self.get_placeholder_mask(mask_input_ids, inputs_embeds)
        image_mask = fallback_image_mask.squeeze(-1) if image_mask is None else image_mask
        video_mask = fallback_video_mask.squeeze(-1) if video_mask is None else video_mask
    image_mask = image_mask.bool()
    video_mask = video_mask.bool()

    if pixel_values is not None:
        # --- Patch.6: Run the patched vision model and scatter image features. ---
        with use_parallel_state(_qwen4_exp_vision_parallel_state(parallel_state)):
            image_outputs: BaseModelOutputWithPooling = self.visual(
                pixel_values.type(self.visual.dtype),
                grid_thw=image_grid_thw,
                return_dict=True,
                **image_vit_kwargs,
            )
        image_embeds = image_outputs.pooler_output
        if sp_enabled:
            image_embeds = gather_outputs(
                image_embeds,
                gather_dim=0,
                group=sp_group,
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_embeds)
        # --- Patch.6 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3: Touch the vision path for FSDP when images are absent. ---
        with use_parallel_state(_qwen4_exp_vision_parallel_state(parallel_state)):
            fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---

    if pixel_values_videos is not None:
        # --- Patch.6: Run the patched vision model and scatter video features. ---
        with use_parallel_state(_qwen4_exp_vision_parallel_state(parallel_state)):
            video_outputs: BaseModelOutputWithPooling = self.visual(
                pixel_values_videos.type(self.visual.dtype),
                grid_thw=video_grid_thw,
                return_dict=True,
                **video_vit_kwargs,
            )
        video_embeds = video_outputs.pooler_output
        if sp_enabled:
            video_embeds = gather_outputs(
                video_embeds,
                gather_dim=0,
                group=sp_group,
            )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask.unsqueeze(-1), video_embeds)
        # --- Patch.6 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.3: Touch the vision path for FSDP when videos are absent. ---
        with use_parallel_state(_qwen4_exp_vision_parallel_state(parallel_state)):
            fake_embeds = self.visual.dummy_forward().pooler_output.mean() * 0.0
        inputs_embeds = inputs_embeds + fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # --- Patch.3 ---

    if sp_enabled:
        inputs_embeds = slice_input_tensor(
            inputs_embeds,
            dim=1,
            padding=False,
            group=sp_group,
        )
        image_mask = slice_input_tensor(
            image_mask,
            dim=1,
            padding=False,
            group=sp_group,
        )
        video_mask = slice_input_tensor(
            video_mask,
            dim=1,
            padding=False,
            group=sp_group,
        )

    ple_input_ids = None
    # Multimodal preprocessing previously zeroed out the image/video placeholders;
    # PLE needs to restore the real model tokens.
    if local_ple_input_ids is not None:
        ple_input_ids = local_ple_input_ids
        ple_input_ids.masked_fill_(image_mask, self.config.image_token_id)
        ple_input_ids.masked_fill_(video_mask, self.config.video_token_id)

    if position_ids is None:
        if sp_enabled:
            raise ValueError("Qwen4-Exp Ulysses/CP requires precomputed position_ids from the data collator.")
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
    # --- Patch.5: Accept VeOmni batch-first precomputed position ids. ---
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

    cu_seq_lens_q = lm_kwargs.get("cu_seq_lens_q")
    if cu_seq_lens_q is None:
        if sp_enabled:
            raise ValueError("Qwen4-Exp Ulysses/CP requires precomputed cu_seq_lens_q from the data collator.")
        cu_seq_lens_q = pos2culen(position_ids[0])
        lm_kwargs["cu_seq_lens_q"] = cu_seq_lens_q
    global_sequence_length = inputs_embeds.shape[1] * sp_size
    # --- Patch.7: Export host-side packed lengths once per forward. ---
    # Every GDN/QSA/PLE layer needs the same segment boundaries. The trainer
    # moves cu_seq_lens_q onto the device, so deriving lengths per layer would
    # sync GPU->CPU once per layer; derive them here once and thread the list
    # through lm_kwargs for all layers to reuse.
    packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
        cu_seq_lens_q.diff().tolist(), inputs_embeds.shape[0], global_sequence_length
    )
    lm_kwargs["packed_seq_lens"] = packed_seq_lens
    # --- Patch.7 ---
    # the final shape passed to language_model is [4, B, S_local]
    if position_ids.shape[0] == 3:
        # --- Patch.7: Rebuild text positions from the host list; culen2pos
        # would sync again through its internal .cpu() call. ---
        text_position_ids = (
            torch.cat([torch.arange(length, device=position_ids.device) for length in packed_seq_lens])
            .unsqueeze(0)
            .to(dtype=position_ids.dtype)
        )
        # --- Patch.7 ---
        if sp_enabled:
            text_position_ids = slice_input_tensor(
                text_position_ids,
                dim=-1,
                padding=False,
                group=sp_group,
            )
        if tuple(text_position_ids.shape) != tuple(position_ids.shape[1:]):
            raise ValueError(
                "Qwen4-Exp text position ids must have shape (batch, sequence) matching the M-RoPE positions; "
                f"got {tuple(text_position_ids.shape)} and {tuple(position_ids.shape[1:])}."
            )
        position_ids = torch.cat((text_position_ids.unsqueeze(0), position_ids), dim=0)

    kwargs.update(lm_kwargs)
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


@config.add_helper_after("Qwen4ExpCausalLMOutputWithPast")
@dataclass
class Qwen4ExpCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen4ExpCausalLMOutputWithPast):
    """Qwen4-Exp output extended with VeOmni fused-loss auxiliary tensors.

    Args:
        fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
            Per-token values produced by VeOmni's fused-linear loss path.
    """


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.__init__
# 1. Bind instance-local CE and load-balancing VeomniOps used by VLM SFT.
# ================================================================
@config.override_method(
    "Qwen4ExpForConditionalGeneration.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniOps",
)
def qwen4_exp_for_conditional_generation_init_patched(self, config):
    super().__init__(config)
    self.model = Qwen4ExpModel(config)
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    impl = resolve_op_impl("cross_entropy_loss_implementation")
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.veomni_lb = VeomniOp(
        "load_balancing_loss",
        "standard",
        resolve_op_impl("load_balancing_loss_implementation"),
    )
    self.load_balancing_loss = partial(load_balancing_loss, op=self.veomni_lb)
    self.post_init()


@config.override_method(
    "Qwen4ExpForConditionalGeneration.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniOps",
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
    # --- Patch.1: Forward the packed position-id layout to the VLM model. ---
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
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1: Use the VeOmni fused-linear-compatible loss path. ---
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
    # --- Patch.1 ---

    # --- Patch.2: Add the sequence-local router auxiliary loss. ---
    # Preserve Qwen4 MoE router auxiliary loss without enabling MTP loss, using
    # the sequence-local padding mask that matches local router logits under
    # Ulysses, consistent with Qwen3.5-MoE.
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        router_attention_mask = attention_mask
        parallel_state = get_parallel_state()
        sp_enabled, sp_group, _, _ = _qwen4_exp_sequence_parallel_state(parallel_state)
        if (
            sp_enabled
            and router_attention_mask is not None
            and router_attention_mask.shape[-1] != outputs.last_hidden_state.shape[1]
        ):
            router_attention_mask = slice_input_tensor(
                router_attention_mask,
                dim=-1,
                padding=False,
                group=sp_group,
            )
        aux_loss = self.load_balancing_loss(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            router_attention_mask,
        )
        if isinstance(loss, torch.Tensor) and isinstance(aux_loss, torch.Tensor):
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


# ================================================================
# Patch: Qwen4ExpTextSparseMoeBlock.forward (MoE router replay)
# 1. When an RL framework has installed a
# replay manager via ``set_active_replay``, the manager may substitute
# ``selected_experts`` with previously recorded target indices. The
# manager's sole responsibility is choosing indices; all model-specific
# post-topk weight math (softmax recompute, gather, renorm, dtype cast)
# is replicated here so the cross-framework controller stays
# model-agnostic.
# ================================================================
@config.override_method(
    "Qwen4ExpTextSparseMoeBlock.forward",
    description="Call maybe_replay_indices for RL router replay",
)
def qwen4_exp_text_sparse_moe_block_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Replay expert indices while preserving native top-k weighting and local expert ops."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
    # --- Patch.1: Recompute weights for replayed indices from raw router logits. ---
    # Match the native router's optional normalization and output dtype.
    if get_active_replay() is not None:
        target_dtype = routing_weights.dtype
        routing_scores = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        selected_experts = maybe_replay_indices(self.gate, routing_scores, selected_experts)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.gate.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(target_dtype)
    # --- Patch.1 ---
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

    expert_output = expert_output + shared_expert_output
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
    return expert_output
