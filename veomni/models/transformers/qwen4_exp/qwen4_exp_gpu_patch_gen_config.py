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
"""Patch configuration for the initial Qwen4-Exp GPU integration.

Regen command:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config -o veomni/models/transformers/qwen4_exp/generated --diff

This first integration targets VLM SFT with sequence parallelism disabled. It
keeps the upstream eager/SDPA QSA implementation for correctness and makes the
unsupported Ulysses path fail explicitly. MTP is intentionally outside the
training model and is filtered by ``checkpoint_tensor_converter.py``.
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
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpTextModel,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpVisionModel,
    apply_mask_to_padding_states,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, torch_compilable_check

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.loss_utils import ForCausalLMLoss, load_balancing_loss
from veomni.models.utils.moe_utils import merged_experts_act_fn_forward
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin
from veomni.utils.seqlen_pos_transform_utils import culen2pos, pos2culen


config = PatchConfig(
    source_module="transformers.models.qwen4_exp.modeling_qwen4_exp",
    target_file="patched_modeling_qwen4_exp_gpu.py",
    description="Qwen4-Exp initial GPU VLM-SFT integration with explicit PLE/QSA limits",
)
config.exclude_from_output("apply_rotary_pos_emb", "apply_rotary_pos_emb_vision", "rotate_half")


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
    description="Call swiglu_mlp for silu/swish, otherwise self.act_fn",
)
def qwen4_exp_text_mlp_forward_patched(self, x):
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


config.add_import("copy", names=["copy"])
config.add_import("dataclasses", names=["dataclass"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.moe.comm", names=["all_to_all"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.add_import("veomni.utils.model_outputs", names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"])
config.add_import("veomni.utils.seqlen_pos_transform_utils", names=["culen2pos", "pos2culen"])
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import(
    "veomni.ops.config",
    names=["resolve_op_impl"],
)
config.add_import(
    "veomni.models.utils.moe_utils",
    names=["merged_experts_act_fn_forward"],
)
config.add_import(
    "veomni.models.loss_utils",
    names=["ForCausalLMLoss", "load_balancing_loss"],
)


# ``all_to_all`` is imported into the generated file; this sentinel exists
# so the PLE replacement class can mention it while the patch config loads.
all_to_all = None


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
# ================================================================
@config.override_method(
    "Qwen4ExpModel.__init__",
    description="Build local patched submodels instead of upstream AutoModel classes",
)
def qwen4_exp_model_init_patched(self, config):
    super().__init__(config)
    # --- Patch.1 ---
    self.visual = Qwen4ExpVisionModel._from_config(config.vision_config)
    self.language_model = Qwen4ExpTextModel._from_config(config.text_config)
    # --- Patch.1 ---
    self.rope_deltas = None
    self.post_init()


# ================================================================
# Helpers: packed-sequence boundaries
# ================================================================
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


# ================================================================
# Patch: Qwen4ExpTextGatedDeltaNet
# 1. Use the bound varlen kernels for both single- and multi-segment batches.
# 2. Run eager fallbacks independently per segment when fused kernels are absent.
# 3. Require VeOmni packing metadata and reject recurrent cache state.
# ================================================================
@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.__init__",
    description="Bind instance-local GDN VeomniOps used by packed varlen training",
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


@config.override_method(
    "Qwen4ExpTextGatedDeltaNet.forward",
    description="Reset Qwen4-Exp GDN convolution and recurrent state at packed boundaries",
)
def qwen4_exp_gated_deltanet_forward_patched(
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

    conv_weight = self.conv1d.weight.squeeze(1)
    if self.veomni_causal_conv1d.impl == "eager":
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
            linear_attn_cu_seq_lens_q,
            activation=self.activation,
            seq_idx=None,
            backend="triton",
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

    if self.veomni_chunk_gated_delta_rule.impl == "eager":
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
    else:
        core_attn_out, _ = self.veomni_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            None,
            linear_attn_cu_seq_lens_q,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


# ================================================================
# Patch: Qwen4ExpTextExperts
# 1. Drop HF's use_experts_implementation decorator so VeOmni owns dispatch.
# 2. Retain the upstream fused checkpoint layout and eager implementation.
# ================================================================
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
# 6. Require explicit packed boundaries and reset n-gram history for every
#    segment; recurrent-cache and non-varlen paths are intentionally unsupported.
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
        if cu_seq_lens_q is None:
            raise ValueError("Qwen4-Exp PLE n-gram history requires cu_seq_lens_q for VeOmni varlen training.")
        packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
            cu_seq_lens_q.diff().tolist(), input_ids.shape[0], input_ids.shape[1]
        )
        # --- Patch.6 ---
        if past_key_values is not None:
            raise ValueError("Qwen4-Exp varlen PLE n-gram history does not support cache state.")
        input_segments = input_ids.split(packed_seq_lens, dim=1)
        shifted_tokens = [
            torch.cat([self._shift_right_ignore_eos(segment, shift) for segment in input_segments], dim=1)
            for shift in range(self.ngram_size)
        ]
        # --- Patch.6 ---
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

        ngram_ids = torch.cat(blocks, dim=-1)
        original_shape = ngram_ids.shape
        flat_ids = ngram_ids.reshape(-1)
        shard_ids = torch.div(flat_ids, self.rows_per_checkpoint_shard, rounding_mode="floor")
        row_ids = torch.remainder(flat_ids, self.rows_per_checkpoint_shard)
        # --- Patch.5 ---
        embeddings = self._distributed_lookup(shard_ids, row_ids, output_dtype=output_dtype)
        # --- Patch.5 ---
        return embeddings.view(*original_shape, -1).flatten(-2)


# ================================================================
# Patch: Qwen4ExpTextPLELayer._short_conv / forward
# 1. Require explicit packed boundaries and reset the dilated convolution for
#    every segment, including single-segment varlen batches.
# 2. Keep FP32 PLE master weights while casting sparse lookup results to the
#    activation dtype before the result all-to-all and downstream projections.
# 3. Reject recurrent cache state; this modeling is training-only.
# ================================================================
@config.override_method(
    "Qwen4ExpTextPLELayer._short_conv",
    description="Reset the Qwen4-Exp PLE dilated convolution at packed boundaries",
)
def qwen4_exp_text_ple_layer_short_conv_patched(
    self,
    hidden_states: torch.Tensor,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seq_lens_q is None:
        raise ValueError("Qwen4-Exp PLE convolution requires cu_seq_lens_q for VeOmni varlen training.")
    packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
        cu_seq_lens_q.diff().tolist(), hidden_states.shape[0], hidden_states.shape[1]
    )
    # --- Patch.1 / Patch.3 ---
    if past_key_values is not None:
        raise ValueError("Qwen4-Exp varlen PLE convolution does not support cache state.")
    outputs = []
    for segment in hidden_states.split(packed_seq_lens, dim=1):
        conv_input = F.pad(segment.transpose(1, 2), (self.short_conv_state_len, 0))
        outputs.append(F.silu(self.conv1d(conv_input)).transpose(1, 2))
    return torch.cat(outputs, dim=1)
    # --- Patch.1 / Patch.3 ---


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


# ================================================================
# Patch: Qwen4ExpTextDecoderLayer.forward
# 1. Thread the collator's packed boundaries into both stateful token mixers.
# ================================================================
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
def qwen4_exp_collate_metadata(batch, _sp_pad):
    """Mark the packed position-id layout without adding model-specific boundaries."""
    batch["qwen4_exp_position_ids_layout"] = "batch_first"


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
# 4. Reject SP until PLE context and QSA global-index semantics are implemented.
# 5. Accept VeOmni's batch-first precomputed M-RoPE layout.
# 6. Infer varlen boundaries from position_ids when absent, reject cache paths,
#    and prepend scalar text positions so three-axis M-RoPE produces a packed
#    causal mask inside Qwen4ExpTextModel.
# ================================================================
@config.override_method(
    "Qwen4ExpModel.forward",
    description="Support VeOmni VLM SFT masks and PLE ids, with an explicit SP guard",
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
    # --- Patch.4 ---
    if get_parallel_state().sp_enabled:
        raise NotImplementedError(
            "Qwen4-Exp VLM SFT currently requires ulysses_size=1 and cp_size=1. "
            "PLE n-gram context and QSA token selection are not yet sequence-parallel safe."
        )
    # --- Patch.4 ---

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.1 ---
    image_mask = kwargs.pop("image_mask", None)
    video_mask = kwargs.pop("video_mask", None)
    position_ids_layout = kwargs.pop("qwen4_exp_position_ids_layout", None)
    if position_ids_layout not in (None, "batch_first"):
        raise ValueError(f"Unsupported Qwen4-Exp position_ids layout: {position_ids_layout!r}")
    if position_ids is None:
        raise ValueError("Qwen4-Exp VeOmni training requires precomputed position_ids.")
    if past_key_values is not None:
        raise ValueError("Qwen4-Exp VeOmni varlen training does not support cache state.")
    if image_mask is None or video_mask is None:
        fallback_image_mask, fallback_video_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        image_mask = fallback_image_mask.squeeze(-1) if image_mask is None else image_mask
        video_mask = fallback_video_mask.squeeze(-1) if video_mask is None else video_mask
    image_mask = image_mask.bool()
    video_mask = video_mask.bool()
    # The initial port does not consume collator-side ViT metadata yet.
    kwargs.pop("multimodal_metadata", None)
    # --- Patch.1 ---

    # --- Patch.2 ---
    ple_input_ids = None
    if self.config.text_config.ple_layer_ids:
        if input_ids is None:
            ple_input_ids = self.language_model.reverse_embedding(inputs_embeds)
        else:
            ple_input_ids = input_ids.clone()
            ple_input_ids.masked_fill_(image_mask, self.config.image_token_id)
            ple_input_ids.masked_fill_(video_mask, self.config.video_token_id)
    # --- Patch.2 ---

    if pixel_values is not None:
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

    # --- Patch.5 ---
    if position_ids_layout == "batch_first":
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

    # --- Patch.6 ---
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    if cu_seq_lens_q is None:
        cu_seq_lens_q = pos2culen(position_ids[0])
        kwargs["cu_seq_lens_q"] = cu_seq_lens_q
    kwargs.setdefault("linear_attn_cu_seq_lens_q", cu_seq_lens_q)
    _qwen4_exp_validate_packed_seq_lens(cu_seq_lens_q.diff().tolist(), inputs_embeds.shape[0], inputs_embeds.shape[1])
    if position_ids.shape[0] == 3:
        text_position_ids = culen2pos(cu_seq_lens_q)
        text_position_ids = text_position_ids.to(device=position_ids.device, dtype=position_ids.dtype)
        if tuple(text_position_ids.shape) != tuple(position_ids.shape[1:]):
            raise ValueError(
                "Qwen4-Exp text position ids must have shape (batch, sequence) matching the M-RoPE positions; "
                f"got {tuple(text_position_ids.shape)} and {tuple(position_ids.shape[1:])}."
            )
        position_ids = torch.cat((text_position_ids.unsqueeze(0), position_ids), dim=0)

    # --- Patch.6 ---
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=None,
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
    impl = resolve_op_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.veomni_lb = VeomniOp(
        "load_balancing_loss",
        "standard",
        resolve_op_impl("load_balancing_loss_implementation"),
    )
    self.load_balancing_loss = partial(load_balancing_loss, op=self.veomni_lb)
    self.post_init()


# ================================================================
# Patch: Qwen4ExpForConditionalGeneration.forward
# 1. Always call the bound ForCausalLMLoss VeomniOp and keep model-only
#    metadata out of loss kwargs.
# 2. Preserve Qwen4 MoE router auxiliary loss without enabling MTP loss.
# ================================================================
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
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    # --- Patch.1 ---
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

    # --- Patch.2 ---
    aux_loss = None
    if kwargs.get("output_router_logits", False):
        aux_loss = self.load_balancing_loss(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
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


# Names resolved at codegen time from generated imports.
is_flash_attention_requested = None
get_max_seqlen = None


@config.modify_init("Qwen4ExpTextQSAIndexer", description="Bind instance-local partial rope VeomniOp")
def qwen4_exp_text_qsa_indexer_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))


@config.override_method(
    "Qwen4ExpTextQSAIndexer.forward",
    description="Always call the local partial rope VeomniOp for single-tensor indexer RoPE",
)
def qwen4_exp_text_qsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Cache | None,
) -> torch.Tensor:
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q, _ = self.veomni_rope(q, q, current_cos, current_sin, unsqueeze_dim=2)

    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

    selected_token_indices = torch.full(
        (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            local_visible_indices = torch.nonzero(
                visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
            ).flatten()
            num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
            if num_complete_blocks > 0:
                block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                    num_complete_blocks, self.compress_ratio
                )

                key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                pooled_keys = self.k_layernorm(pooled_keys)
                group_starts = block_token_indices[:, 0]
                pooled_keys = pooled_keys.unsqueeze(1)
                block_key_states, _ = self.veomni_rope(
                    pooled_keys,
                    pooled_keys,
                    full_cos[batch_idx].index_select(0, group_starts),
                    full_sin[batch_idx].index_select(0, group_starts),
                )
                block_key_states = block_key_states.squeeze(1)

                scores = torch.matmul(
                    q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
            else:
                selected_tokens = torch.tensor([], device=hidden_states.device)
            tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
    )
    scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

    return selected_token_mask


@config.modify_init("Qwen4ExpTextAttention", description="Bind instance-local rope and attention VeomniOps")
def qwen4_exp_text_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "partial", resolve_op_impl("rotary_pos_emb_implementation"))
    self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)


@config.override_method(
    "Qwen4ExpTextAttention.forward",
    description="Always call the local rope and attention VeomniOps",
)
def qwen4_exp_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
    if attention_mask.is_floating_point():
        attention_mask = attention_mask + selected_token_mask
    else:
        attention_mask = attention_mask & selected_token_mask

    position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@config.modify_init("Qwen4ExpVisionAttention", description="Bind instance-local rope and attention VeomniOps")
def qwen4_exp_vision_attention_bind_ops(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "full", resolve_op_impl("rotary_pos_emb_vision_implementation"))
    self.veomni_attn = VeomniOp("attention", "standard", self.config._attn_implementation)


@config.override_method(
    "Qwen4ExpVisionAttention.forward",
    description="Always call the local attention VeomniOp",
)
def qwen4_exp_vision_attention_forward_patched(
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
    cos, sin = position_embeddings
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = self.veomni_attn

    if is_flash_attention_requested(self.config):
        max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
        attn_output, _ = attention_interface(
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
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output
