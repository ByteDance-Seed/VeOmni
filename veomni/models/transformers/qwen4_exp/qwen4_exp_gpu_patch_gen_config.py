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

from copy import copy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpTextModel,
    Qwen4ExpVisionModel,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, torch_compilable_check

from veomni.distributed.moe.comm import all_to_all
from veomni.distributed.parallel_state import get_parallel_state
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin


config = PatchConfig(
    source_module="transformers.models.qwen4_exp.modeling_qwen4_exp",
    target_file="patched_modeling_qwen4_exp_gpu.py",
    description="Qwen4-Exp initial GPU VLM-SFT integration with explicit PLE/QSA limits",
)

config.add_import("copy", names=["copy"])
config.add_import("dataclasses", names=["dataclass"])
config.add_import("functools", names=["partial"])
config.add_import("types", names=["SimpleNamespace"])
config.add_import("torch.distributed", alias="dist", is_from_import=False)
config.add_import("veomni.distributed.moe.comm", names=["all_to_all"])
config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.add_import("veomni.utils.model_outputs", names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin"])
config.add_post_import_block(
    """
    # Bound by ``_bind_veomni_ops`` before model construction. Qwen4-Exp
    # currently uses the upstream eager/SDPA QSA implementation; only the
    # expert and loss paths opt into VeOmni kernels in this initial port.
    from veomni.ops.dispatch import OpSlot
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    """
)


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
    ) -> torch.Tensor:
        input_ids = input_ids.long()
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
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
# Patch: Qwen4ExpTextPLELayer.forward
# 1. Keep FP32 PLE master weights while casting sparse lookup results to the
#    activation dtype before the result all-to-all and downstream projections.
# ================================================================
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
) -> torch.Tensor:
    # --- Patch.1 ---
    embeddings = self.ple_embedding(input_ids, past_key_values, output_dtype=hidden_states.dtype)
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
    output = gated_value + self._short_conv(gated_value_normed, past_key_values)
    return output


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
    """Mark the packed position-id layout without relying on ambiguous tensor shapes."""
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

    if position_ids is None:
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
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
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
