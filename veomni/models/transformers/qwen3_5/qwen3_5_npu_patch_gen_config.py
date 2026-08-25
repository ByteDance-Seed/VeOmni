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
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    apply_mask_to_padding_states,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    _Qwen3_5FakeForPosID,
    collate_multimodal_metadata,
    get_position_id,
    mm_token_type_ids_from_input_ids,
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
    qwen3_5_vision_model_rot_pos_emb,
)
from veomni.ops.kernels.attention._replicated_dummy import (
    is_replicated_dummy_sequence_parallel,
    reject_public_sequence_parallel_bypass,
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
config.add_import(
    "veomni.ops.kernels.attention._replicated_dummy",
    names=[
        "_DUMMY_SP_TOKEN",
        "_call_replicated_dummy_checkpointed_module",
        "_replicated_dummy_sequence_parallel",
        "is_replicated_dummy_sequence_parallel",
        "reject_public_sequence_parallel_bypass",
    ],
)
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_seq_scatter_heads_grouped", "gather_heads_scatter_seq"],
)
config.add_import(
    "veomni.distributed.context_parallel.gdn_headwise",
    names=[
        "compile_gdn_headwise_layout",
        "prepare_gdn_headwise_inputs",
        "restore_gdn_headwise_output",
    ],
)
config.add_import(
    "veomni.distributed.context_parallel.gdn_lossless",
    names=[
        "align_gdn_varlen_chunks",
        "aligned_gdn_cu_seqlens",
        "attach_state_dependency",
        "compile_gdn_lossless_runtime_plan",
        "exchange_conv_halo",
        "make_state_participation",
        "make_state_template",
        "owned_to_physical",
        "physical_to_owned_grouped",
        "receive_initial_state",
        "send_final_state",
        "trim_conv_halo",
        "unpad_gdn_varlen_output",
    ],
)
config.add_import(
    "veomni.distributed.context_parallel.gdn_kcp",
    names=[
        "get_kcp_affine_backend_identity",
        "kcp_plan_requires_affine_scan",
        "prepare_kcp_affine_summary",
        "resolve_kcp_initial_state",
    ],
)
config.add_import("veomni.distributed.context_parallel.gdn_runtime", names=["make_gdn_cp_runtime_observer"])
config.add_import(
    "veomni.ops.kernels.gated_delta_rule.backend_adapter",
    names=[
        "call_chunk_gated_delta_rule",
        "prepare_gated_delta_rule_qk",
        "requires_chunked_varlen_metadata",
        "resolve_kcp_affine_implementation",
    ],
)
config.add_import(
    "veomni.distributed.context_parallel.packed_sharding",
    names=[
        "reorder_sample_major_to_ulysses_rank_major",
        "reorder_ulysses_rank_major_to_sample_major",
        "ulysses_local_head_count",
    ],
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
    from veomni.ops.dispatch import OpSlot, OpsConfigSlot
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "partial")
    veomni_apply_rotary_pos_emb_vision = OpSlot("rotary_pos_emb_vision", "full")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_rms_norm_gated = OpSlot("rms_norm_gated", "standard")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")
    veomni_gdn_context_parallel_implementation = OpsConfigSlot(
        "gdn_context_parallel_implementation", "disabled"
    )
    """
)


# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
torch_chunk_gated_delta_rule = None  # noqa: F811 — also imported above for the forward patch
gather_seq_scatter_heads = None
gather_seq_scatter_heads_grouped = None
gather_heads_scatter_seq = None
gather_outputs = None
slice_input_tensor = None
sp_pad_and_slice = None
veomni_rms_norm_gated = None  # OpSlot, declared in post-import block above
veomni_causal_conv1d = None  # OpSlot, declared in post-import block above
veomni_chunk_gated_delta_rule = None  # OpSlot, declared in post-import block above
veomni_gdn_context_parallel_implementation = None  # OpsConfigSlot, declared in post-import block above
# Names referenced by the patched Qwen3_5TextModel.forward; resolved at
# codegen time from the imports already present in the generated modeling file.
DynamicCache = None
create_causal_mask = None
Qwen3_5ModelOutputWithPast = None

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
    cu_seqlens_list: list[int] | None = None,
    chunk_indices: dict | None = None,
    chunk_indices_list: dict | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

    # Modification: hardware-neutral lossless GDN CP.  The generic ownership
    # and communication layer deliberately has no torch_npu import; only this
    # NPU patch converts kernel metadata to the device-specific representation.
    parallel_state = get_parallel_state()
    cp_enabled = parallel_state.cp_enabled
    headwise_enabled = self.gdn_context_parallel_implementation == "headwise_lossless"
    if cp_enabled and self.gdn_context_parallel_implementation not in (
        "state_passing_lossless",
        "kcp",
        "headwise_lossless",
    ):
        raise RuntimeError(
            "GDN context parallelism requires gdn_context_parallel_implementation to be "
            "'state_passing_lossless', 'kcp', or 'headwise_lossless'."
        )
    if not cp_enabled and self.gdn_context_parallel_implementation != "disabled":
        raise RuntimeError("The selected GDN CP implementation requires an initialized context-parallel group.")
    gdn_lossless_plan = None
    gdn_headwise_layout = None
    gdn_cp_observer = None
    ulysses_local_cu = None
    backend_impl = self._veomni_chunk_gated_delta_rule_impl
    kcp_affine_impl = None
    kcp_affine_backend = None
    if self.gdn_context_parallel_implementation == "kcp":
        kcp_affine_impl = resolve_kcp_affine_implementation(backend_impl)
        kcp_affine_backend = get_kcp_affine_backend_identity(kcp_affine_impl)
    if cp_enabled:
        if batch_size != 1 or cu_seq_lens_q is None:
            raise RuntimeError("Lossless GDN CP requires a packed batch of size one and global valid cu_seqlens.")
        if cache_params is not None:
            raise NotImplementedError("Lossless GDN CP currently supports training forwards without KV cache only.")
        if cu_seqlens_list is None:
            raise RuntimeError(
                "Lossless GDN CP requires host linear_attn_cu_seqlens_list_q; "
                "the data collator must materialize it before device transfer."
            )
        valid_points = [int(point) for point in cu_seqlens_list]
        valid_lengths = [end - start for start, end in zip(valid_points, valid_points[1:])]
        if not headwise_enabled:
            plan_key = (tuple(valid_lengths), parallel_state.cp_size, parallel_state.ulysses_size)
            cached_plan = getattr(self, "_gdn_lossless_plan_cache", None)
            if cached_plan is None or cached_plan[0] != plan_key:
                cached_plan = (
                    plan_key,
                    compile_gdn_lossless_runtime_plan(
                        valid_lengths,
                        cp_group=parallel_state.cp_group,
                        ulysses_size=parallel_state.ulysses_size,
                    ),
                )
                self._gdn_lossless_plan_cache = cached_plan
            gdn_lossless_plan = cached_plan[1]
            observer = getattr(self, "gdn_cp_runtime_evidence", None)
            expected_identity = (
                self.gdn_context_parallel_implementation,
                gdn_lossless_plan.plan_hash,
                gdn_lossless_plan.cp_size,
                gdn_lossless_plan.cp_rank,
                kcp_affine_backend,
            )
            live_identity = None
            if observer is not None:
                identity = observer.identity
                live_identity = (
                    identity.implementation,
                    identity.ownership_plan_hash,
                    identity.cp_size,
                    identity.cp_rank,
                    identity.affine_backend,
                )
            if live_identity != expected_identity:
                observer = make_gdn_cp_runtime_observer(
                    self.gdn_context_parallel_implementation,
                    plan=gdn_lossless_plan,
                    affine_backend=kcp_affine_backend,
                )
                self.gdn_cp_runtime_evidence = observer
            gdn_cp_observer = observer
            local_physical_lengths = [
                length // (parallel_state.cp_size * parallel_state.ulysses_size)
                for length in gdn_lossless_plan.global_plan.ring_physical_lengths
            ]
            ulysses_local_cu = [0]
            for length in local_physical_lengths:
                ulysses_local_cu.append(ulysses_local_cu[-1] + length)

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
    ulysses_enabled = parallel_state.ulysses_enabled
    head_parallel_rank = parallel_state.ulysses_rank if ulysses_enabled else 0
    if headwise_enabled:
        sp_group = parallel_state.sp_group
        sp_size = parallel_state.sp_size
        head_parallel_rank = parallel_state.sp_rank
        if sp_group is None or head_parallel_rank < 0:
            raise RuntimeError("headwise_lossless requires the flattened CP x Ulysses process group")
        q_proj, k_proj, v_proj = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q_proj = q_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k_proj = k_proj.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v_proj = v_proj.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        headwise_inputs = (q_proj, k_proj, v_proj, b, a)
        headwise_plan_key = (
            tuple(valid_points),
            parallel_state.cp_size,
            sp_size,
            tuple(tuple(tensor.shape) for tensor in headwise_inputs),
            str(mixed_qkv.dtype),
            mixed_qkv.device.type,
        )
        cached_headwise_layout = getattr(self, "_gdn_headwise_layout_cache", None)
        if cached_headwise_layout is None or cached_headwise_layout[0] != headwise_plan_key:
            cached_headwise_layout = (
                headwise_plan_key,
                compile_gdn_headwise_layout(
                    headwise_inputs,
                    cu_seqlens=valid_points,
                    group=sp_group,
                    cp_size=parallel_state.cp_size,
                ),
            )
            self._gdn_headwise_layout_cache = cached_headwise_layout
        gdn_headwise_layout = cached_headwise_layout[1]
        expected_sp_size = parallel_state.cp_size * parallel_state.ulysses_size
        expected_sp_rank = parallel_state.cp_rank * parallel_state.ulysses_size + (
            parallel_state.ulysses_rank if parallel_state.ulysses_enabled else 0
        )
        if (
            sp_size != expected_sp_size
            or gdn_headwise_layout.world_size != expected_sp_size
            or head_parallel_rank != expected_sp_rank
            or gdn_headwise_layout.rank != expected_sp_rank
        ):
            raise RuntimeError(
                "headwise_lossless requires CP-major/Ulysses-inner flattened SP rank order: "
                f"sp_size={sp_size}, layout_world={gdn_headwise_layout.world_size}, "
                f"sp_rank={head_parallel_rank}, layout_rank={gdn_headwise_layout.rank}, "
                f"expected_size={expected_sp_size}, expected_rank={expected_sp_rank}"
            )
        (q_proj, k_proj, v_proj, b, a), _ = prepare_gdn_headwise_inputs(
            headwise_inputs,
            group=sp_group,
            layout=gdn_headwise_layout,
        )
        local_num_k_heads = self.num_k_heads // sp_size
        local_num_v_heads = self.num_v_heads // sp_size
        local_key_dim = self.head_k_dim * local_num_k_heads
        local_value_dim = self.head_v_dim * local_num_v_heads
        mixed_qkv = torch.cat(
            (
                q_proj.reshape(batch_size, gdn_headwise_layout.total_valid_tokens, -1),
                k_proj.reshape(batch_size, gdn_headwise_layout.total_valid_tokens, -1),
                v_proj.reshape(batch_size, gdn_headwise_layout.total_valid_tokens, -1),
            ),
            dim=-1,
        )
        if not getattr(self, "_gdn_headwise_runtime_logged", False):
            logger.info(
                "VEOMNI_GDN_CP_RUNTIME impl=headwise_lossless "
                f"sp_size={sp_size} packed_a2a_single=true cp_size={parallel_state.cp_size}"
            )
            self._gdn_headwise_runtime_logged = True
    elif ulysses_enabled:
        ulysses_group = parallel_state.ulysses_group
        ulysses_size = parallel_state.ulysses_size
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

        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)

        # KCP keeps its ownership/state algorithm unchanged, but transports
        # all five projections in one dense all_to_all_single. This preserves
        # the independent Ulysses tensor layout and wire bytes while removing
        # four collective launches in both forward and backward.
        if self.gdn_context_parallel_implementation == "kcp":
            q_proj, k_proj, v_proj, b, a = gather_seq_scatter_heads_grouped(
                (q_proj, k_proj, v_proj, b, a),
                seq_dim=1,
                head_dim=2,
                group=ulysses_group,
            )
            if not getattr(self, "_gdn_kcp_grouped_ulysses_logged", False):
                logger.info(
                    "VEOMNI_GDN_KCP_RUNTIME grouped_ulysses_a2a=true "
                    f"ulysses_size={ulysses_size} cp_size={parallel_state.cp_size}"
                )
                self._gdn_kcp_grouped_ulysses_logged = True
        else:
            # All-to-all: gather full sequence, scatter heads ->
            # [B, S_full, local_heads, head_dim].
            q_proj = gather_seq_scatter_heads(q_proj, seq_dim=1, head_dim=2, group=ulysses_group)
            k_proj = gather_seq_scatter_heads(k_proj, seq_dim=1, head_dim=2, group=ulysses_group)
            v_proj = gather_seq_scatter_heads(v_proj, seq_dim=1, head_dim=2, group=ulysses_group)
            b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2, group=ulysses_group)
            a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2, group=ulysses_group)

        if cp_enabled:
            q_proj = reorder_ulysses_rank_major_to_sample_major(
                q_proj, ulysses_local_cu, ulysses_size=ulysses_size, sequence_dim=1
            )
            k_proj = reorder_ulysses_rank_major_to_sample_major(
                k_proj, ulysses_local_cu, ulysses_size=ulysses_size, sequence_dim=1
            )
            v_proj = reorder_ulysses_rank_major_to_sample_major(
                v_proj, ulysses_local_cu, ulysses_size=ulysses_size, sequence_dim=1
            )
            b = reorder_ulysses_rank_major_to_sample_major(
                b, ulysses_local_cu, ulysses_size=ulysses_size, sequence_dim=1
            )
            a = reorder_ulysses_rank_major_to_sample_major(
                a, ulysses_local_cu, ulysses_size=ulysses_size, sequence_dim=1
            )

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

    gdn_core_cu = cu_seq_lens_q
    if cp_enabled and not headwise_enabled:
        # These projections share the same token ownership plan and dtype.
        # Route them in one packed all-to-all instead of paying three
        # collective launches in forward and three inverse launches in
        # backward.  Packing changes neither wire bytes nor tensor values.
        mixed_qkv, b, a = physical_to_owned_grouped(
            (mixed_qkv, b, a),
            plan=gdn_lossless_plan,
            cp_group=parallel_state.cp_group,
            sequence_dim=1,
            observer=gdn_cp_observer,
        )
        gdn_core_cu = mixed_qkv.new_tensor(gdn_lossless_plan.owned_cu_seqlens, dtype=torch.int32)

    if use_precomputed_states:
        # Modification: keep this disabled until FLA causal_conv1d_update decode path is validated.
        raise NotImplementedError("use_precomputed_states=True is not supported yet for causal_conv1d_update now.")
    else:
        if cache_params is not None:
            mixed_qkv_t = mixed_qkv.transpose(1, 2)
            conv_state = F.pad(mixed_qkv_t, (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            # Modification: shard conv1d weights per live head-parallel rank.
            if ulysses_enabled or headwise_enabled:
                conv_weight = self._get_local_conv1d_weight(
                    ulysses_rank=head_parallel_rank,
                    local_key_dim=local_key_dim,
                    local_value_dim=local_value_dim,
                )
            else:
                conv_weight = self.conv1d.weight.squeeze(1)
            if cp_enabled and not headwise_enabled and gdn_lossless_plan.local.owned_token_count == 0:
                pass
            elif headwise_enabled and gdn_headwise_layout.total_valid_tokens == 0:
                pass
            else:
                conv_cu = gdn_core_cu
                if cp_enabled and not headwise_enabled:
                    mixed_qkv, conv_cu = exchange_conv_halo(
                        mixed_qkv,
                        plan=gdn_lossless_plan,
                        cp_group=parallel_state.cp_group,
                        kernel_size=self.conv_kernel_size,
                        sequence_dim=1,
                        observer=gdn_cp_observer,
                    )
                # NPU causal-conv consumes device CU metadata.
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=conv_weight,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                    backend="triton",
                    cu_seqlens=conv_cu.npu(),
                )[0]
                if cp_enabled and not headwise_enabled:
                    mixed_qkv = trim_conv_halo(
                        mixed_qkv,
                        plan=gdn_lossless_plan,
                        kernel_size=self.conv_kernel_size,
                        sequence_dim=1,
                    )
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
    # Modification: slice A_log/dt_bias for the active local head shard.
    if ulysses_enabled or headwise_enabled:
        v_head_offset = head_parallel_rank * local_num_v_heads
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
        elif headwise_enabled:
            if gdn_headwise_layout.total_valid_tokens == 0:
                dependency = query.sum() + key.sum() + value.sum() + g.sum() + beta.sum()
                core_attn_out = value + dependency * 0
                last_recurrent_state = None
            else:
                query_gdr, key_gdr, use_qk_l2norm_in_kernel = prepare_gated_delta_rule_qk(
                    query,
                    key,
                    implementation=backend_impl,
                )
                core_attn_out, last_recurrent_state = call_chunk_gated_delta_rule(
                    self.chunk_gated_delta_rule,
                    query_gdr,
                    key_gdr,
                    value,
                    implementation=backend_impl,
                    metadata_is_canonical=not requires_chunked_varlen_metadata(backend_impl),
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=gdn_core_cu.npu(),
                    cu_seqlens_list=valid_points,
                    chunk_indices=chunk_indices,
                    chunk_indices_list=chunk_indices_list,
                )
        elif cp_enabled:
            aligned_host_cu = aligned_gdn_cu_seqlens(gdn_lossless_plan.owned_cu_seqlens)
            query_gdr, key_gdr, value_gdr, g_gdr, beta_gdr, aligned_cu, unpad_index = align_gdn_varlen_chunks(
                query,
                key,
                value,
                g,
                beta,
                gdn_core_cu,
                cu_seqlens_list=gdn_lossless_plan.owned_cu_seqlens,
            )
            if gdn_lossless_plan.local.owned_token_count == 0:
                # Empty owners still participate in KCP/state communication,
                # but the external Mojo norm has no rows to launch.  Keep the
                # tensors untouched and make the later kernel flag explicit.
                use_qk_l2norm_in_kernel = False
            else:
                query_gdr, key_gdr, use_qk_l2norm_in_kernel = prepare_gated_delta_rule_qk(
                    query_gdr,
                    key_gdr,
                    implementation=backend_impl,
                    force_external=self.gdn_context_parallel_implementation == "kcp",
                )
            if self.gdn_context_parallel_implementation == "kcp":
                # KCP's affine pre-scan and the local GDR core must consume the
                # exact same backend-normalized key. Normalizing once here also
                # gives both paths one shared autograd edge.
                needs_affine_readiness = not getattr(
                    self, "_gdn_kcp_affine_ready", False
                ) and kcp_plan_requires_affine_scan(gdn_lossless_plan)
                initial_state = resolve_kcp_initial_state(
                    key_gdr,
                    value_gdr,
                    g_gdr,
                    beta_gdr,
                    plan=gdn_lossless_plan,
                    cp_group=parallel_state.cp_group,
                    cu_seqlens=aligned_cu,
                    cu_seqlens_list=aligned_host_cu,
                    use_qk_l2norm=False,
                    affine_impl=kcp_affine_impl,
                    extra_participation=make_state_participation(query_gdr),
                    coordinate_readiness=needs_affine_readiness,
                    observer=gdn_cp_observer,
                )
                if needs_affine_readiness:
                    self._gdn_kcp_affine_ready = True
            else:
                state_template = make_state_template(query_gdr, value_gdr, aligned_cu)
                participation = make_state_participation(query, key, value, g, beta)
                initial_state = receive_initial_state(
                    plan=gdn_lossless_plan,
                    cp_group=parallel_state.cp_group,
                    state_template=state_template,
                    participation=participation,
                    observer=gdn_cp_observer,
                )
            if gdn_lossless_plan.local.owned_token_count == 0:
                core_attn_out = value.new_empty(value.shape)
                last_recurrent_state = initial_state
                if self.gdn_context_parallel_implementation == "kcp":
                    # Empty owners still have to traverse KCP AG/RS and every
                    # ownership-A2A backward ordinal. Preserve those graph
                    # edges without changing the empty numerical output.
                    core_attn_out = attach_state_dependency(core_attn_out, initial_state)
            else:
                if requires_chunked_varlen_metadata(backend_impl):
                    # The original precomputed metadata describes the physical
                    # input. Build the owned/chunk-aligned metadata from host CU
                    # once per layer+plan, then reuse it across steps. This avoids
                    # _ensure_varlen_metadata copying aligned_cu back to the host.
                    from veomni.ops.kernels.gated_delta_rule.varlen_metadata import (
                        precompute_varlen_metadata,
                    )

                    metadata_key = (tuple(aligned_host_cu), local_num_v_heads, str(query.device))
                    cached_metadata = getattr(self, "_gdn_lossless_npu_metadata_cache", None)
                    if cached_metadata is None or cached_metadata[0] != metadata_key:
                        cached_metadata = (
                            metadata_key,
                            precompute_varlen_metadata(
                                cu_seqlens=torch.tensor(aligned_host_cu, dtype=torch.int32),
                                num_heads=local_num_v_heads,
                                chunk_size=32,
                                device=query.device,
                            ),
                        )
                        self._gdn_lossless_npu_metadata_cache = cached_metadata
                    aligned_cu_list, aligned_chunk_indices, aligned_chunk_indices_list = cached_metadata[1]
                else:
                    # Legacy cu/host-CU backends deliberately do not consume
                    # chunk maps. Do not construct or pass maps they cannot
                    # consume; the adapter still validates canonical CU data.
                    aligned_cu_list = aligned_host_cu
                    aligned_chunk_indices = None
                    aligned_chunk_indices_list = None
                core_attn_out, last_recurrent_state = call_chunk_gated_delta_rule(
                    self.chunk_gated_delta_rule,
                    query_gdr,
                    key_gdr,
                    value_gdr,
                    implementation=backend_impl,
                    metadata_is_canonical=not requires_chunked_varlen_metadata(backend_impl),
                    g=g_gdr,
                    beta=beta_gdr,
                    initial_state=initial_state,
                    output_final_state=self.gdn_context_parallel_implementation == "state_passing_lossless",
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=aligned_cu.npu(),
                    cu_seqlens_list=aligned_cu_list,
                    chunk_indices=aligned_chunk_indices,
                    chunk_indices_list=aligned_chunk_indices_list,
                )
                core_attn_out = unpad_gdn_varlen_output(core_attn_out, unpad_index)
            if self.gdn_context_parallel_implementation == "state_passing_lossless":
                if last_recurrent_state is None:
                    raise RuntimeError("Lossless state passing requires a final recurrent state.")
                final_state = send_final_state(
                    last_recurrent_state,
                    plan=gdn_lossless_plan,
                    cp_group=parallel_state.cp_group,
                    observer=gdn_cp_observer,
                )
                core_attn_out = attach_state_dependency(core_attn_out, final_state)
        else:
            # Modification: use direct args and pass cu_seqlens for varlen FLA attention.
            backend_impl = self._veomni_chunk_gated_delta_rule_impl
            query_gdr, key_gdr, use_qk_l2norm_in_kernel = prepare_gated_delta_rule_qk(
                query,
                key,
                implementation=backend_impl,
            )
            core_attn_out, last_recurrent_state = call_chunk_gated_delta_rule(
                self.chunk_gated_delta_rule,
                query_gdr,
                key_gdr,
                value,
                implementation=backend_impl,
                metadata_is_canonical=not requires_chunked_varlen_metadata(backend_impl),
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seq_lens_q.npu(),
                cu_seqlens_list=cu_seqlens_list,
                chunk_indices=chunk_indices,
                chunk_indices_list=chunk_indices_list,
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

    if headwise_enabled:
        core_attn_out = restore_gdn_headwise_output(
            core_attn_out,
            layout=gdn_headwise_layout,
            group=parallel_state.sp_group,
        )
    elif cp_enabled:
        core_attn_out = owned_to_physical(
            core_attn_out,
            plan=gdn_lossless_plan,
            cp_group=parallel_state.cp_group,
            sequence_dim=1,
            observer=gdn_cp_observer,
        )
        if ulysses_enabled:
            core_attn_out = reorder_sample_major_to_ulysses_rank_major(
                core_attn_out,
                ulysses_local_cu,
                ulysses_size=ulysses_size,
                sequence_dim=1,
            )

    # Modification: gather attention output back to sequence-sharded layout before gated norm.
    if ulysses_enabled and not headwise_enabled:
        core_attn_out = gather_heads_scatter_seq(
            core_attn_out, head_dim=2, seq_dim=1, group=parallel_state.ulysses_group
        )

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


# ── DecoderLayer forward (NPU: plumb precomputed varlen metadata to GDN) ───────


@config.override_method(
    "Qwen3_5DecoderLayer.forward",
    description="Extract and pass cu_seq_lens_q + precomputed varlen metadata for AscendC GDN kernels in Qwen3_5DecoderLayer.forward",
)
def qwen3_5_decoder_layer_forward_patched(
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
    linear_attn_cu_seq_lens_q = kwargs.pop("linear_attn_cu_seq_lens_q", cu_seq_lens_q)
    # Keep the host CU list in kwargs for full-attention Ring CP; the flash
    # wrapper consumes it before calling any non-CP backend. Linear attention
    # also reuses the same once-per-forward metadata below.
    linear_attn_cu_seqlens_list = kwargs.pop("linear_attn_cu_seqlens_list_q", None)
    linear_attn_chunk_indices = kwargs.pop("chunk_indices_q", None)
    linear_attn_chunk_indices_list = kwargs.pop("chunk_indices_list_q", None)

    # Token Mixer
    if self.layer_type == "linear_attention":
        # Modification: pass linear-attention cu_seqlens + precomputed metadata through to GatedDeltaNet.forward.
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seq_lens_q=linear_attn_cu_seq_lens_q,
            cu_seqlens_list=linear_attn_cu_seqlens_list,
            chunk_indices=linear_attn_chunk_indices,
            chunk_indices_list=linear_attn_chunk_indices_list,
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


# ── TextModel forward (NPU: precompute varlen metadata once for all GDN layers) ─


@config.override_method(
    "Qwen3_5TextModel.forward",
    description="Precompute varlen metadata (cu_seqlens_list, chunk_indices, chunk_indices_list) once for all AscendC GDN layers to avoid per-layer tolist overhead",
)
def qwen3_5_text_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Qwen3_5ModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    # the hard coded `4` is for text, temporal, height and width.
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

    # Modification: precompute varlen metadata once for all GDN layers.  Keep
    # the full-attention rank-local CU and the GDN global-valid CU separate:
    # their boundaries differ under context parallelism.
    parallel_state = get_parallel_state()
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    linear_attn_cu_seq_lens_q = kwargs.get("linear_attn_cu_seq_lens_q", cu_seq_lens_q)
    if cu_seq_lens_q is not None and "cu_seqlens_list_q" not in kwargs:
        kwargs["cu_seqlens_list_q"] = [int(point) for point in cu_seq_lens_q.detach().cpu().tolist()]
    if linear_attn_cu_seq_lens_q is not None and "linear_attn_cu_seqlens_list_q" not in kwargs:
        kwargs["linear_attn_cu_seqlens_list_q"] = [
            int(point) for point in linear_attn_cu_seq_lens_q.detach().cpu().tolist()
        ]
    first_gdn_for_metadata = next(
        (
            getattr(layer, "linear_attn", None)
            for layer in self.layers[: self.config.num_hidden_layers]
            if getattr(layer, "layer_type", None) == "linear_attention"
        ),
        None,
    )
    backend_impl_for_metadata = (
        getattr(first_gdn_for_metadata, "_veomni_chunk_gated_delta_rule_impl", None)
        if first_gdn_for_metadata is not None
        else None
    )
    if linear_attn_cu_seq_lens_q is not None and requires_chunked_varlen_metadata(backend_impl_for_metadata):
        from veomni.ops.kernels.gated_delta_rule.varlen_metadata import precompute_varlen_metadata

        # Match the live head shard.  ``headwise_lossless`` uses the flattened
        # CP x Ulysses group; legacy state/KCP modes shard heads only by Ulysses.
        head_shard_size = (
            parallel_state.sp_size
            if getattr(first_gdn_for_metadata, "gdn_context_parallel_implementation", "disabled")
            == "headwise_lossless"
            else parallel_state.ulysses_size
        )
        num_v_heads = ulysses_local_head_count(
            self.config.linear_num_value_heads,
            head_shard_size,
        )
        linear_attn_cu_seqlens_list = kwargs.get("linear_attn_cu_seqlens_list_q")
        metadata_cu = (
            torch.tensor(linear_attn_cu_seqlens_list, dtype=torch.int32)
            if linear_attn_cu_seqlens_list is not None
            else linear_attn_cu_seq_lens_q
        )
        cu_seqlens_list, chunk_indices, chunk_indices_list = precompute_varlen_metadata(
            cu_seqlens=metadata_cu,
            num_heads=num_v_heads,
            chunk_size=64,
            device=inputs_embeds.device,
        )
        kwargs["linear_attn_cu_seqlens_list_q"] = cu_seqlens_list
        kwargs["chunk_indices_q"] = chunk_indices
        kwargs["chunk_indices_list_q"] = chunk_indices_list

    # TTX's first forward+VJP launch must happen outside decoder-layer
    # checkpointing.  Non-reentrant checkpoint recomputation re-enters the
    # layer body with a partially rebuilt autograd graph; doing the TTX
    # compile/warmup there can trigger StopRecomputationError or leave CP
    # ranks at different readiness points.  Prepare each distinct shape once
    # before the first checkpointed layer and coordinate failures across CP.
    if self.training and inputs_embeds.device.type == "npu" and parallel_state.cp_enabled:
        first_gdn = next(
            (
                getattr(layer, "linear_attn", None)
                for layer in self.layers[: self.config.num_hidden_layers]
                if getattr(layer, "layer_type", None) == "linear_attention"
            ),
            None,
        )
        if (
            first_gdn is not None
            and getattr(first_gdn, "gdn_context_parallel_implementation", "disabled") == "kcp"
            and linear_attn_cu_seq_lens_q is not None
        ):
            num_v_heads = ulysses_local_head_count(
                self.config.linear_num_value_heads,
                parallel_state.ulysses_size,
            )
            # Do not call a child projection here: decoder layers are fully
            # sharded by FSDP2 and their pre-forward hook has not run yet.  The
            # root FSDP cast_forward_inputs hook has already cast
            # ``inputs_embeds`` to the decoder compute dtype; an active device
            # autocast policy takes precedence.  This gives the TTX readiness
            # key the same dtype contract without bypassing FSDP materialization
            # or adding a one-token forward.
            device_type = inputs_embeds.device.type
            try:
                autocast_enabled = torch.is_autocast_enabled(device_type)
            except TypeError:  # older torch signature
                autocast_enabled = torch.is_autocast_enabled()
            autocast_dtype = torch.get_autocast_dtype(device_type) if autocast_enabled else None
            key_value_dtype = autocast_dtype or inputs_embeds.dtype
            beta_dtype = autocast_dtype or inputs_embeds.dtype
            warmup_signature = (
                inputs_embeds.device.type,
                inputs_embeds.device.index,
                int(num_v_heads),
                int(self.config.linear_key_head_dim),
                int(self.config.linear_value_head_dim),
                key_value_dtype,
                key_value_dtype,
                torch.float32,
                beta_dtype,
                get_kcp_affine_backend_identity(
                    resolve_kcp_affine_implementation(first_gdn._veomni_chunk_gated_delta_rule_impl)
                ),
            )
            if getattr(self, "_gdn_kcp_affine_warmup_signature", None) != warmup_signature:
                prepare_kcp_affine_summary(
                    resolve_kcp_affine_implementation(first_gdn._veomni_chunk_gated_delta_rule_impl),
                    device=inputs_embeds.device,
                    num_heads=int(num_v_heads),
                    key_dim=int(self.config.linear_key_head_dim),
                    value_dim=int(self.config.linear_value_head_dim),
                    key_dtype=key_value_dtype,
                    value_dtype=key_value_dtype,
                    g_dtype=torch.float32,
                    beta_dtype=beta_dtype,
                    cp_group=parallel_state.cp_group,
                    reference=inputs_embeds,
                )
                self._gdn_kcp_affine_warmup_signature = warmup_signature

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask

        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
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


@config.override_method(
    "Qwen3_5VisionModel.forward",
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens. Keep cu_seqlens on CPU to avoid per-layer NPU→CPU sync.",
)
def qwen3_5_vision_model_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    # Precomputed ViT metadata — a per-modality sub-dict Model.forward selects
    # from `multimodal_metadata` and passes as the single `vit_metadata` kwarg.
    # All .get() below fall back to None for callers that bypass MainCollator.
    # See .agents/knowledge/multimodal_metadata.md.
    reject_public_sequence_parallel_bypass(kwargs)
    vit_metadata = kwargs.pop("vit_metadata", None) or {}
    sequence_parallel_enabled = get_parallel_state().sp_enabled and not is_replicated_dummy_sequence_parallel()
    precomputed_grid_thw_list = vit_metadata.get("grid_thw_list")
    precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
    precomputed_max_seqlen = vit_metadata.get("max_seqlen")

    hidden_states = self.patch_embed(hidden_states)

    # Prefer the precomputed Python list (emitted by the data pipeline);
    # fallback `grid_thw.tolist()` covers callers that bypass MainCollator.
    # ``rot_pos_emb`` and ``fast_pos_embed_interpolate`` are permissive
    # (accept list or tensor) so they reuse the same materialisation.
    grid_thw_list = precomputed_grid_thw_list
    if grid_thw_list is None:
        grid_thw_list = grid_thw.tolist()

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)

    # --- Patch.1: Sequence parallel padding and slicing for position embeddings ---
    if sequence_parallel_enabled:
        # Note: grid_thw records the original, unpadded visual shapes. However, the data collator
        # pads the visual sequence (hidden_states) to a multiple of (sp_size * pad_scale)
        # to support Sequence Parallelism and subsequent spatial merging.
        #
        # pad_scale=4 matches the 4-to-1 spatial merge (2x2 pooling) ratio in the Qwen-VL Vision Tower.
        # We must manually pad and slice the generated position embeddings to ensure they
        # correctly align with the padded and sharded hidden states.
        pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
    # --- Patch.1 ---

    hidden_states = hidden_states + pos_embeds

    # ``total_seq_len`` is the patch count BEFORE any SP-pad. Derived host-side
    # from grid_thw_list so it stays a plain Python int even when cu_seqlens
    # is precomputed.
    total_seq_len = sum(t * h * w for t, h, w in grid_thw_list)

    # Prefer precomputed cu_seqlens (already includes any sp-pad tail entry
    # appended by the model's ``collate_multimodal_metadata`` collate hook).
    # Fallback builds host-side from grid_thw_list and handles sp-pad inline
    # below — same net behaviour. dtype selection:
    #  - FA2 requires cu_seqlens_q dtype int32
    #  - torch.onnx.export requires cu_seqlens_q same dtype as grid_thw
    # See https://github.com/huggingface/transformers/pull/34852 for context.
    if precomputed_cu_seqlens is not None:
        cu_seqlens = precomputed_cu_seqlens.to(
            hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            non_blocking=True,
        )
    else:
        cu_seqlens_list = [0]
        for t, h, w in grid_thw_list:
            frame_len = h * w
            for _ in range(t):
                cu_seqlens_list.append(cu_seqlens_list[-1] + frame_len)
        cu_seqlens = torch.tensor(
            cu_seqlens_list,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )

    rotary_pos_emb = self.rot_pos_emb(grid_thw_list)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    # --- Patch.2: Flatten full-sequence rotary embeddings using the actual total sequence length ---
    # In Sequence Parallelism, hidden_states.size(0) only represents the local shard length.
    # We must use total_seq_len (derived from unpadded grid_thw) to flatten the global
    # rotary_pos_emb. This ensures the embeddings cover the entire original sequence
    # before they are padded and sliced in Patch 3 to match the sharded hidden_states.
    rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
    # --- Patch.2 ---

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    pad_seq_len = 0
    if sequence_parallel_enabled:
        # --- Patch.3: Sequence parallel padding and slicing for sin/cos rotary embeddings ---
        cos, sin = position_embeddings
        # Similar to Patch.1, we pad and slice the rotary embeddings to align with the
        # padded hidden states, using pad_scale=4 to match the 4-to-1 spatial merge ratio.
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)
        # --- Patch.3 ---

        # --- Patch.4: Pad cu_seqlens to align with the padded hidden_states buffer under SP ---
        # The Data Collator pads hidden_states to a multiple of (sp_size * pad_scale),
        # but cu_seqlens (derived from grid_thw) only covers the original unpadded sequence.
        # We must extend cu_seqlens to cover the entire padded buffer by treating the
        # padding region as an additional "virtual sample". This ensures that varlen
        # kernels (like FlashAttention) process the full buffer, preventing shape
        # mismatches or collective communication hangs during subsequent Sequence
        # Parallel operations (e.g., All-to-All).
        sp_size = get_parallel_state().sp_size
        # Calculate global padding: (local_seq_len * num_ranks) - original_total_len
        # (total_seq_len is already a host int — no `.item()` sync needed here.)
        pad_seq_len = seq_len * sp_size - total_seq_len
        # Precomputed cu_seqlens already has the sp-pad tail entry appended by
        # the model's ``collate_multimodal_metadata`` collate hook; only the
        # fallback path needs to extend it here.
        if pad_seq_len > 0 and precomputed_cu_seqlens is None:
            # Append a new entry to cu_seqlens to include the padding tokens as a final segment
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.4 ---

    # --- Patch.5: Pre-compute max_seqlen once on the host ---
    # `flash_attn_varlen_func` expects `max_seqlen_q/k` as Python ints; passing
    # a 0-D GPU tensor forces an `.item()` inside the C++ binding. The HF body
    # of Qwen3_5VisionAttention.forward recomputes `(cu_seqlens[1:] - cu_seqlens[:-1]).max()`
    # per block, costing one host-device sync per ViT block per micro-batch
    # (~32 blocks × micro_batches per step). We hoist the computation here so
    # it happens once per ViT forward and thread the resulting int through
    # `**kwargs` to every block; the patched Qwen3_5VisionAttention.forward
    # picks it up via `vision_max_seqlen` and falls back to the original
    # recompute when the key is absent (so non-VeOmni callers keep working).
    # Gate is two-pronged:
    #   (a) `_VEOMNI_VISION_ATTENTION_PATCHED` — set per generated file. True
    #       only in GPU generated files where the consumer override is
    #       registered. NPU configs inject False because they reuse upstream
    #       HF Qwen3_5VisionAttention.forward, which recomputes max_seqlen and
    #       would leak the unused kwarg into `attention_interface(**kwargs)`.
    #   (b) `is_flash_attention_requested(self.config)` — only FA's
    #       `flash_attn_varlen_func` benefits from the int hand-off; eager
    #       and sdpa paths in the consumer pop+discard the kwarg, so the
    #       host sync would be wasted.
    if _VEOMNI_VISION_ATTENTION_PATCHED and is_flash_attention_requested(self.config):
        if precomputed_max_seqlen is not None:
            # Collator-side max already accounts for sp-pad; use as-is.
            kwargs["vision_max_seqlen"] = precomputed_max_seqlen
        else:
            max_frame_len = max((h * w for _, h, w in grid_thw_list), default=0)
            kwargs["vision_max_seqlen"] = max(max_frame_len, pad_seq_len)
    # --- Patch.5 ---

    # --- Patch.6: Keep cu_seqlens on CPU to avoid per-layer NPU→CPU sync. ---
    cu_seqlens = cu_seqlens.to("cpu")
    # --- Patch.6 ---

    for blk in self.blocks:
        if is_replicated_dummy_sequence_parallel():
            hidden_states = _call_replicated_dummy_checkpointed_module(
                _DUMMY_SP_TOKEN,
                blk,
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

    merged_hidden_states = self.merger(hidden_states)

    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
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
