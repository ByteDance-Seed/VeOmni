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
Patch configuration for Qwen3_5Moe GPU/SP patched modeling generation.

Regen command:
patchgen veomni.models.transformers.qwen3_5_moe.qwen3_5_moe_gpu_patch_gen_config -o veomni/models/transformers/qwen3_5_moe/generated --diff

Patches applied:
1. Fused MoE expert replacement (merged gate_up_proj layout).
2. Device-agnostic GatedDeltaNet init and varlen FLA forward.
3. DecoderLayer forward with cu_seq_lens_q passthrough.
4. Fused loss + aux_loss in ForConditionalGeneration.
"""

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
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeModelOutputWithPast,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.qwen3_5.qwen3_5_gpu_patch_gen_config import (
    _mtp_loss_weight,
    compute_mtp_loss,
    make_mtp_labels,
    qwen3_5_gated_deltanet_forward_patched,
    qwen3_5_gated_deltanet_get_local_conv1d_weight,
    qwen3_5_gated_deltanet_init_patched,
    qwen3_5_model_get_image_features,
    qwen3_5_model_get_placeholder_mask,
    qwen3_5_text_model_forward_patched,
    qwen3_5_text_model_update_linear_attn_mask,
    qwen3_5_vision_attention_forward_patched,
    qwen3_5_vision_model_dummy_forward,
    qwen3_5_vision_model_fast_pos_embed_interpolate,
    qwen3_5_vision_model_forward,
    qwen3_5_vision_model_rot_pos_emb,
)
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.model_outputs import FusedLinearAuxOutputMixin, MoeCausalLMOutputWithLogProbs
from veomni.utils.moe_router_replay import get_active_replay, maybe_replay_indices


logger = logging.get_logger(__name__)


config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_gpu.py",
    description="Qwen3_5Moe with LigerKernel GPU replacements, fused MoE, and VeOmni SP/fused loss patches",
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
config.add_import("veomni.utils.constants", names=["IGNORE_INDEX", "IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
# Surface ``MoeCausalLMOutputWithLogProbs`` so the patched text ``forward`` can return
# per-token log-probs in the unified MoE output dataclass.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.add_import("veomni.utils.moe_router_replay", names=["get_active_replay", "maybe_replay_indices"])
config.add_helper(_mtp_loss_weight)
config.add_helper(compute_mtp_loss)
config.add_helper(make_mtp_labels)
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # Selection of FusedRMSNormGated / causal_conv1d / chunk_gated_delta_rule
    # has moved into OpSlot guards below (driven by OpsImplementationConfig).
    # These None placeholders preserve two pieces of the original module:
    #   (1) the upstream HF top-level
    #       `is_fast_path_available = all((causal_conv1d_fn, ...))` resolves
    #       to False, keeping the legacy warning behaviour; and
    #   (2) the decode-only `*_update` / `fused_recurrent_*` aliases satisfy
    #       the `<fla_name> or <torch_fallback>` assignments in __init__
    #       (the precomputed-state path raises NotImplementedError anyway).
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
    # Bound at model-build time by _bind_veomni_ops() in auto.py. The three
    # linear-attention slots replace the previous import-time fla/torch
    # selection inside Qwen3_5MoeGatedDeltaNet.__init__ /forward.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_moe_experts_forward = OpSlot("moe_experts", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    veomni_rms_norm_gated = OpSlot("rms_norm_gated", "standard")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")
    """
)

# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None
gather_outputs = None
slice_input_tensor = None
veomni_rms_norm_gated = None  # OpSlot, declared in post-import block above
veomni_causal_conv1d = None  # OpSlot, declared in post-import block above
veomni_chunk_gated_delta_rule = None  # OpSlot, declared in post-import block above

# Mirror the GPU sentinel from qwen3_5_gpu_patch_gen_config: this config
# registers Qwen3_5MoeVisionAttention.forward as the consumer, so the
# pre-computed `vision_max_seqlen` int is safe to write. See Patch.5 in
# qwen3_5_gpu_patch_gen_config.py for the full rationale.
config.add_post_import_block("_VEOMNI_VISION_ATTENTION_PATCHED = True")


# ── RMSNorm (OpSlot guard, functional Liger kernel) ──────────────────────────


@config.override_method(
    "Qwen3_5MoeRMSNorm.forward",
    description="OpSlot guard for Liger fused RMSNorm (Qwen3.5 1+weight formulation)",
)
def qwen3_5_moe_rmsnorm_forward_patched(self, x):
    # Modification: OpSlot guard — use fused RMSNorm kernel when bound.
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(x, self.weight, self.eps)
    # Original HF code below, unchanged.
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


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
    description="Avoid in-place += on custom autograd Function output; call maybe_replay_indices for RL router replay",
)
def qwen3_5_moe_sparse_moe_block_forward_patched(
    self, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
    # MoE router replay: when an RL framework has installed a manager via
    # ``set_active_replay``, the manager may substitute ``selected_experts``
    # with previously recorded target indices. The manager's sole
    # responsibility is choosing indices; all model-specific post-topk
    # weight math (softmax recompute, gather, renorm, dtype cast) is
    # replicated here so the cross-framework controller stays
    # model-agnostic. transformers v5.8 fixed Qwen3.5-MoE's ``TopKRouter``
    # the same way as Qwen3-MoE (#715): it now returns pre-softmax
    # ``router_logits`` and discards its internal post-softmax matrix after
    # top-k, so we recompute ``softmax`` here to feed the RR contract.
    # Qwen3.5-MoE's native router always renormalizes the top-k probs, so
    # the gathered weights are renormalized unconditionally.
    if get_active_replay() is not None:
        target_dtype = routing_weights.dtype
        routing_scores = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        selected_experts = maybe_replay_indices(self.gate, routing_scores, selected_experts)
        routing_weights = routing_scores.gather(1, selected_experts)
        routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(target_dtype)
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

    # Modification: use out-of-place add instead of `expert_output += shared_expert_output`
    # to avoid "Output of MergedFc1TritonFusedMoeExpertFunctionBackward is a view and is
    # being modified inplace" RuntimeError from PyTorch autograd.
    expert_output = expert_output + shared_expert_output
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
    return expert_output, router_logits


# ── ViT patches ───────────────────────────────────────────────────────────────

config.override_method(
    "Qwen3_5MoeModel.get_image_features",
    replacement=qwen3_5_model_get_image_features,
    description="Remove unnecessary split operation to maintain contiguous memory layout.",
)

config.override_method(
    "Qwen3_5MoeModel.get_placeholder_mask",
    replacement=qwen3_5_model_get_placeholder_mask,
    description="Extract multimodal placeholder masks from input_ids using self-defined placeholder IDs.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.rot_pos_emb",
    replacement=qwen3_5_vision_model_rot_pos_emb,
    description="Accept pre-materialized grid_thw metadata to avoid redundant host sync in vision RoPE setup.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.fast_pos_embed_interpolate",
    replacement=qwen3_5_vision_model_fast_pos_embed_interpolate,
    description="Optimized bilinear interpolation for high-resolution vision embeddings, adapted from vLLM.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.forward",
    replacement=qwen3_5_vision_model_forward,
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens.",
)

config.override_method(
    "Qwen3_5MoeVisionModel.dummy_forward",
    replacement=qwen3_5_vision_model_dummy_forward,
    description="Add dummy_forward to prevent FSDP reduce-scatter hang on uneven multimodal batches.",
)

config.override_method(
    "Qwen3_5MoeVisionAttention.forward",
    replacement=qwen3_5_vision_attention_forward_patched,
    description=(
        "Read pre-computed `vision_max_seqlen` (Python int) from kwargs to avoid "
        "the per-block GPU->CPU sync that flash_attn_varlen_func incurs when "
        "`max_length_q/k` are 0-D GPU tensors (FA's C++ binding `.item()`s them)."
    ),
)


config.override_method(
    "Qwen3_5MoeTextModel.forward",
    replacement=qwen3_5_text_model_forward_patched,
    name_map={"Qwen3_5": "Qwen3_5Moe"},
    description="Expose MTP context when requested by the outer MTP objective",
)


@config.override_method(
    "Qwen3_5MoeModel.forward",
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-GPU sync avoidance via pre-computed metadata."
    ),
)
def qwen3_5_moe_model_forward_patched(
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
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5MoeModelOutputWithPast:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Token type IDs for multimodal inputs.
    cache_position (`torch.LongTensor`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # --- Patch.1: Support Ulysses SP by using pre-computed image and video masks ---
    # We use pre-computed masks to ensure all ranks have a consistent view of multimodal
    # placeholder positions. If masks are not provided, we reconstruct the full sequence
    # via all_gather to compute them locally.
    image_mask = kwargs.get("image_mask", None)
    video_mask = kwargs.get("video_mask", None)

    # if None, calculate mask
    if video_mask is None and image_mask is None:
        if get_parallel_state().sp_enabled:
            input_ids_list = [torch.zeros_like(input_ids) for i in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            input_ids = torch.cat(input_ids_list, dim=1)
        image_mask, video_mask = self.get_placeholder_mask(input_ids)
    # --- Patch.1 ---

    # --- Patch.4: Pop pre-computed Flash Attention kwargs to avoid ViT forward re-computation ---
    # The LM-level flash-attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) are injected for packed-sequence attention. They must not reach the ViT, which computes its own `cu_seqlens`
    flash_attn_kwargs = {}
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.4 ---

    # --- Patch.6 ---
    # Mirror of qwen3_5: unpack per-modality ViT kwargs from
    # `multimodal_metadata` (collator-precomputed) so the patched ViT
    # forward can skip the in-forward .tolist() / cu_seqlens build.
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
    # --- Patch.6 ---

    # --- Patch.1: Support Ulysses SP by transposing layout for multimodal scattering ---
    if get_parallel_state().sp_enabled:
        # Transpose from (batch, local_seq, full_hidden) to (batch, full_seq, local_hidden).
        # This gives each rank visibility over the ENTIRE sequence length, which is
        # necessary to scatter vision features into their correct global positions
        # as defined by the global pre-computed masks.
        inputs_embeds = gather_outputs(inputs_embeds, gather_dim=1, group=get_parallel_state().sp_group)

    # --- Patch.1 ---

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True, **image_vit_kwargs
        )
        image_embeds = image_outputs.pooler_output

        # --- Patch.1: Shard image_embeds for sequence parallel scatter ---
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)

        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # `masked_scatter` consumes exactly `image_mask.sum()` elements from `image_embeds`, taking the
        # leading rows in order — image-placeholder positions in `input_ids` are laid out in the same
        # order as their vision tokens, and the data collator pads the vision sequence only at the
        # *end*. So any padded vision rows are trailing and simply go unused; no `image_embeds[:n]`
        # slice is needed, which also removes the `image_mask.sum().item()` host-device sync.
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        # sequence parallel patch for image_mask
        if get_parallel_state().sp_enabled:
            seq_len = image_mask.shape[1]

            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            image_mask = image_mask[:, rank_start:rank_end]
        # --- Patch.1 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.2: Dummy forward to prevent FSDP reduce-scatter hang on uneven multimodal batches ---
        # add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values while others get valid pixel_values
        vision_output = self.visual.dummy_forward()
        fake_embeds = vision_output.pooler_output
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.2 ---

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True, **video_vit_kwargs
        )
        video_embeds = video_outputs.pooler_output

        # --- Patch.1: Shard video_embeds for sequence parallel scatter ---
        # sequence parallel patch for video embeds
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            video_embeds = gather_outputs(video_embeds, gather_dim=0, group=get_parallel_state().sp_group)

        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # As with `image_embeds` above: `masked_scatter` uses exactly `video_mask.sum()` leading rows,
        # any collator-padded vision rows are trailing and unused — no `video_embeds[:n]` slice (and no
        # `video_mask.sum().item()` host-device sync) needed.
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        # sequence parallel patch for video_mask
        if get_parallel_state().sp_enabled:
            seq_len = video_mask.shape[1]

            seq_per_rank = seq_len // get_parallel_state().sp_size
            rank_start = get_parallel_state().sp_rank * seq_per_rank
            rank_end = rank_start + seq_per_rank

            video_mask = video_mask[:, rank_start:rank_end]
        # --- Patch.1 ---
    elif get_parallel_state().fsdp_enabled:
        # --- Patch.2: Dummy forward for video encoder to avoid FSDP hang ---
        # add dummy ViT forward to avoid FSDP reduce-scatter hang
        # when some ranks get None pixel_values_videos while others get valid pixel_values_videos
        vision_output = self.visual.dummy_forward()
        fake_embeds = vision_output.pooler_output
        fake_embeds = fake_embeds.mean() * 0.0
        fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + fake_embeds
        # --- Patch.2 ---

    # --- Patch.1: Final transpose back to standard sequence-sharded layout ---
    if get_parallel_state().sp_enabled:
        # Restore the layout to (batch, local_seq, full_hidden) for subsequent
        # transformer layers, which expect standard Sequence Parallel sharding.
        inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, group=get_parallel_state().sp_group)

    # --- Patch.1 ---

    if position_ids is None:
        # v5 `compute_3d_position_ids` gates M-RoPE on `mm_token_type_ids`
        # via its `can_compute_mrope` check; without it the multimodal
        # branch silently falls through and the call returns `None`,
        # leaving `position_ids=None` for the language model. Derive
        # `mm_token_type_ids` from `input_ids` here so the M-RoPE branch
        # actually runs whenever multimodal grids are present.
        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            mm_token_type_ids = mm_token_type_ids_from_input_ids(  # noqa: F821 defined via add_helper
                input_ids, self.config
            )
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )
    else:
        # --- Patch.3: Transpose pre-computed position_ids if they follow VeOmni collation format ---
        # When position_ids are pre-computed during data preprocessing (for varlen/packed data),
        # they are typically collated into (batch_size, 3, seq_len) shape. We transpose them
        if position_ids.dim() == 3 and position_ids.shape[1] == 3:
            position_ids = position_ids.transpose(0, 1).contiguous()
        # --- Patch.3 ---

    # --- Patch.4: Restore pre-computed Flash Attention kwargs for language model ---
    kwargs.update(flash_attn_kwargs)
    # --- Patch.4 ---

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    output_kwargs = dict(outputs)
    output_kwargs["rope_deltas"] = self.rope_deltas
    if getattr(outputs, "mtp_context", None) is not None:
        return Qwen3_5MoeMTPContextOutput(**output_kwargs)  # noqa: F821
    return Qwen3_5MoeModelOutputWithPast(**output_kwargs)


# Surface ``Qwen3_5MoeCausalLMOutputWithLogProbs`` so the patched multimodal
# ``forward`` can return per-token log-probs while preserving ``rope_deltas``.
# See qwen3_5_gpu_patch_gen_config.py for why @auto_docstring is skipped.
@config.add_helper_after("Qwen3_5MoeCausalLMOutputWithPast")
@dataclass
class Qwen3_5MoeCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen3_5MoeCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
        Per-token tensors produced by the fused-linear loss path
        (``log_probs`` / ``entropy``; plus ``distillation_losses`` /
        ``student_mass`` / ``teacher_mass`` on the top-k distillation path).
        ``None`` on the plain loss path; populated when ``return_log_probs=True``.
    """

    loss_dict: dict[str, torch.Tensor] | None = None


@config.add_helper_after("Qwen3_5MoeModelOutputWithPast")
@dataclass
class Qwen3_5MoeMTPContextOutput(Qwen3_5MoeModelOutputWithPast):
    mtp_context: dict | None = None


@config.add_helper_after("Qwen3_5MoeDecoderLayer")
class Qwen3_5MoeMTP(nn.Module):
    """Qwen3.5 MoE multi-token predictor with one layer per prediction depth."""

    def __init__(self, config):
        """Build the shared fusion modules and depth-specific decoder layers."""
        super().__init__()
        assert not getattr(config, "mtp_use_dedicated_embeddings", False)
        num_layers = int(config.mtp_num_hidden_layers)
        assert "full_attention" in config.layer_types
        layer_idx = config.layer_types.index("full_attention")
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(config, layer_idx) for _ in range(num_layers)])
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states, inputs_embeds, position_embeddings, attention_mask=None, position_ids=None, **kwargs
    ):
        """Return recurrent hidden states and optional router logits for every MTP depth."""
        assert kwargs.get("past_key_values") is None and not kwargs.get("use_cache", False)
        output_router_logits = kwargs.pop("output_router_logits", False)
        depth_hidden_states = []
        depth_router_logits = [] if output_router_logits else None
        for depth, layer in enumerate(self.layers):
            shift = depth + 1
            shifted_embeds = F.pad(inputs_embeds, (0, 0, 0, shift))[:, shift:, :]
            hidden_states = self.fc(
                torch.cat([self.pre_fc_norm_embedding(shifted_embeds), self.pre_fc_norm_hidden(hidden_states)], dim=-1)
            )
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                return_router_logits=output_router_logits,
                **kwargs,
            )
            if output_router_logits:
                hidden_states, router_logits = hidden_states
                if router_logits is None:
                    raise ValueError(f"MTP depth {depth} returned no router logits.")
                depth_router_logits.append(router_logits)
            hidden_states = self.norm(hidden_states)
            depth_hidden_states.append(hidden_states)
        return tuple(depth_hidden_states), tuple(depth_router_logits) if depth_router_logits is not None else None


@config.add_helper
def compute_mtp_router_aux_loss(
    router_loss_fn,
    foundation_router_logits,
    mtp_router_logits,
    attention_mask,
    mtp_labels,
    num_experts,
    top_k,
):
    """Compute one load-balancing loss over trunk and MTP routers with layer-specific masks."""
    if not isinstance(foundation_router_logits, tuple) or not isinstance(mtp_router_logits, tuple):
        raise ValueError("Foundation and MTP router logits must both be tuples when router output is enabled.")
    if mtp_labels.ndim != 3:
        raise ValueError(
            f"MTP labels must have shape [batch, depth, sequence]; got mtp_labels.shape={tuple(mtp_labels.shape)}."
        )

    batch_size, num_depths, sequence_length = mtp_labels.shape
    if len(mtp_router_logits) != num_depths:
        raise ValueError(
            "MTP router-logit depth must match the label depth; "
            f"got {len(mtp_router_logits)} router row(s) and {num_depths} label row(s)."
        )

    combined_router_logits = foundation_router_logits + mtp_router_logits
    expected_tokens = batch_size * sequence_length
    for layer_idx, router_logits in enumerate(combined_router_logits):
        if router_logits.ndim != 2 or router_logits.shape != (expected_tokens, num_experts):
            raise ValueError(
                "Each router-logit tensor must have shape [batch * sequence, num_experts]; "
                f"layer {layer_idx} has shape={tuple(router_logits.shape)}, "
                f"expected=({expected_tokens}, {num_experts})."
            )

    if attention_mask is None:
        foundation_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=mtp_labels.device,
        )
    else:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "Router attention_mask must have shape [batch, sequence]; "
                f"got attention_mask.shape={tuple(attention_mask.shape)}, "
                f"expected=({batch_size}, {sequence_length})."
            )
        foundation_mask = attention_mask

    foundation_masks = foundation_mask.unsqueeze(0).expand(len(foundation_router_logits), -1, -1)
    mtp_masks = (
        (mtp_labels != IGNORE_INDEX)
        .to(  # noqa: F821
            device=foundation_mask.device,
            dtype=foundation_mask.dtype,
        )
        .transpose(0, 1)
    )
    layer_attention_mask = torch.cat((foundation_masks, mtp_masks), dim=0)
    # Present all router rows as one logical layer so both the upstream eager
    # implementation (2D masks only) and VeOmni kernels apply the same global
    # load-balancing formula with depth-specific masks.
    flattened_router_logits = (torch.cat(combined_router_logits, dim=0),)
    flattened_attention_mask = layer_attention_mask.flatten(0, 1)
    aux_loss = router_loss_fn(
        flattened_router_logits,
        num_experts,
        top_k,
        flattened_attention_mask,
    )
    return aux_loss, combined_router_logits


@config.add_helper
def mm_token_type_ids_from_input_ids(input_ids, config):
    # transformers v5 VLMs require `mm_token_type_ids` to compute multimodal
    # RoPE (M-RoPE): text=0, image=1, video=2 per token. HF's processor emits
    # it; VeOmni's data pipeline carries modality only via the multimodal
    # token ids inside `input_ids`, so derive the type ids from those here.
    # `config` selects the token-id namespace and must match `input_ids`: the
    # live model config on the `forward` path, the IMAGE/VIDEO_INPUT_INDEX fake
    # config in the `get_position_id` precompute path. Do not unify the two
    # call sites onto one config.
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == config.image_token_id] = 1
    mm_token_type_ids[input_ids == config.video_token_id] = 2
    return mm_token_type_ids


@config.add_helper
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    # v5 `get_rope_index` requires `mm_token_type_ids`; derive it from
    # `input_ids` when the data pipeline did not pass it explicitly.
    if kwargs.get("mm_token_type_ids") is None and kwargs.get("input_ids") is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids_from_input_ids(  # noqa: F821 defined via add_helper
            kwargs["input_ids"], self.config
        )
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


@config.add_helper
def collate_multimodal_metadata(batch, sp_pad):
    """Derive ``multimodal_metadata`` for the Qwen3.5-VL-MoE ViT.

    Module-level so ``get_metadata_collate_func`` can hand it to VeOmni's
    collator as a picklable callable (mirrors ``get_position_id``). Runs
    purely on CPU inside the collator after SP padding — every value it
    produces (CPU int tensors / Python ints / lists) is consumed by the ViT
    forward without a host-device sync.

    ``batch`` is the packed (+ SP-padded) batch dict; ``sp_pad`` maps
    ``pixel_values`` / ``pixel_values_videos`` to the number of patch rows
    the SP collator appended. Mutates ``batch`` in place, writing
    ``batch["multimodal_metadata"]``.
    """
    md = {}
    # ViT varlen-attention metadata, derived from the HF processor's
    # ``*_grid_thw`` CPU LongTensor (packed across the batch by the collator
    # via DataCollateInfo pack_dim=0). ``.tolist()`` here is a pure-CPU op —
    # the collator runs in dataloader workers, no host-device sync.
    # Temporal unroll: each (t, h, w) expands to ``t`` cu steps of ``h * w``.
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
        # SP-pad tail: the collator zero-pads pixel_values to SP-divisible;
        # those patches become one synthetic "image" so varlen attention
        # treats them as an independent sequence (mirrors the position_ids==0
        # text-side SP-pad convention). Discarded after the per-rank slice.
        pad = sp_pad.get(pad_key, 0)
        if pad > 0:
            cu.append(cu[-1] + pad)
            max_hw = max(max_hw, pad)
        # device='cpu': this runs in CPU dataloader workers — pin to CPU so a
        # global torch.set_default_device('cuda') can't misallocate it.
        md[f"vit_{modality}_cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cpu")
        md[f"vit_{modality}_max_seqlen"] = max_hw

    if md:
        batch["multimodal_metadata"] = md


# ================================================================
# Helper: _Qwen3_5MoeFakeForPosID  (emitted into the generated file via
# add_helper). Picklable fake `self` used by `get_position_id_func`.
# ================================================================
@config.add_helper
class _Qwen3_5MoeFakeForPosID(SimpleNamespace):
    """Picklable fake `self` used by `get_position_id_func` — must be a
    module-level class so `get_vision_position_ids` survives the pickle
    round-trip that happens when this object reaches a dataloader worker
    via `multiprocessing.spawn`. Assigning a bound method to a plain
    `SimpleNamespace` instance deadlocks on unpickle (`AttributeError:
    'types.SimpleNamespace' object has no attribute 'get_vision_position_ids'`)
    because pickle reduces a bound method to `getattr(self.__self__, name)`
    and `name` is the very attribute being restored from `__dict__`. As a
    class attribute, the method is resolved via class lookup and never has
    to be pickled."""

    def get_vision_position_ids(self, *args, **kwargs):
        return Qwen3_5MoeModel.get_vision_position_ids(self, *args, **kwargs)


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_position_id_func",
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)
def qwen3_5_moe_forconditional_generation_get_position_id_func(self):
    fake_config = copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    # Use a module-level fake-self class (see `_Qwen3_5MoeFakeForPosID`
    # above) instead of `SimpleNamespace + bound-method on instance`. The
    # bound-method form survives single-process callers but deadlocks on
    # `multiprocessing.spawn`'s pickle round-trip used by the streaming
    # dataloader workers — see the class's docstring for the full rationale.
    fake_model = _Qwen3_5MoeFakeForPosID(config=fake_config)  # noqa: F821
    return partial(get_position_id, Qwen3_5MoeModel.get_rope_index, fake_model)  # noqa: F821


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_metadata_collate_func",
    description="Expose CPU-side ViT multimodal-metadata derivation to the VeOmni collator",
)
def qwen3_5_moe_forconditional_generation_get_metadata_collate_func(self):
    # collate_multimodal_metadata is a module-level helper (added via
    # add_helper) — a bare function reference is picklable for the DataLoader
    # workers; the Qwen3.5-VL-MoE ViT formula needs no model config.
    return collate_multimodal_metadata  # noqa: F821 defined via add_helper


# ── MoE Expert replacement (merged gate_up_proj layout) ─────────────────────────


@config.replace_class(
    "Qwen3_5MoeExperts",
    description="Remove @use_experts_implementation decorator and add OpSlot-based fused MoE dispatch",
)
class PatchedQwen3_5MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors.

    Replaces the HF class to remove the @use_experts_implementation decorator
    (which routes to grouped_mm and bypasses our fused MoE path) and to add
    VeOmni fused MoE dispatch via OpSlot.
    """

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
        # Modification: OpSlot guard — dispatch to fused MoE kernel when bound.
        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights)

        # Original HF eager loop below, unchanged.
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

config.override_method(
    "Qwen3_5MoeTextModel._update_linear_attn_mask",
    replacement=qwen3_5_text_model_update_linear_attn_mask,
    description="Avoid host-device sync: decide linear-attention padding-mask zeroing without reading GPU scalars.",
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
    return_router_logits = kwargs.pop("return_router_logits", False)
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Modification: read varlen metadata from kwargs and enforce it for linear-attention varlen kernels.
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    assert cu_seq_lens_q is not None, (
        "cu_seq_lens_q must be provided to support varlen Flash Linear Attention, varlen Conv1D,"
        "and to remove the full Flash Attention CPU-GPU sync."
    )
    linear_attn_cu_seq_lens_q = kwargs.pop("linear_attn_cu_seq_lens_q", cu_seq_lens_q)

    # Token Mixer
    if self.layer_type == "linear_attention":
        # Modification: pass linear-attention cu_seqlens through to Qwen3_5MoeGatedDeltaNet.forward.
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seq_lens_q=linear_attn_cu_seq_lens_q,
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
    router_logits = None
    if isinstance(hidden_states, tuple):
        hidden_states, router_logits = hidden_states
    hidden_states = residual + hidden_states
    if return_router_logits:
        return hidden_states, router_logits
    return hidden_states


# ── ForCausalLM forward (fused loss + aux_loss) ──────────────────────────────────


@config.override_method(
    "Qwen3_5MoeForCausalLM.forward", description="Support fused cross entropy path in Qwen3_5MoeForCausalLM.forward"
)
def qwen3_5_moe_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    cache_position (`torch.LongTensor`, *optional*):
        Indices depicting the position of the input sequence tokens in the sequence.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

    >>> model = Qwen3_5MoeForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            # Modification: VeOmni's patched `loss_function` (via LOSS_MAPPING)
            # returns (loss, logits, fused_linear_aux); unpack to match the
            # OpSlot branch above.
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        # Modification: OpSlot guard for load-balancing loss.
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None:
            loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)

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


# ── ForConditionalGeneration forward (fused loss + aux_loss) ─────────────────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.__init__",
    description="Build the MTP head when enabled",
)
def qwen3_5_moe_forconditional_generation_init_patched(self, config):
    """Initialize Qwen3.5-MoE conditional generation and its optional MTP head."""
    super().__init__(config)
    self.model = Qwen3_5MoeModel(config)
    self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    self.mtp = None
    weight = _mtp_loss_weight(config.text_config)  # noqa: F821
    if weight is not None:
        assert not get_parallel_state().sp_enabled, "Qwen3.5 MoE MTP does not support sequence parallel."
        self.mtp = Qwen3_5MoeMTP(config.text_config)  # noqa: F821
        logger.info_rank0(
            f"Qwen3.5 MoE MTP enabled: {config.text_config.mtp_num_hidden_layers} layer(s), loss weight {weight}."
        )
    self.post_init()


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_extra_collate_infos",
    description="Declare the MTP label collate rule",
)
def qwen3_5_moe_forconditional_generation_get_extra_collate_infos(self):
    """Declare the packing rule for MoE MTP labels when enabled."""
    if self.mtp is None:
        return {}
    return {"mtp_labels": (-1, True, IGNORE_INDEX, 1)}  # noqa: F821


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_sample_collate_func",
    description="Expose the per-sample MTP label shift",
)
def qwen3_5_moe_forconditional_generation_get_sample_collate_func(self):
    """Return the per-sample hook that creates all configured MTP depth labels."""
    if self.mtp is None:
        return None
    return partial(make_mtp_labels, num_depths=len(self.mtp.layers))  # noqa: F821


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
    output_router_logits: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    mtp_labels: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Qwen3_5MoeCausalLMOutputWithLogProbs:
    """Run MoE conditional generation and combine foundation, MTP, and router losses."""
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
    )
    requires_mtp_context = self.mtp is not None and labels is not None
    if requires_mtp_context and mtp_labels is None:
        raise ValueError("Qwen3.5 MoE MTP loss requires `mtp_labels` when `labels` are provided.")

    model_kwargs = dict(kwargs)
    model_kwargs["return_mtp_context"] = requires_mtp_context
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
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        **model_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
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
            # Modification: VeOmni's patched `loss_function` (via LOSS_MAPPING)
            # returns (loss, logits, fused_linear_aux); unpack to match the
            # OpSlot branch above.
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
    else:
        logits = self.lm_head(hidden_states)

    loss_dict = None
    mtp_router_logits = None
    if requires_mtp_context:
        mtp_context = getattr(outputs, "mtp_context", None)
        if mtp_context is None:
            raise RuntimeError("Qwen3.5 MoE MTP context was requested but the language model did not return it.")
        mtp_hidden_states, mtp_router_logits = self.mtp(
            hidden_states=outputs[0],
            inputs_embeds=mtp_context["inputs_embeds"],
            position_embeddings=mtp_context["position_embeddings"],
            attention_mask=mtp_context["attention_mask"],
            position_ids=mtp_context["position_ids"],
            cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
            cu_seq_lens_k=kwargs.get("cu_seq_lens_k"),
            max_length_q=kwargs.get("max_length_q"),
            max_length_k=kwargs.get("max_length_k"),
            output_router_logits=output_router_logits,
        )
        mtp_loss_fn = veomni_causal_lm_loss if veomni_causal_lm_loss.use_non_eager_impl else self.loss_function
        mtp_loss = compute_mtp_loss(  # noqa: F821
            mtp_loss_fn,
            mtp_hidden_states,
            mtp_labels,
            weights=self.lm_head.weight,
            vocab_size=self.config.text_config.vocab_size,
            **kwargs,
        )
        weight = _mtp_loss_weight(self.config.text_config)  # noqa: F821
        loss_dict = {"foundation_loss": loss, "mtp_loss": weight * mtp_loss}

    router_logits = outputs.router_logits
    aux_loss = None
    if output_router_logits:
        router_loss_fn = (
            veomni_load_balancing_loss if veomni_load_balancing_loss.use_non_eager_impl else load_balancing_loss_func
        )
        if mtp_router_logits is not None:
            aux_loss, router_logits = compute_mtp_router_aux_loss(  # noqa: F821
                router_loss_fn,
                outputs.router_logits,
                mtp_router_logits,
                attention_mask,
                mtp_labels,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
            )
        else:
            aux_loss = router_loss_fn(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None and isinstance(aux_loss, torch.Tensor):
            loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)
            if loss_dict is not None:
                loss_dict["foundation_loss"] = loss

    return Qwen3_5MoeCausalLMOutputWithLogProbs(
        loss=loss,
        loss_dict=loss_dict,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=router_logits,
        rope_deltas=outputs.rope_deltas,
        fused_linear_aux=fused_linear_aux,
    )


# ── Expert parallel plan ─────────────────────────────────────────────────────


@config.override_method(
    "Qwen3_5MoeForConditionalGeneration.get_parallel_plan",
    description="Register Qwen3_5Moe expert parallel plan for v5 generated modeling",
)
def qwen3_5_moe_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()
