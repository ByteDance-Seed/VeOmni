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

# Patch https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py

import copy
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as hf_qwen3vlmoe
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    load_balancing_loss_func,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as _Qwen3VLMoeForConditionalGeneration,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel as _Qwen3VLMoeModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextExperts as _Qwen3VLMoeTextExperts,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs

from ....data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    sp_pad_and_slice,
)
from ....ops import fused_moe_forward
from ....utils import logging
from ....utils.data_balance.data_balance import Qwen3VLEncoderDataBalance
from ....utils.device import is_torch_npu_available
from ..attention_utils import VARLEN_ATTENTION_TYPES


logger = logging.get_logger(__name__)


# ================================================================
# Patch: Qwen3VLMoeTextExperts
# 1. add fused MoE forward for better performance and enable EP
# ================================================================
class Qwen3VLMoeTextExperts(_Qwen3VLMoeTextExperts):
    def __init__(self, config):
        super().__init__(config)
        self.moe_implementation = getattr(config, "_moe_implementation", "eager")

    # --- Patch.1 ---
    def fused_moe_forward(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        # we need compact top-k routing weights for fused implementation
        assert routing_weights is not None, "routing_weights must be provided for fused implementation"

        # Split gate_up_proj into gate_proj and up_proj
        gate_proj = self.gate_up_proj[..., : self.expert_dim]
        up_proj = self.gate_up_proj[..., self.expert_dim :]
        # Transpose weights to match expected shape for fused_moe_forward
        # Expected: fc1_1/fc1_2: (num_experts, intermediate_size, hidden_dim)
        #          fc2: (num_experts, hidden_dim, intermediate_size)
        # Current: gate_proj/up_proj: (num_experts, hidden_size, expert_dim)
        #         down_proj: (num_experts, expert_dim, hidden_size)
        gate_proj_t = gate_proj.transpose(1, 2).contiguous()  # (num_experts, expert_dim, hidden_size)
        up_proj_t = up_proj.transpose(1, 2).contiguous()  # (num_experts, expert_dim, hidden_size)
        down_proj_t = self.down_proj.transpose(1, 2).contiguous()  # (num_experts, hidden_size, expert_dim)

        next_states = fused_moe_forward(
            num_experts=self.num_experts,
            routing_weights=routing_weights,  # Use compact top-k weights
            selected_experts=router_indices,
            hidden_states=hidden_states,
            fc1_1_weight=gate_proj_t,
            fc1_2_weight=up_proj_t,
            fc2_weight=down_proj_t,
        )
        # Reshape output to match eager mode: (batch_size, seq_len, hidden_size)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

    # --- Patch.1 ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.training and self.moe_implementation == "fused":
            return self.fused_moe_forward(hidden_states, router_weights, router_indices, routing_weights)
        else:
            return super().forward(hidden_states, router_weights, router_indices)


# ================================================================
# Patch: Qwen3VLMoeTextSparseMoeBlock.forward
# 1. parse routing weights for fused MoE forward
# ================================================================
def Qwen3VLMoeTextSparseMoeBlock_forward(
    self: Qwen3VLMoeTextSparseMoeBlock, hidden_states: torch.Tensor
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    router_logits = self.gate(hidden_states)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
    hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
    # --- Patch.1 ---
    routed_out = self.experts(hidden_states, router_weights, router_indices, routing_weights)
    # --- Patch.1 ---
    return routed_out, router_logits


# ================================================================
# Patch: Qwen3VLMoePreTrainedModel.get_parallel_plan
# 1. add parallel plan for expert parallelism
# ================================================================
# --- Patch.1 ---
def get_parallel_plan(self):
    from .parallel_plan import get_parallel_plan

    return get_parallel_plan()


# --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLMoeVisionAttention.forward
# 1. add flash_attention_3 support
# 2. use precomputed max_seqlen in advance to avoid per-layer cpu-gpu sync
# ================================================================
def Qwen3VLMoeVisionAttention_forward(
    self: Qwen3VLMoeVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- Patch.1 ---
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        # --- Patch.1 ---
        # Flash Attention 2: Use cu_seqlens for variable length attention
        # --- Patch.2 ---
        # max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        # --- Patch.2 ---
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
        # Other implementations: Process each chunk separately
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


# ================================================================
# Patch: Qwen3VLMoeVisionModel.dummy_forward
# 1. add dummy_forward to avoid FSDP reduce-scatter hang when some ranks
# get None pixel_values while others get valid pixel_values
# ================================================================
# --- Patch.1 ---
def Qwen3VLMoeVisionModel_dummy_forward(
    self: Qwen3VLMoeVisionModel,
    encoder_data_balance: Qwen3VLEncoderDataBalance = None,
):
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        # If using SP, pixel_values is sliced but grid_thw is not
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)

        if encoder_data_balance is not None:
            # add dummy data to avoid encoder data balance communication hang
            pixel_values, grid_thw = encoder_data_balance.balance_data(pixel_values, grid_thw)
            dummy_image_embeds, dummy_deepstack_image_embeds = self(hidden_states=pixel_values, grid_thw=grid_thw)
            dummy_image_embeds, dummy_deepstack_image_embeds = encoder_data_balance.data_bridge(
                dummy_image_embeds, dummy_deepstack_image_embeds
            )
            return dummy_image_embeds, dummy_deepstack_image_embeds

        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    return self(**dummy_data)


# --- Patch.1 ---


# ================================================================
# Patch: Qwen3VLMoeVisionModel.forward
# 1. slice pos embedding if using sp to let sharded hidden_states get its corresponding pos embedding
# 2. get before-sliced full seq from cu_seqlens
# 3. slice pos embedding when using sp to let sp-sliced hidden_states get its corresponding pos embedding
# 4. pad cu_seqlens when using sp to match the padded hidden_states
# 5. calculate max_seqlen from cu_seqlens here to avoid per layer CPU-GPU sync
# 6. move cu_seqlens to cpu when using NPU to avoid per layer CPU-GPU sync when using FA
# ================================================================
def Qwen3VLMoeVisionModel_forward(
    self: Qwen3VLMoeVisionModel, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    # --- Patch.1 ---
    if get_parallel_state().sp_enabled:
        pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
    # --- Patch.1 ---

    hidden_states = hidden_states + pos_embeds

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    # --- Patch.2 ---
    rotary_pos_emb = rotary_pos_emb.reshape(cu_seqlens[-1], -1)
    # --- Patch.2 ---
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    if get_parallel_state().sp_enabled:
        cos, sin = position_embeddings
        # --- Patch.3 ---
        cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
        sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
        position_embeddings = (cos, sin)
        # --- Patch.3 ---

        # --- Patch.4 ---
        sp_size = get_parallel_state().sp_size
        # Calculate the last one padding seq_len : seq_len * sp_size - total_seq_len
        pad_seq_len = seq_len * sp_size - cu_seqlens[-1].item()
        if pad_seq_len > 0:
            # Add this extra sequence to cu_seqlens with the padding length
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.4 ---

    deepstack_feature_lists = []
    # --- Patch.5 ---
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().detach().cpu().item()
    # --- Patch.5 ---
    # --- Patch.6 ---
    if is_torch_npu_available():
        cu_seqlens = cu_seqlens.cpu()
    # --- Patch.6 ---
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)

    return hidden_states, deepstack_feature_lists


# ================================================================
# Patch: Qwen3VLMoeTextModel._deepstack_process
# 1. Handle case when visual_pos_masks is None (both image and video pixel_values are None)
# Still call this operation but just add 0.0 to hidden_states to avoid FSDP reduce scatter stuck
# ================================================================
def Qwen3VLMoeTextModel__deepstack_process(
    self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
):
    # --- Patch.1 ---
    if visual_pos_masks is None:
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + visual_embeds.mean() * 0.0
        return hidden_states
    # --- Patch.1 ---

    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this
    return hidden_states


# ================================================================
# Patch: Qwen3VLMoeModel
# 1. set moe_implementation
# 2. skip torch.split in get_image_features
# 3. patch get_placeholder_mask for veomni usage
# 4. sequence parallel forward for sp sliced input_embeds & image_mask
# & video_mask $ deepstack embeds
# 5. dummy forward patch
# 6. handle precomputed position_ids with shape (bs, dim, seq_len)
# 7. Use precomputed flash attention kwargs to avoid CPU-GPU sync
# ================================================================
@auto_docstring
class Qwen3VLMoeModel(_Qwen3VLMoeModel):
    def __init__(self, config):
        # --- Patch.1 ---
        moe_implementation = getattr(config, "_moe_implementation", "eager")
        config.text_config._moe_implementation = moe_implementation
        # --- Patch.1 ---
        super().__init__(config)

        if config.encoder_data_balance:
            self.encoder_data_balance = Qwen3VLEncoderDataBalance(
                sorting_algo_name=config.encoder_data_balance_sorting_algo,
                spatial_merge_unit=self.visual.spatial_merge_unit,
            )
        else:
            self.encoder_data_balance = None

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        if self.encoder_data_balance is not None:
            # balance encoder's data
            pixel_values, image_grid_thw = self.encoder_data_balance.balance_data(pixel_values, image_grid_thw)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        if self.encoder_data_balance is not None:
            # recover the data distribution to fit subsequent process
            image_embeds, deepstack_image_embeds = self.encoder_data_balance.data_bridge(
                image_embeds, deepstack_image_embeds
            )
        # --- Patch.2 ---
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        # --- Patch.2 ---
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(self, input_ids: torch.LongTensor, **kwargs):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        # TODO(szl): check token_id with verl rollout tensor_dict
        # --- Patch.3 ---
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        # --- Patch.3 ---
        return special_image_mask, special_video_mask

    @auto_docstring
    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # --- Patch.4 ---
        # we use the pre-computed image and video mask to support ulysses
        image_mask = kwargs.get("image_mask", None)
        video_mask = kwargs.get("video_mask", None)

        # if None, all gather sp group input_ids and calculate mask
        if video_mask is None and image_mask is None:
            input_ids_list = [torch.zeros_like(input_ids) for i in range(get_parallel_state().sp_size)]
            dist.all_gather(input_ids_list, input_ids, group=get_parallel_state().sp_group)
            image_mask, video_mask = self.get_placeholder_mask(torch.cat(input_ids_list, dim=0))
        # --- Patch.4 ---

        # --- Patch.7 ---
        # Pop flash attention kwargs for ViT, they should only be used for language model
        # Qwen3L ViT input images seq lens should be computed during ViT forward using grid_thw
        text_flash_attn_kwargs = {}
        for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
            if key in kwargs:
                text_flash_attn_kwargs[key] = kwargs.pop(key)
        # --- Patch.7 ---

        # --- Patch.4 ---
        if get_parallel_state().sp_enabled:
            # (batch_size, seq_len // sp_size, hidden_size) to  (batch_size, seq_len, hidden_size // sp_size)
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )
        # --- Patch.4 ---

        # Initialize fake_deepstack to None
        fake_deepstack = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            # --- Patch.4 ---
            # sequence parallel patch for image_embeds
            if get_parallel_state().sp_enabled:
                # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
                image_embeds = gather_seq_scatter_heads(
                    image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
                deepstack_image_embeds = [
                    gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                    for embed in deepstack_image_embeds
                ]
            n_image_tokens = image_mask.sum().long().item()
            embeds_image_mask = (
                image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
            )

            # Slice tensor to drop any padded image tokens
            image_embeds = image_embeds[:n_image_tokens]
            deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

            # sequence parallel patch for image_mask & deepstack_image_embeds
            if get_parallel_state().sp_enabled:
                seq_len = image_mask.shape[1]

                seq_per_rank = seq_len // get_parallel_state().sp_size
                rank_start = get_parallel_state().sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                deepstack_offset = image_mask[:, :rank_start].sum().item()
                image_mask = image_mask[:, rank_start:rank_end]
                deepstack_len = image_mask.sum().item()

                deepstack_image_embeds = [
                    embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_image_embeds
                ]
            # --- Patch.4 ---

        elif get_parallel_state().fsdp_enabled or self.encoder_data_balance is not None:
            # --- Patch.5 ---
            # add dummy ViT forward to avoid FSDP reduce-scatter hang and encoder data balance communication hang
            # when some ranks get None pixel_values while others get valid pixel_values
            fake_embeds, fake_deepstack = self.visual.dummy_forward(encoder_data_balance=self.encoder_data_balance)
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.5 ---

        if pixel_values_videos is not None:
            # --- Patch.4 ---
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # sequence parallel patch for video embeds
            if get_parallel_state().sp_enabled:
                # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
                video_embeds = gather_seq_scatter_heads(
                    video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )
                deepstack_video_embeds = [
                    gather_outputs(embed, gather_dim=0, group=get_parallel_state().sp_group)
                    for embed in deepstack_video_embeds
                ]
            n_video_tokens = video_mask.sum().long().item()
            embeds_video_mask = (
                video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
            )

            # Slice tensor to drop any padded video tokens
            video_embeds = video_embeds[:n_video_tokens]
            deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

            # sequence parallel patch for video_mask & deepstack_video_embeds
            if get_parallel_state().sp_enabled:
                seq_len = video_mask.shape[1]

                seq_per_rank = seq_len // get_parallel_state().sp_size
                rank_start = get_parallel_state().sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                deepstack_offset = video_mask[:, :rank_start].sum().item()
                video_mask = video_mask[:, rank_start:rank_end]
                deepstack_len = video_mask.sum().item()

                deepstack_video_embeds = [
                    embed[deepstack_offset : deepstack_offset + deepstack_len] for embed in deepstack_video_embeds
                ]
            # --- Patch.4 ---
        elif get_parallel_state().fsdp_enabled or self.encoder_data_balance is not None:
            # --- Patch.5 ---
            # add dummy ViT forward to avoid FSDP reduce-scatter hang
            # when some ranks get None pixel_values_videos while others get valid pixel_values_videos
            fake_embeds, fake_deepstack = self.visual.dummy_forward(encoder_data_balance=self.encoder_data_balance)
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds
            # --- Patch.5 ---

        # --- Patch.4 ---
        if get_parallel_state().sp_enabled:
            # (batch_size, seq_len, hidden_size // sp_size) back to (batch_size, seq_len // sp_size, hidden_size)
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )
        # --- Patch.4 ---

        visual_pos_masks = None
        deepstack_visual_embeds = None

        # use pixel_values and pixel_values_videos instead of masks
        if pixel_values is not None and pixel_values_videos is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif pixel_values is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif pixel_values_videos is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        else:
            # Modification:
            # Both pixel_values and pixel_values_videos are None
            # still pass fake_deepstack to language_model to trigger _deepstack_process
            # to avoid FSDP backward reduce scatter stuck
            # visual_pos_masks remains None, so _deepstack_process will just add 0.0
            if fake_deepstack is not None:
                deepstack_visual_embeds = fake_deepstack
            else:
                deepstack_visual_embeds = None

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        else:
            # --- Patch.6 ---
            if position_ids.dim() == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()
            # --- Patch.6 ---

        if get_parallel_state().sp_enabled:
            position_ids = sp_pad_and_slice(position_ids, dim=-1)

        # --- Patch.7 ---
        kwargs.update(text_flash_attn_kwargs)
        # --- Patch.7 ---
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


# ================================================================
# Patch: Qwen3VLMoeForConditionalGeneration
# 1. wrapped Qwen3VLMoeModel.get_rope_index to use in process_sample for obtaining position_ids in advance
# 2. use the unified loss function to handle Ulysses internally to reduce redudnecy code
# 3. overwrite token ids with veomni constants
# ================================================================


# --- Patch.1 ---
def get_position_id(main_func, self, **kwargs):
    # must be a global func for multiproceesing serialize
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


# --- Patch.1 ---


class Qwen3VLMoeForConditionalGeneration(_Qwen3VLMoeForConditionalGeneration):
    # --- Patch.1 ---
    def get_position_id_func(self):
        fake_config = copy.copy(self.config)
        # --- Patch.3 ---
        fake_config.image_token_id = IMAGE_INPUT_INDEX
        fake_config.video_token_id = VIDEO_INPUT_INDEX
        # --- Patch.3 ---
        fake_model = SimpleNamespace(config=fake_config)
        return partial(get_position_id, Qwen3VLMoeModel.get_rope_index, fake_model)

    # --- Patch.1 ---

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        >>> model = Qwen3VLMoeForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image in short."},
                ],
            }
        ]

        >>> # Preparation for inference
        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        >>> inputs = inputs.to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        >>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
        ```"""

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

        # --- Patch.2 ---
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
        # --- Patch.2 ---

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )


def apply_veomni_qwen3vlmoe_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen3_VL_MoE.")

    hf_qwen3vlmoe.Qwen3VLMoeForConditionalGeneration = Qwen3VLMoeForConditionalGeneration
    hf_qwen3vlmoe.Qwen3VLMoeModel = Qwen3VLMoeModel
    hf_qwen3vlmoe.Qwen3VLMoeTextModel._deepstack_process = Qwen3VLMoeTextModel__deepstack_process
    hf_qwen3vlmoe.Qwen3VLMoeVisionModel.forward = Qwen3VLMoeVisionModel_forward
    hf_qwen3vlmoe.Qwen3VLMoeVisionModel.dummy_forward = Qwen3VLMoeVisionModel_dummy_forward
    hf_qwen3vlmoe.Qwen3VLMoeVisionAttention.forward = Qwen3VLMoeVisionAttention_forward
    hf_qwen3vlmoe.Qwen3VLMoePreTrainedModel.get_parallel_plan = get_parallel_plan
    hf_qwen3vlmoe.Qwen3VLMoeTextSparseMoeBlock.forward = Qwen3VLMoeTextSparseMoeBlock_forward
    hf_qwen3vlmoe.Qwen3VLMoeTextExperts = Qwen3VLMoeTextExperts
