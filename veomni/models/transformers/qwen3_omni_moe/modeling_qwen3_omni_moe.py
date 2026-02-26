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
#
# Patch for transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe
# Adds support for: Sequence Parallelism (SP), FSDP, Expert Parallelism (EP),
# Liger kernel, fused MoE, pre-computed masks,
# VeOmni data constants, and multiprocessing-compatible position ID generation.

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as hf_qwen3_omni_moe
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from ....data.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
    slice_position_embedding,
    sp_pad_and_slice,
    unpad_tensor,
)
from ....distributed.sequence_parallel.ulysses import _Gather
from ....ops import fused_moe_forward
from ....ops.fused_cross_entropy import ForCausalLMLoss
from ....utils import logging
from ..attention_utils import VARLEN_ATTENTION_TYPES


logger = logging.get_logger(__name__)


# ================================================================
# Patch: Qwen3OmniMoeVisionAttention.forward
# 1. [SP] Use VARLEN_ATTENTION_TYPES instead of hardcoded "flash_attention_2"
#    so that flash_attention_3 and other varlen implementations are also
#    handled correctly with cu_seqlens (required when SP pads cu_seqlens
#    with an extra entry for the padding chunk).
# ================================================================
def Qwen3OmniMoeVisionAttention_forward(
    self: hf_qwen3_omni_moe.Qwen3OmniMoeVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    _rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = hf_qwen3_omni_moe.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = hf_qwen3_omni_moe.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # [SP] Use VARLEN_ATTENTION_TYPES to support flash_attention_3 and other varlen implementations.
    # The varlen path passes cu_seqlens directly to the kernel; the else-branch splits by full-sequence
    # lengths which causes a size mismatch when SP pads cu_seqlens with an extra entry.
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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
            for q, k, v in zip(*splits, strict=False)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ================================================================
# Patch: Qwen3OmniMoeVisionEncoder
# 1. [SP] Slice pos_embeds and rotary position embeddings to match the SP-sharded hidden_states
# 2. [SP] Extend cu_seqlens with a padding entry when the total seq length is not divisible by sp_size
# 3. [FSDP] dummy_forward to prevent reduce-scatter hang when some ranks receive None pixel_values
# ================================================================
class Qwen3OmniMoeVisionEncoder(hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
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

        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None

        # [SP] Slice pos embedding so each SP rank's hidden_states shard gets the matching pos embedding.
        # pad_scale=4 matches the padding applied to hidden_states.
        if sp_group is not None:
            pos_embeds = sp_pad_and_slice(pos_embeds, dim=0, pad_value=0, pad_scale=4)
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
        # [SP] Capture total_seq_len from cu_seqlens before any SP slicing; equals seq_len when SP is off.
        total_seq_len = cu_seqlens[-1]
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # [SP] Slice rotary position embeddings to match the SP-sharded hidden_states.
        if sp_group is not None:
            cos, sin = position_embeddings
            cos = sp_pad_and_slice(cos, dim=0, pad_value=0, pad_scale=4)
            sin = sp_pad_and_slice(sin, dim=0, pad_value=0, pad_scale=4)
            position_embeddings = (cos, sin)

        # [SP] Append a padding entry to cu_seqlens to cover the padded tail on the last rank.
        if sp_group is not None:
            ps = get_parallel_state()
            sp_size = getattr(ps, "sp_size", 1)
            pad_seq_len = seq_len * sp_size - total_seq_len.item()
            if pad_seq_len > 0:
                new_cumsum = cu_seqlens[-1] + pad_seq_len
                cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
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

    # [FSDP] Prevent reduce-scatter hang when some ranks receive None pixel_values
    # while others receive valid pixel_values.
    def dummy_forward(self):
        if get_parallel_state().sp_enabled:
            sp_size = get_parallel_state().sp_size
            pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
            # If using SP, pixel_values is sliced but grid_thw is not
            grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
            dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        else:
            pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
            grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
            dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
        return self(**dummy_data)


# ================================================================
# Patch: Qwen3OmniMoeAudioEncoder
# 1. [SP] Gather input_features along the time dim and strip SP padding before chunking
# 2. [SP] Slice hidden_states before encoder layers; extend cu_seqlens for the padded tail
# 3. [FSDP] dummy_forward to prevent reduce-scatter hang when some ranks have no audio data
# ================================================================
class Qwen3OmniMoeAudioEncoder(hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder):
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        aftercnn_lens = hf_qwen3_omni_moe._get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        # [SP] input_features is (num_mel_bins, total_len); gather along the time dimension (dim=1)
        # and strip any SP padding before chunking.
        if get_parallel_state().sp_enabled:
            unpadded_input_len = torch.sum(chunk_lengths)
            input_features = gather_outputs(input_features, gather_dim=1, group=get_parallel_state().sp_group)
            sp_input_padding = input_features.size(1) - unpadded_input_len
            if sp_input_padding > 0:
                input_features = unpad_tensor(input_features, dim=1, padding_size=sp_input_padding)

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = hf_qwen3_omni_moe._get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        # [SP] Slice hidden_states along the seq dim (dim=0) before encoder layers.
        # Extend cu_seqlens to cover the padded tail on the last rank if needed.
        if get_parallel_state().sp_enabled:
            unpadded_hidden_len = cu_seqlens[-1]
            hidden_states = slice_input_tensor(hidden_states, dim=0, group=get_parallel_state().sp_group)
            pad_seq_len = hidden_states.size(0) * get_parallel_state().sp_size - unpadded_hidden_len
            if pad_seq_len > 0:
                cu_seqlens = torch.cat([cu_seqlens, (cu_seqlens[-1] + pad_seq_len).unsqueeze(0)], dim=0)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    # [FSDP] Prevent reduce-scatter hang when some ranks have no audio data
    # while others receive valid audio data.
    def dummy_forward(self):
        """
        Dummy forward to avoid FSDP reduce-scatter hang when some ranks have no audio data.
        input_features shape is (num_mel_bins, total_len), feature_lens is (num_audios,).
        """
        if getattr(self, "_dummy_data", None) is None:
            # Minimal valid input: one audio clip of length n_window*2 (smallest non-zero chunk)
            min_len = self.n_window * 2
            input_features = torch.zeros((self.num_mel_bins, min_len), dtype=self.dtype, device=self.device)
            feature_lens = torch.tensor([min_len], dtype=torch.long, device=self.device)
            self._dummy_data = {
                "input_features": input_features,
                "feature_lens": feature_lens,
            }
        return self(**self._dummy_data)


# ================================================================
# Patch: Qwen3OmniMoeThinkerTextModel
# 1. [SP] Slice rotary position embeddings to match the SP-sharded hidden_states
# 2. [FSDP] Handle None visual_pos_masks in _deepstack_process
# 3. [Mask] visual_pos_masks is now pre-computed without an extra trailing dim
# ================================================================
class Qwen3OmniMoeThinkerTextModel(hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # [SP] Slice rotary position embeddings so each SP rank's hidden_states shard gets the
        # matching positional encoding.
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        if sp_group is not None:
            position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self,  # noqa: PLR6301
        hidden_states,
        visual_pos_masks,
        visual_embeds,
    ):
        # [FSDP] visual_pos_masks is None when both pixel_values and pixel_values_videos are None.
        # Still touch visual_embeds so FSDP reduce-scatter stays in sync across ranks.
        if visual_pos_masks is None:
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states + visual_embeds.mean() * 0.0
            return hidden_states

        # [Mask] Mask is pre-computed in the correct 2D format; squeeze trailing dim if still 3D.
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        if visual_pos_masks.ndim == 3:
            visual_pos_masks = visual_pos_masks[..., 0]
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


# ================================================================
# Patch: Qwen3OmniMoeThinkerTextSparseMoeBlock
# 1. [MoE/eager]  Original nn.ModuleList experts; raises error if EP is enabled
# 2. [MoE/fused]  Qwen3OmniMoeThinkerExperts with stacked weights for fused MoE kernel and EP
# ================================================================


# NEW: Qwen3OmniMoeThinkerExperts
# Stacked expert weights for fused MoE forward in Qwen3-Omni-MoE thinker layers.
# Used when _moe_implementation == "fused".
class Qwen3OmniMoeThinkerExperts(nn.Module):
    """Stacked expert weights for fused MoE forward in Qwen3-Omni-MoE thinker layers.

    Stores all expert weights as 3-D tensors (num_experts, out, in) so that
    the fused MoE kernel and Expert Parallelism can operate on them directly.
    The parameter names match the merged-checkpoint convention:
      *.mlp.experts.gate_proj, *.mlp.experts.up_proj, *.mlp.experts.down_proj
    """

    def __init__(self, config) -> None:
        super().__init__()
        from transformers.activations import ACT2FN  # noqa: PLC0415

        num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
        # Shape convention (same as fused_moe_forward expectation):
        #   gate_proj / up_proj : (num_experts, intermediate_size, hidden_size)
        #   down_proj            : (num_experts, hidden_size,       intermediate_size)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        return fused_moe_forward(
            module=self,
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj,  # (num_experts, intermediate_size, hidden_size)
            fc1_2_weight=self.up_proj,  # (num_experts, intermediate_size, hidden_size)
            fc2_weight=self.down_proj,  # (num_experts, hidden_size,       intermediate_size)
        )


class Qwen3OmniMoeThinkerTextSparseMoeBlock(hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock):
    def __init__(self, config) -> None:
        # Call grandparent (nn.Module) init to avoid re-running the original __init__
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        if self._moe_implementation == "fused":
            # experts sub-module holds stacked weights; its name "experts" ensures
            # parameter FQNs align with the merged checkpoint and parallel_plan.
            self.experts = Qwen3OmniMoeThinkerExperts(config)
        elif self._moe_implementation == "eager":
            self.experts = nn.ModuleList(
                [
                    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            raise ValueError(
                f"Invalid _moe_implementation: {self._moe_implementation!r}. Expected 'eager' or 'fused'."
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self._moe_implementation == "eager":
            ps = get_parallel_state()
            if ps.ep_enabled:
                raise NotImplementedError(
                    "eager MoE does not support Expert Parallelism (EP). Set _moe_implementation='fused' to use EP."
                )
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        else:
            # fused: delegate entirely to Qwen3OmniMoeThinkerExperts
            final_hidden_states = self.experts(hidden_states, routing_weights, selected_experts, self.num_experts)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# ================================================================
# NEW: get_position_id
# Global function for multiprocessing serialization to generate position_ids
# ================================================================
def get_position_id(main_func, self, **kwargs):
    """
    This function is used during the data preprocessing stage to generate position_ids
    and associated parameters (e.g., rope_deltas) for a **single sample** (bs = 1).
    This function is a global function for multiprocessing serialization.
    Args:
        main_func: model.get_position_id
        self: An object holding model-specific information (e.g., SimpleNamespace(config=...)).
        **kwargs: Additional arguments passed to `main_func` (e.g., input_ids).
    Returns:
        dict:
            - "position_ids": Tensor of shape (dim, l), with the batch dimension squeezed.
            - other necessary parameters with the batch dimension squeezed (e.g., rope_deltas).

    Example usage:
        class Model:
            def get_position_id_func(self):  # Used in data_transform during training
                fake_model = SimpleNamespace(config=self.config)
                return partial(get_position_id, main_func, fake_model)

        model = Model()
        func = model.get_position_id_func()
        position_func_returns = func(input_ids=input_ids.unsqueeze(0), **kwargs)
        position_ids = position_func_returns['position_ids']  # shape: (dim, l)

    If a model does not implement `get_position_id_func()`, a default fallback for position_ids can be:
        position_id_returns = {
            "position_ids": torch.arange(0, len(text_inputs["input_ids"])).unsqueeze(0)  # shape: (dim, l)
        }
    """
    position_ids, rope_deltas = main_func(self, **kwargs)  # position_ids (dim, 1, l), rope_deltas (1, 1)
    assert len(position_ids.shape) == 3 and position_ids.shape[1] == 1
    assert len(rope_deltas.shape) == 2 and rope_deltas.shape[0] == 1
    return {"position_ids": position_ids.squeeze(1), "rope_deltas": rope_deltas.squeeze(0)}


# ================================================================
# Patch: Qwen3OmniMoeThinkerForConditionalGeneration
# 1. [Constants] Use VeOmni data constants for multimodal token indices
# 2. [PosID] get_position_id_func for multiprocessing data preprocessing
# 3. [Audio] get_audio_features handles flat input and SP
# 4. [Mask]  Use pre-computed image_mask/video_mask/audio_mask
# 5. [ViT]   Pop flash-attention kwargs before ViT forward
# 6. [SP]    gather_seq_scatter_heads on input/image/video/audio embeddings
# 7. [FSDP]  Dummy ViT/audio forward when pixel_values/input_features is None
# 8. [SP]    gather_heads_scatter_seq after multimodal merging
# 9. [SP]    all_gather deepstack embeddings then select per-rank visual token slice
# 10.[Loss]  Delegate loss to ForCausalLMLoss
# 11.[PosIDs] Transpose pre-computed position_ids from (bs, 3, L) to (3, bs, L)
# 12.[RoPE]  get_rope_index supports per-video use_audio_in_video via audio_seqlens
# ================================================================
class Qwen3OmniMoeThinkerForConditionalGeneration(hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # [Constants] Use VeOmni data constants for multimodal token indices.
        self.image_token_index = IMAGE_INPUT_INDEX
        self.video_token_index = VIDEO_INPUT_INDEX
        self.audio_token_index = AUDIO_INPUT_INDEX

    def get_position_id_func(self):
        fake_model = SimpleNamespace(
            config=self.config,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            audio_token_index=self.audio_token_index,
            spatial_merge_size=self.spatial_merge_size,
            get_llm_pos_ids_for_vision=partial(
                hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision, None
            ),
            get_chunked_index=partial(
                hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_chunked_index, None
            ),
        )
        return partial(
            get_position_id,
            Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index,
            fake_model,
        )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [RoPE] Patch: support mixed data of video_w_audio & video_w/o_audio.
        # The HF implementation uses a global `use_audio_in_video` flag, which cannot handle
        # a batch where some videos have audio and others don't.  We perform the same per-video
        # check used in Qwen2.5-Omni: audio_seqlens[audio_idx] == 0 means no audio for that
        # video (the placeholder entry is consumed here to keep audio_idx aligned).
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    # Audio in Video (bos is shared: vision_start immediately followed by audio_start)
                    if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = hf_qwen3_omni_moe._get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only (token-level) — audio track determined per-video via audio_seqlens
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                        # --- Patch: support mixed data of video_w_audio & video_w/o_audio ---
                        # Determine per-video whether this video has an audio track.
                        # audio_seqlens[audio_idx] == 0 means no audio for this video;
                        # we consume the zero-length placeholder to keep audio_idx aligned.
                        if audio_seqlens is not None:
                            if audio_seqlens[audio_idx] == 0:
                                use_audio_in_video = False
                                audio_idx += 1  # consume zero-length placeholder
                            else:
                                use_audio_in_video = True
                        else:
                            use_audio_in_video = False
                        # --- Patch end ---

                        if not use_audio_in_video:
                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t)
                                * second_per_grids[video_idx].cpu().float()
                                * position_id_per_seconds
                            ).float()
                            llm_pos_ids = self.get_llm_pos_ids_for_vision(
                                st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                            )
                            video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                            llm_pos_ids_list.append(llm_pos_ids)

                            st += int(text_len + bos_len + video_len + eos_len)
                            video_idx += 1
                            remain_videos -= 1
                        else:
                            audio_len = hf_qwen3_omni_moe._get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                            audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t)
                                * second_per_grids[video_idx].cpu().float()
                                * position_id_per_seconds
                            ).float()
                            video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                                st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                            )
                            video_data_index, audio_data_index = 0, 0
                            while (
                                video_data_index < video_llm_pos_ids.shape[-1]
                                and audio_data_index < audio_llm_pos_ids.shape[-1]
                            ):
                                if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                    llm_pos_ids_list.append(
                                        video_llm_pos_ids[:, video_data_index : video_data_index + 1]
                                    )
                                    video_data_index += 1
                                else:
                                    llm_pos_ids_list.append(
                                        audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1]
                                    )
                                    audio_data_index += 1
                            if video_data_index < video_llm_pos_ids.shape[-1]:
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                                )
                            if audio_data_index < audio_llm_pos_ids.shape[-1]:
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                                )
                            video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                            st += int(text_len + bos_len + audio_len + video_len + eos_len)
                            audio_idx += 1
                            video_idx += 1
                            remain_videos -= 1
                            remain_audios -= 1

                    # Audio in Video (token-level: <vision_start><audio_start>...) — HF compatibility
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        audio_len = hf_qwen3_omni_moe._get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st += int(text_len + bos_len + audio_len + video_len + eos_len)
                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            # Unpack into flat (num_mel_bins, total_len) format expected by audio_tower; SP is handled inside.
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            # Compatibility: handle pre-processed (total_len, num_mel_bins) input without mask.
            if input_features.ndim == 2 and input_features.size(-1) == self.audio_tower.num_mel_bins:
                input_features = input_features.transpose(0, 1)  # -> (num_mel_bins, total_len)

        if audio_feature_lengths is not None:
            feature_lens = audio_feature_lengths
        else:
            feature_lens = torch.tensor([input_features.size(1)], dtype=torch.long, device=input_features.device)
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features

    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: Optional[bool] = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> Union[tuple, hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # [Mask] Use pre-computed masks to avoid all-gather for complete mask info when using SP.
        assert "image_mask" in kwargs, "image_mask should have already been computed in process_sample"
        assert "video_mask" in kwargs, "video_mask should have already been computed in process_sample"
        assert "audio_mask" in kwargs, "audio_mask should have already been computed in process_sample"
        image_mask = kwargs.pop("image_mask")
        video_mask = kwargs.pop("video_mask")
        audio_mask = kwargs.pop("audio_mask")

        # [ViT] Pop flash-attention kwargs before ViT forward. ViT computes its own cu_seqlens from
        # grid_thw and must not receive the LLM-level seqlen kwargs:
        # https://github.com/huggingface/transformers/blob/94df0e65602922be2831b3faa457a2bde78b936b/src/transformers/modeling_flash_attention_utils.py#L432-L450
        flash_attn_kwargs = {}
        for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
            if key in kwargs:
                flash_attn_kwargs[key] = kwargs.pop(key)

        # [SP] Gather seq and scatter heads on inputs_embeds so multimodal fill-back operates on the
        # full sequence: (batch_size, seq_len // sp_size, hidden_size) -> (batch_size, seq_len, hidden_size // sp_size)
        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_seq_scatter_heads(
                inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
            )

        # 2. Merge text, audios, image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # [SP] audio_tower returns seq-sliced features; gather seq and scatter heads to match
            # inputs_embeds layout before fill-back.
            if self.training and get_parallel_state().sp_enabled:
                audio_features = gather_seq_scatter_heads(
                    audio_features, seq_dim=0, head_dim=1, group=get_parallel_state().sp_group
                )
            # Drop any padding tokens beyond the actual audio placeholder count.
            n_audio_tokens = audio_mask.sum().long().item()
            audio_features = audio_features[:n_audio_tokens]
            audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_expanded, audio_features)
        elif get_parallel_state().fsdp_enabled:
            # [FSDP] Dummy audio tower forward to keep reduce-scatter in sync when some ranks
            # have no audio data while others do.
            fake_audio = self.audio_tower.dummy_forward().last_hidden_state.mean() * 0.0
            fake_audio = fake_audio.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_audio

        # Initialize fake_deepstack to None
        fake_deepstack = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # [SP] Gather seq and scatter heads on image_embeds:
            # (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size // sp_size)
            if self.training and get_parallel_state().sp_enabled:
                image_embeds = gather_seq_scatter_heads(
                    image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )

            # [Mask] Get token count from pre-computed mask and expand to inputs_embeds shape.
            n_image_tokens = image_mask.sum().long().item()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)

            # Drop any padded image tokens beyond the actual placeholder count.
            image_embeds = image_embeds[:n_image_tokens]
            deepstack_image_embeds = [embed[:n_image_tokens] for embed in deepstack_image_embeds]
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        elif get_parallel_state().fsdp_enabled:
            # [FSDP] Dummy ViT forward to keep reduce-scatter in sync when some ranks receive
            # None pixel_values while others receive valid pixel_values.
            fake_embeds, fake_deepstack = self.visual.dummy_forward()
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds

        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # [SP] Gather seq and scatter heads on video_embeds:
            # (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size // sp_size)
            if self.training and get_parallel_state().sp_enabled:
                video_embeds = gather_seq_scatter_heads(
                    video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
                )

            # [Mask] Get token count from pre-computed mask and expand to inputs_embeds shape.
            n_video_tokens = video_mask.sum().long().item()
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)

            # Drop any padded video tokens beyond the actual placeholder count.
            video_embeds = video_embeds[:n_video_tokens]
            deepstack_video_embeds = video_embeds_multiscale
            deepstack_video_embeds = [embed[:n_video_tokens] for embed in deepstack_video_embeds]
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        elif get_parallel_state().fsdp_enabled:
            # [FSDP] Dummy ViT forward to keep reduce-scatter in sync when some ranks receive
            # None pixel_values_videos while others receive valid pixel_values_videos.
            fake_embeds, fake_deepstack = self.visual.dummy_forward()
            fake_embeds = fake_embeds.mean() * 0.0
            fake_embeds = fake_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + fake_embeds

        # Prepare rank-local masks for deepstack use (set after SP scatter below).
        rank_image_mask = None
        rank_video_mask = None

        # [SP] Restore seq-parallel layout after fill-back:
        # (batch_size, seq_len, hidden_size // sp_size) -> (batch_size, seq_len // sp_size, hidden_size)
        if self.training and get_parallel_state().sp_enabled:
            inputs_embeds = gather_heads_scatter_seq(
                inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
            )

            # [SP] all_gather deepstack embeddings and select the per-rank visual token slice
            # using the full-sequence mask.
            sp_size = get_parallel_state().sp_size
            sp_rank = get_parallel_state().sp_rank

            if pixel_values is not None:
                # all_gather: (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size)
                deepstack_image_embeds = [
                    _Gather.apply(get_parallel_state().sp_group, embed, 0, False) for embed in deepstack_image_embeds
                ]

                # image_mask is (batch_size, seq_len, hidden_size // sp_size) at this point.
                image_mask_1d = image_mask[..., 0]  # (batch_size, seq_len)
                seq_len = image_mask_1d.shape[1]
                seq_per_rank = seq_len // sp_size
                rank_start = sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                rank_image_mask = image_mask_1d[:, rank_start:rank_end]  # (batch_size, seq_len // sp_size)
                offset = image_mask_1d[:, :rank_start].sum().item()
                num_visual_tokens = rank_image_mask.sum().item()
                deepstack_image_embeds = [
                    embed[offset : offset + num_visual_tokens] for embed in deepstack_image_embeds
                ]

            if pixel_values_videos is not None:
                # all_gather: (seq_len // sp_size, hidden_size) -> (seq_len, hidden_size)
                deepstack_video_embeds = [
                    _Gather.apply(get_parallel_state().sp_group, embed, 0, False) for embed in deepstack_video_embeds
                ]

                video_mask_1d = video_mask[..., 0]  # (batch_size, seq_len)
                seq_len = video_mask_1d.shape[1]
                seq_per_rank = seq_len // sp_size
                rank_start = sp_rank * seq_per_rank
                rank_end = rank_start + seq_per_rank

                rank_video_mask = video_mask_1d[:, rank_start:rank_end]
                offset = video_mask_1d[:, :rank_start].sum().item()
                num_visual_tokens = rank_video_mask.sum().item()
                deepstack_video_embeds = [
                    embed[offset : offset + num_visual_tokens] for embed in deepstack_video_embeds
                ]

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values is not None and pixel_values_videos is not None:
            # Both image and video: merge masks and interleave deepstack embeddings.
            # Reuse the rank-local sliced masks when SP is active.
            image_mask = rank_image_mask if rank_image_mask is not None else image_mask[..., 0]
            video_mask = rank_video_mask if rank_video_mask is not None else video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif pixel_values is not None:
            image_mask = rank_image_mask if rank_image_mask is not None else image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif pixel_values_videos is not None:
            video_mask = rank_video_mask if rank_video_mask is not None else video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        else:
            # [FSDP] No visual input: still pass fake_deepstack so _deepstack_process is called on all
            # ranks (visual_pos_masks=None makes it a no-op that keeps reduce-scatter in sync).
            if fake_deepstack is not None:
                deepstack_visual_embeds = fake_deepstack

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids is not None:
            # [PosIDs] During training with rmpad_with_pos_ids, position_ids are computed per sample as
            # (3, L) and collated to (bs, 3, L). Transpose to (3, bs, L) as the model expects.
            if position_ids.ndim == 3 and position_ids.shape[1] == 3:
                position_ids = position_ids.transpose(0, 1).contiguous()  # (bs, 3, L) -> (3, bs, L)

        # [ViT] Restore flash-attention kwargs for the language model forward.
        kwargs.update(flash_attn_kwargs)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )

        hidden_states = outputs[0]
        # [Loss] Delegate to ForCausalLMLoss which handles label shifting (non-SP), Liger/fused kernel
        # selection, and SP loss reduction.
        # NOTE: ForCausalLMLoss pads labels to (S+1) then slices [..., 1:] back to length S, so
        # shift_labels stays at length S — matching the full, unsliced hidden_states. Do NOT
        # pre-slice hidden_states[..., :-1, :] here or the shapes will mismatch.
        if labels is not None:
            loss, logits = ForCausalLMLoss(
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                ignore_index=IGNORE_INDEX,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        aux_loss = None
        if output_router_logits:
            aux_loss = hf_qwen3_omni_moe.load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


# ================================================================
# Patch: Qwen3OmniMoeForConditionalGeneration
# 1. [MoE] Propagate _moe_implementation from top-level config to thinker/text configs
# 2. Simplified forward for training: only forward thinker, skip talker/code2wav
# ================================================================
class Qwen3OmniMoeForConditionalGeneration(hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration):
    def __init__(self, config):
        # [MoE] Propagate _moe_implementation so SparseMoeBlock picks up the correct mode.
        moe_implementation = getattr(config, "_moe_implementation", "eager")
        config.thinker_config._moe_implementation = moe_implementation
        config.thinker_config.text_config._moe_implementation = moe_implementation
        super().__init__(config)

    def forward(
        self,
        **kwargs,
    ) -> Union[tuple, hf_qwen3_omni_moe.Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        thinker_outputs = self.thinker(
            **kwargs,
        )
        # TODO: talker_outputs
        return thinker_outputs


def _get_parallel_plan(_self):
    from .parallel_plan import get_parallel_plan

    return get_parallel_plan()


# ================================================================
# apply_veomni_qwen3_omni_moe_patch
# Central entry point to apply all VeOmni patches to HF Qwen3OmniMoe classes
# ================================================================
def apply_veomni_qwen3_omni_moe_patch():
    logger.info_rank0("Apply VeOmni patch to Qwen3_Omni_MoE.")

    # Fix _no_split_modules: use the correct decoder layer class name
    hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModel._no_split_modules = [
        "Qwen3OmniMoeThinkerTextDecoderLayer",
        "Qwen3OmniMoeVisionBlock",
    ]

    # Patch parallel plan support
    hf_qwen3_omni_moe.Qwen3OmniMoePreTrainedModel.get_parallel_plan = _get_parallel_plan

    # Patch VisionAttention forward
    hf_qwen3_omni_moe.Qwen3OmniMoeVisionAttention.forward = Qwen3OmniMoeVisionAttention_forward

    # Replace classes with VeOmni subclasses
    hf_qwen3_omni_moe.Qwen3OmniMoeVisionEncoder = Qwen3OmniMoeVisionEncoder
    hf_qwen3_omni_moe.Qwen3OmniMoeAudioEncoder = Qwen3OmniMoeAudioEncoder
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextModel = Qwen3OmniMoeThinkerTextModel
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock
    hf_qwen3_omni_moe.Qwen3OmniMoeThinkerForConditionalGeneration = Qwen3OmniMoeThinkerForConditionalGeneration
    hf_qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration = Qwen3OmniMoeForConditionalGeneration
