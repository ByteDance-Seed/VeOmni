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
"""Qwen4-Exp NPU patches with opt-in Triton Ascend QSA token selection.

Regenerate with:
patchgen veomni.models.transformers.qwen4_exp.qwen4_exp_npu_patch_gen_config --diff

The shared GPU integration stays unchanged. QSA dispatch and packed metadata
are added only to the generated NPU model.
"""

from copy import deepcopy
from typing import Callable

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen4ExpModelOutputWithPast,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, torch_compilable_check

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import (
    _qwen4_exp_validate_packed_seq_lens,
)
from veomni.models.transformers.qwen4_exp.qwen4_exp_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.ops.kernels.qsa.eager import qsa_indexer_forward_eager
from veomni.utils.seqlen_pos_transform_utils import culen2pos, pos2culen


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_qwen4_exp_npu.py"
config.description = "Qwen4-Exp NPU VLM-SFT integration with PLE sharding and optional Triton QSA"
# Replace the shared model forward only on NPU to reuse packed lengths across QSA layers.
config.patches = [patch for patch in config.patches if patch.target != "Qwen4ExpModel.forward"]
config.add_import("veomni.ops.kernels.qsa.eager", names=["qsa_indexer_forward_eager"])
config.add_post_import_block('veomni_qsa_indexer = OpSlot("qsa_indexer", "standard")')

veomni_qsa_indexer = None


@config.override_method(
    "Qwen4ExpTextQSAIndexer.forward",
    description="Dispatch Qwen4-Exp QSA through the VeOmni op slot",
)
def qwen4_exp_text_qsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Cache | None,
    cu_seq_lens_q: torch.Tensor | None = None,
    qsa_packed_seq_lens=None,
) -> torch.Tensor:
    if veomni_qsa_indexer.use_non_eager_impl:
        return veomni_qsa_indexer(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cu_seq_lens_q=cu_seq_lens_q,
            qsa_packed_seq_lens=qsa_packed_seq_lens,
        )
    return qsa_indexer_forward_eager(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cu_seq_lens_q=cu_seq_lens_q,
    )


@config.override_method(
    "Qwen4ExpTextAttention.forward",
    description="Pass packed sequence boundaries to the Qwen4-Exp QSA indexer",
)
def qwen4_exp_text_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # --- Patch.1 ---
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
    qsa_packed_seq_lens = kwargs.get("qsa_packed_seq_lens")
    selected_token_mask = self.indexer(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cu_seq_lens_q=cu_seq_lens_q,
        qsa_packed_seq_lens=qsa_packed_seq_lens,
    )
    # --- Patch.1 ---
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
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
    return self.o_proj(attn_output), attn_weights


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
    packed_seq_lens = _qwen4_exp_validate_packed_seq_lens(
        cu_seq_lens_q.diff().tolist(), inputs_embeds.shape[0], inputs_embeds.shape[1]
    )
    kwargs.setdefault("qsa_packed_seq_lens", packed_seq_lens)
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
