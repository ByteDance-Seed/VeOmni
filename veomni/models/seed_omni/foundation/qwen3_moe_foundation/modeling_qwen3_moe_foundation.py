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

from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack

from .....data.constants import AUDIO_INPUT_INDEX, IGNORE_INDEX, IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from .....distributed.parallel_state import get_parallel_state
from .....distributed.sequence_parallel import reduce_sequence_parallel_loss
from .....utils import logging
from .....utils.import_utils import is_liger_kernel_available
from ....transformers.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    apply_multimodal_rotary_pos_emb,
)
from ....transformers.qwen3_moe import modeling_qwen3_moe as qwen3_moe
from ....transformers.qwen3_moe.modeling_qwen3_moe import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    load_balancing_loss_func,
)
from ....transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer as OriginalQwen3MoeDecoderLayer
from ..base import BaseFoundationModelMixin
from .configuration_qwen3_moe_foundation import Qwen3MoeFoundationConfig


if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore
logger = logging.get_logger(__name__)


def parse_position_id_kwargs(input_ids: torch.Tensor, attention_mask: torch.Tensor, grid_thw: Dict = {}, **kwargs):
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "image" in grid_thw:
        return_dict["image_grid_thw"] = grid_thw["image"]
    if "video" in grid_thw:
        return_dict["video_grid_thw"] = grid_thw["video"]
        return_dict["second_per_grids"] = torch.tensor([1.0] * len(grid_thw["video"])).to(input_ids.device)
    if "feature_lengths" in kwargs:  # audio 'grid_thw'
        return_dict["audio_seqlens"] = kwargs["feature_lengths"]["audio"]
    return return_dict


def prepare_fa2_from_position_ids(position_ids):
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = cu_seq_lens.diff().max()

    return (indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


class Qwen3MoeFoundationRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3MoeDecoderLayer(OriginalQwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeFoundationConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            position_ids[0]
        )  # remove multimodal channel dimension
        cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
        max_length_q, max_length_k = max_seq_lens
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MultimodalQwen3MoeModel(Qwen3MoeModel):
    def __init__(self, config):
        Qwen3MoePreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
            ]  # overwrite decoder forward
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.rope_type == "3d_rope":
            self.rotary_emb = Qwen3MoeFoundationRotaryEmbedding(config=config)  # overwrite position embedding
        else:
            self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.rope_type == "3d_rope":
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids[0])  # 1, bs, len -> bs, len
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
        return output if return_dict else output.to_tuple()


class Qwen3MoeFoundationModel(BaseFoundationModelMixin, Qwen3MoeForCausalLM):
    config_class = Qwen3MoeFoundationConfig

    def __init__(self, config: Qwen3MoeFoundationConfig, **kwargs):
        Qwen3MoePreTrainedModel.__init__(self, config, **kwargs)
        self.spatial_merge_size = 2  # following qwen2vl
        self.rope_deltas = None  # for inference
        self.rope_type = config.rope_type  # 1d_rope/3d_rope

        self.model = MultimodalQwen3MoeModel(config)  # overwrite model.forward()
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.image_token_index = IMAGE_INPUT_INDEX
        self.video_token_index = VIDEO_INPUT_INDEX
        self.audio_token_index = AUDIO_INPUT_INDEX

        if self.rope_type == "3d_rope":
            self.mrope_section = [16, 24, 24]  # following qwen25omni

            qwen3_moe.apply_rotary_pos_emb = partial(apply_multimodal_rotary_pos_emb, mrope_section=self.mrope_section)

        self.post_init()

    def get_position_id_func(self):
        if self.rope_type == "3d_rope":
            fake_model = SimpleNamespace(
                config=SimpleNamespace(
                    vision_start_token_id=151669,  # only support Qwen3moe tokenizer TODO: unify this
                    audio_start_token_id=151671,
                    position_id_per_seconds=25,  # following qwen25omni
                    seconds_per_chunk=2.0,  # following qwen25omni
                ),
                image_token_index=self.image_token_index,
                video_token_index=self.video_token_index,
                audio_token_index=self.audio_token_index,
                spatial_merge_size=self.spatial_merge_size,
            )
            return [
                parse_position_id_kwargs,
                Qwen2_5OmniThinkerForConditionalGeneration.get_position_id_func(fake_model),
            ]
        else:
            return None  # the default position_id_func is torch.arange(len(input_ids))

    def prepare_inputs_for_generation(
        self,
        input_ids,
        rope_deltas=None,
        **kwargs,
    ):
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas
        # TODO: support qwen25omni inference
        return super().prepare_inputs_for_generation(
            input_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not get_parallel_state().sp_enabled and labels is not None:
            # Shift so that tokens < n predict n
            labels = labels[..., 1:].contiguous()
            labels = labels.view(-1)
            if (
                position_ids is not None
                and position_ids.size(0) == 1
                and not (torch.diff(position_ids, dim=-1) >= 0).all()
            ):
                position_ids_ = position_ids[:, 0].flatten()  # multimodal position ids is (bs, dim, seq_len)
                indices_q = torch.arange(position_ids_.size(0), device=position_ids_.device, dtype=torch.int32)
                cu_seq_lens = torch.cat(
                    (
                        indices_q[position_ids_ == 0],
                        torch.tensor(position_ids_.size(), device=position_ids_.device, dtype=torch.int32),
                    )
                )
                labels[cu_seq_lens[1:-1] - 1] = IGNORE_INDEX

        if position_ids is not None:
            position_ids = position_ids.transpose(0, 1).contiguous()  # bs, dim, len -> dim, bs, len

        if inputs_embeds is not None:
            input_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
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
            labels = labels.view(-1)  # flatten label
            if is_liger_kernel_available():
                loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="mean")
                if not get_parallel_state().sp_enabled:
                    # Shift so that tokens < n predict n
                    hidden_states = hidden_states[..., :-1, :].contiguous()

                hidden_states = hidden_states.view(-1, self.config.hidden_size)
                loss = loss_fct(self.lm_head.weight, hidden_states, labels)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                logits = self.lm_head(hidden_states)
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()

                if not get_parallel_state().sp_enabled:
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()

                # Flatten the tokens
                logits = logits.view(-1, self.vocab_size)
                torch.distributed.barrier()
                loss = loss_fct(logits, labels)

            if get_parallel_state().sp_enabled:
                num_valid_tokens = (labels != IGNORE_INDEX).sum()
                loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
        else:
            logits = self.lm_head(hidden_states)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
