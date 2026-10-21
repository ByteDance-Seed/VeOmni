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
"""LLaDA2.0-Uni image-understanding (I2T) training modeling.

The backbone is LLaDA2's MoE mask predictor: fused ``query_key_value`` attention with
qk RMSNorm and RoPE on the first ``rotary_dim`` channels, a grouped sigmoid router with
an ``expert_bias`` that only steers selection, routed experts plus one shared expert,
and dense MLPs in the first ``first_k_dense_replace`` layers. Images enter as tokens:
the frozen GlmImage vision tower and VQ codec under ``model.visual`` / ``model.vqmodel``
map each image to VQ codes, which fill the ``image_mask`` slots as vocabulary ids
``image_token_offset + code`` and are embedded by ``word_embeddings``. The codec's
argmin is not differentiable, so nothing upstream of it can train.

Weights load from the checkpoint's own key layout (see
``scripts/convert_llada2uni_to_veomni.py``); per-expert weights are fused at load time.
Trained by `DllmTrainer` with ``data_type: llada2_uni``; generation is not ported.
"""

from collections.abc import Callable

import torch
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.glm_image.modeling_glm_image import GlmImageVisionModel, GlmImageVQVAE
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.ops.kernels.moe_experts.standard.activation_gate import refuse_fused_moe_on_unsupported_activation
from veomni.ops.kernels.swiglu_mlp import has_plain_linear_projections

from ..llada.dllm_utils import bidirectional_attention_mask, dllm_lm_head_loss, ulysses_local_image_slots
from ..llada_moe.modeling_llada_moe import eager_attention_forward, rotate_half
from .configuration_llada2_uni import LLaDA2UniConfig


def apply_partial_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE on the first ``cos.shape[-1]`` channels of each head; the rest pass through."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class LLaDA2UniRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.veomni_rms_norm = VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


class LLaDA2UniRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: LLaDA2UniConfig, device=None):
        super().__init__()
        if config.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("LLaDA2.0-Uni uses the default RoPE; scaled variants are not ported.")
        dim = config.rotary_dim
        base = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class LLaDA2UniAttention(nn.Module):
    def __init__(self, config: LLaDA2UniConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        if config.use_qk_norm:
            self.query_layernorm = LLaDA2UniRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = LLaDA2UniRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.query_key_value(hidden_states).view(
            *input_shape, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_partial_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
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
        return self.dense(attn_output)


class LLaDA2UniMLP(nn.Module):
    def __init__(self, config: LLaDA2UniConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.veomni_swiglu_mlp = VeomniOp("swiglu_mlp", "standard", resolve_op_impl("swiglu_mlp_implementation"))

    def forward(self, x):
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


class LLaDA2UniExperts(nn.Module):
    """Routed experts, fused as ``gate_up_proj [E, 2I, H]`` (gate rows first) and ``down_proj [E, H, I]``."""

    def __init__(self, config: LLaDA2UniConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.hidden_act = config.hidden_act
        self.veomni_moe = VeomniOp("moe_experts", "standard", resolve_op_impl("moe_implementation"))
        if self.veomni_moe.impl != "eager":
            refuse_fused_moe_on_unsupported_activation(config, self.veomni_moe.impl)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        if self.veomni_moe.impl != "eager":
            unused = self.gate_up_proj.new_empty(0)
            return self.veomni_moe(
                hidden_states,
                top_k_weights.to(hidden_states.dtype),
                top_k_index,
                unused,
                unused,
                self.down_proj,
                self.gate_up_proj,
                num_experts=self.num_experts,
                activation=self.hidden_act,
            )
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            gate, up = nn.functional.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


class LLaDA2UniGate(nn.Module):
    """Sigmoid scores in float32; ``expert_bias`` shifts which experts win, not their weights."""

    def __init__(self, config: LLaDA2UniConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

    def group_limited_topk(self, scores: torch.Tensor) -> torch.Tensor:
        num_tokens = scores.shape[0]
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        return torch.topk(scores.masked_fill(~score_mask.bool(), float("-inf")), k=self.top_k, dim=-1)[1]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = nn.functional.linear(hidden_states.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        topk_idx = self.group_limited_topk(scores + self.expert_bias)
        topk_weight = torch.gather(scores, dim=1, index=topk_idx)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor


class LLaDA2UniSparseMoeBlock(nn.Module):
    def __init__(self, config: LLaDA2UniConfig):
        super().__init__()
        self.experts = LLaDA2UniExperts(config)
        self.gate = LLaDA2UniGate(config)
        self.shared_experts = LLaDA2UniMLP(config, config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        topk_idx, topk_weight = self.gate(hidden_flat)
        routed = self.experts(hidden_flat, topk_idx, topk_weight).view_as(hidden_states)
        return routed + self.shared_experts(hidden_states)


class LLaDA2UniDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LLaDA2UniConfig, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2UniAttention(config, layer_idx)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = LLaDA2UniSparseMoeBlock(config)
        else:
            self.mlp = LLaDA2UniMLP(config, config.intermediate_size)
        self.input_layernorm = LLaDA2UniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2UniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class LLaDA2UniPreTrainedModel(PreTrainedModel):
    config: LLaDA2UniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2UniDecoderLayer", "GlmImageVisionBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, LLaDA2UniExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, LLaDA2UniGate):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.expert_bias)


class LLaDA2UniModel(LLaDA2UniPreTrainedModel):
    def __init__(self, config: LLaDA2UniConfig):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LLaDA2UniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LLaDA2UniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2UniRotaryEmbedding(config)
        # The tower is frozen preprocessing. Pin it to sdpa so it cannot inherit
        # `veomni_flash_*` from the backbone, which would run the text-side
        # flash kernel on the vision sequence.
        self.visual = GlmImageVisionModel._from_config(config.vision_config, attn_implementation="sdpa")
        self.vqmodel = GlmImageVQVAE._from_config(config.vq_config)
        self.visual.requires_grad_(False)
        self.vqmodel.requires_grad_(False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        self.vqmodel.eval()
        return self

    @torch.no_grad()
    def get_image_tokens(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Packed patches -> each image's VQ codes in row-major patch order, offset into the vocabulary."""
        hidden = self.visual(pixel_values, grid_thw=image_grid_thw).last_hidden_state
        codes = []
        for features, (grid_t, grid_h, grid_w) in zip(
            torch.split(hidden, image_grid_thw.prod(dim=-1).tolist()), image_grid_thw.tolist()
        ):
            features = features.view(grid_t, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
            codes.append(self.vqmodel.encode(features).image_tokens.reshape(-1))
        return torch.cat(codes).long() + self.config.image_token_offset

    def _dummy_image(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        vision = self.config.vision_config
        grid = torch.tensor([[1, 2, 2]], device=device)
        return torch.zeros(4, vision.in_channels * vision.patch_size**2, device=device), grid

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_mask: torch.BoolTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None:
            raise ValueError("LLaDA2.0-Uni is ported for training only; there is no KV cache.")
        if pixel_values is not None:
            if input_ids is None or image_mask is None or image_grid_thw is None:
                raise ValueError("Images need `input_ids`, `image_mask` and `image_grid_thw` to place their tokens.")
            image_tokens = self.get_image_tokens(pixel_values, image_grid_thw)
            if int(image_mask.sum()) != image_tokens.numel():
                raise ValueError(
                    f"{int(image_mask.sum())} image slots but the images encode to {image_tokens.numel()} VQ tokens."
                )
            slots, image_tokens = ulysses_local_image_slots(image_mask, image_tokens, input_ids.shape[1])
            input_ids = input_ids.masked_scatter(slots, image_tokens)
        else:
            # FSDP all-gathers the tokenizer on every rank that runs it, so a rank without
            # images still runs a dummy one and discards the codes, in eval as in training.
            self.get_image_tokens(*self._dummy_image(self.word_embeddings.weight.device))
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask = bidirectional_attention_mask(self.config, inputs_embeds, attention_mask, position_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return MoeModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class LLaDA2UniForConditionalGeneration(LLaDA2UniPreTrainedModel):
    _tied_weights_keys = {}

    def __init__(self, config: LLaDA2UniConfig):
        super().__init__(config)
        self.model = LLaDA2UniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def _validate_checkpoint_missing_parameters(self, missing: set[str]) -> None:
        if missing:
            raise ValueError(
                f"The checkpoint is missing {len(missing)} LLaDA2.0-Uni tensor(s), e.g. {sorted(missing)[:5]}; "
                "convert it with `scripts/convert_llada2uni_to_veomni.py`, which also merges the image tokenizer."
            )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_mask: torch.BoolTensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_weight: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor`, *optional*):
            The original token at each masked position and `-100` elsewhere, not shifted.
        loss_weight (`torch.Tensor`, *optional*):
            Per-token weight `1 / (t * N)` drawn with the mask; the returned `loss` is the local
            `sum_i w_i * CE_i`, which `DllmTrainer` divides by the global sample count.
        """
        if (labels is None) != (loss_weight is None):
            raise ValueError(
                "LLaDA2.0-Uni is trained with the masked-diffusion objective: pass `labels` and `loss_weight` "
                "together (see the `llada2_uni` data transform)."
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_mask=image_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        loss = logits = None
        if labels is not None:
            loss = dllm_lm_head_loss(hidden_states, self.lm_head, labels, loss_weight)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        return MoeCausalLMOutputWithPast(loss=loss, logits=logits)

    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan as _get_parallel_plan

        return _get_parallel_plan()


__all__ = ["LLaDA2UniForConditionalGeneration", "LLaDA2UniModel", "LLaDA2UniPreTrainedModel"]


from ..llada_moe.checkpoint_tensor_converter import (  # noqa: E402
    convert_llada_moe_fqn_to_index_mapping,
    create_llada_moe_checkpoint_tensor_converter,
)


for _model_cls in (LLaDA2UniForConditionalGeneration, LLaDA2UniModel):
    _model_cls._create_checkpoint_tensor_converter = staticmethod(create_llada_moe_checkpoint_tensor_converter)
    _model_cls._convert_fqn_to_index_mapping = staticmethod(convert_llada_moe_fqn_to_index_mapping)
