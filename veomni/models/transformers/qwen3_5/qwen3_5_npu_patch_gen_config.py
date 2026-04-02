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
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_5.qwen3_5_npu_patch_gen_config -o veomni/models/transformers/qwen3_5/generated --diff

Language-model focused patches from qwen3_next example:
1. Device-agnostic GatedDeltaNet init and varlen FLA forward.
2. DecoderLayer forward with cu_seq_lens_q passthrough.
3. Use VeOmni fused loss path in Qwen3_5ForConditionalGeneration.forward.
"""

from copy import copy
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5Config,
    Qwen3_5DynamicCache,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5RMSNormGated,
    apply_mask_to_padding_states,
    torch_chunk_gated_delta_rule,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import sp_pad_and_slice
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.utils.device import get_device_id


logger = logging.get_logger(__name__)


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
config.add_import("veomni.utils.device", names=["get_device_id"])
config.add_import(
    "veomni.distributed.sequence_parallel.ulysses",
    names=["gather_seq_scatter_heads", "gather_heads_scatter_seq"],
)


@config.override_method(
    "Qwen3_5MoeRMSNorm.forward",
    description="Use eager Qwen3Next-style RMSNorm (1+weight centered formulation) for NPU patchgen",
)
def qwen3_5_moe_rmsnorm_forward_patched(self, x):
    # Modification: OpSlot guard — use fused RMSNorm kernel when bound.
    if veomni_rms_norm.has_kernel:
        return veomni_rms_norm(x, self.weight, self.eps)
    # Original HF code below, unchanged.
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)


config.add_import("veomni.distributed.sequence_parallel", names=["sp_pad_and_slice"])
config.add_import("veomni.utils.constants", names=["IMAGE_INPUT_INDEX", "VIDEO_INPUT_INDEX"])
config.drop_import_names(
    "FusedRMSNormGated",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "qwen3_5")
    veomni_causal_conv1d = OpSlot("causal_conv1d", "standard")
    veomni_chunk_gated_delta_rule = OpSlot("chunk_gated_delta_rule", "standard")

    FusedRMSNormGated = None
    fused_recurrent_gated_delta_rule = None
    causal_conv1d_update = None
    causal_conv1d_fn = None
    chunk_gated_delta_rule = None
    """
)


# Dummy definitions for names that exist in the generated file's scope but not here.
# The patchgen only extracts the function body; these are resolved at codegen time.
FusedRMSNormGated = None
Qwen3_5GatedDeltaNet = None
causal_conv1d_fn = None
causal_conv1d_update = None
torch_causal_conv1d_update = None
chunk_gated_delta_rule = None
torch_chunk_gated_delta_rule = None  # noqa: F811 — also imported above for the forward patch
fused_recurrent_gated_delta_rule = None
torch_recurrent_gated_delta_rule = None
is_fast_path_available = None
gather_seq_scatter_heads = None
gather_heads_scatter_seq = None


@config.override_method(
    "Qwen3_5GatedDeltaNet.__init__",
    description="Use device-agnostic get_device_id() for FusedRMSNormGated init",
)
def qwen3_5_gated_deltanet_init_patched(self, config: Qwen3_5Config, layer_idx: int):
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
    self.act = ACT2FN[config.hidden_act]
    self.layer_norm_epsilon = config.rms_norm_eps

    # QKV
    self.conv_dim = self.key_dim * 2 + self.value_dim
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=False,
        kernel_size=self.conv_kernel_size,
        groups=self.conv_dim,
        padding=self.conv_kernel_size - 1,
    )

    # time step projection (discretization)
    # instantiate once and copy inv_dt in init_weights of PretrainedModel
    self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

    A = torch.empty(self.num_v_heads).uniform_(0, 16)
    self.A_log = nn.Parameter(torch.log(A))

    self.norm = (
        Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        if FusedRMSNormGated is None
        else FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            # Modification: use device-agnostic get_device_id() instead of hardcoded device
            device=get_device_id(),
            dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
        )
    )

    self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
    self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

    if not is_fast_path_available:
        logger.warning_once(
            "The fast path is not available because one of the required library is not installed. Falling back to "
            "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
            " https://github.com/Dao-AILab/causal-conv1d"
        )

    self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
    self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
    self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
    self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)


@config.override_method(
    "Qwen3_5GatedDeltaNet._get_local_conv1d_weight",
    description="Shard depthwise conv1d weights for local heads under Ulysses SP",
)
def qwen3_5_gated_deltanet_get_local_conv1d_weight(
    self, ulysses_rank: int, local_key_dim: int, local_value_dim: int
) -> torch.Tensor:
    # Modification: shard depthwise conv1d weights to match head-sharded mixed_qkv channels.
    w_full = self.conv1d.weight.squeeze(1)
    assert w_full.shape[0] == self.key_dim * 2 + self.value_dim, (
        f"conv1d weight dim ({w_full.shape[0]}) must match "
        f"(2 * key_dim + value_dim) ({self.key_dim * 2 + self.value_dim})"
    )
    k_off = ulysses_rank * local_key_dim
    v_off = ulysses_rank * local_value_dim
    w_q = w_full[k_off : k_off + local_key_dim]
    w_k = w_full[self.key_dim + k_off : self.key_dim + k_off + local_key_dim]
    w_v = w_full[2 * self.key_dim + v_off : 2 * self.key_dim + v_off + local_value_dim]
    return torch.cat([w_q, w_k, w_v], dim=0)


@config.override_method(
    "Qwen3_5GatedDeltaNet.forward",
    description="Support varlen flash linear attention and Ulysses SP in Qwen3_5GatedDeltaNet.forward",
)
def qwen3_5_gated_deltanet_forward_patched(
    self,
    hidden_states: torch.Tensor,
    cache_params: Qwen3_5DynamicCache | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    # Modification: plumb varlen sequence metadata to FLA kernels.
    cu_seq_lens_q: torch.Tensor | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

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
    ulysses_enabled = get_parallel_state().ulysses_enabled
    if ulysses_enabled:
        ulysses_group = get_parallel_state().ulysses_group
        ulysses_size = get_parallel_state().ulysses_size
        ulysses_rank = get_parallel_state().ulysses_rank
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

        # All-to-all: gather full sequence, scatter heads -> [B, S_full, local_heads, head_dim]
        q_proj = gather_seq_scatter_heads(q_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        k_proj = gather_seq_scatter_heads(k_proj, seq_dim=1, head_dim=2, group=ulysses_group)
        v_proj = gather_seq_scatter_heads(v_proj, seq_dim=1, head_dim=2, group=ulysses_group)

        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2, group=ulysses_group)
        a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2, group=ulysses_group)

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

    if use_precomputed_states:
        # Modification: keep this disabled until FLA causal_conv1d_update decode path is validated.
        raise NotImplementedError("use_precomputed_states=True is not supported yet for causal_conv1d_update now.")
    else:
        if cache_params is not None:
            mixed_qkv_t = mixed_qkv.transpose(1, 2)
            conv_state = F.pad(mixed_qkv_t, (self.conv_kernel_size - mixed_qkv_t.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if veomni_causal_conv1d.has_kernel:
            # Modification: shard conv1d weights per Ulysses rank to match head-sharded channels.
            if ulysses_enabled:
                conv_weight = self._get_local_conv1d_weight(
                    ulysses_rank=ulysses_rank,
                    local_key_dim=local_key_dim,
                    local_value_dim=local_value_dim,
                )
            else:
                conv_weight = self.conv1d.weight.squeeze(1)
            # mixed_qkv is [B, S, D] — FLA causal_conv1d expects [B, S, D].
            mixed_qkv = veomni_causal_conv1d(
                x=mixed_qkv,
                weight=conv_weight,
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
                backend="triton",
                cu_seqlens=cu_seq_lens_q.npu(),
            )[0]
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
    # Modification: slice A_log/dt_bias for local V-heads under Ulysses SP.
    if ulysses_enabled:
        v_head_offset = ulysses_rank * local_num_v_heads
        v_head_slice = slice(v_head_offset, v_head_offset + local_num_v_heads)
        g = -self.A_log[v_head_slice].float().exp() * F.softplus(a.float() + self.dt_bias[v_head_slice])
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        if veomni_chunk_gated_delta_rule.has_kernel:
            # Modification: use direct args and pass cu_seqlens for varlen FLA attention.
            core_attn_out, last_recurrent_state = veomni_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seq_lens_q.npu(),
            )
        else:
            raise RuntimeError(
                "Varlen training requires FLA. Install flash-linear-attention so "
                "chunk_gated_delta_rule supports cu_seqlens."
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

    # Modification: gather attention output back to sequence-sharded layout before gated norm.
    if ulysses_enabled:
        core_attn_out = gather_heads_scatter_seq(
            core_attn_out, head_dim=2, seq_dim=1, group=get_parallel_state().ulysses_group
        )

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


@config.override_method(
    "Qwen3_5DecoderLayer.forward",
    description="Extract and pass cu_seq_lens_q for varlen linear attention in Qwen3_5DecoderLayer.forward",
)
def qwen3_5_decoder_layer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.FloatTensor:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Modification: read varlen metadata from kwargs and enforce it for linear-attention varlen kernels.
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    assert cu_seq_lens_q is not None, (
        "cu_seq_lens_q must be provided to support varlen Flash Linear Attention, varlen Conv1D,"
        "and to remove the full Flash Attention CPU-NPU sync."
    )

    # Token Mixer
    if self.layer_type == "linear_attention":
        # Modification: pass cu_seq_lens_q through to Qwen3_5GatedDeltaNet.forward.
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
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


@config.override_method(
    "Qwen3_5Model.get_image_features",
    description="Remove unnecessary split operation to maintain contiguous memory layout.",
)
def qwen3_5_model_get_image_features(
    self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None
):
    r"""
    Processes images through the vision tower and returns features as a single contiguous tensor.

    Optimization Note:
    We removed the original implementation's 'split' operation that breaks vision
    features into a list of tensors. In VeOmni, we maintain a single flattened tensor
    to support Sequence Parallelism (SP) and FSDP2 efficiently. Keeping features
    contiguous avoids Python list-overhead and enables direct execution of
    vectorized kernels in the main forward pass.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    vision_output: BaseModelOutputWithPooling = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
    return vision_output


@config.override_method(
    "Qwen3_5Model.get_placeholder_mask",
    description="Extract multimodal placeholder masks from input_ids using self-defined placeholder IDs.",
)
def qwen3_5_model_get_placeholder_mask(self, input_ids: torch.LongTensor, **kwargs):
    """
    Identifies positions of multimodal placeholder tokens (images and videos) in input_ids.

    Optimization Note:
    We simplified this method by removing 'inputs_embeds' from the argument list.
    In VeOmni, we primarily rely on 'input_ids' and self-defined placeholder IDs
    (e.g., IMAGE_INPUT_INDEX) instead of original Qwen token IDs. This decoupling
    ensures that the data pipeline remains model-agnostic.
    """
    special_image_mask = input_ids == self.config.image_token_id
    special_video_mask = input_ids == self.config.video_token_id
    return special_image_mask, special_video_mask


@config.override_method(
    "Qwen3_5VisionModel.fast_pos_embed_interpolate",
    description="Optimized bilinear interpolation for high-resolution vision embeddings, adapted from vLLM.",
)
def qwen3_5_vision_model_fast_pos_embed_interpolate(self, grid_thw):
    """
    Efficient implementation adapted from vLLM's Qwen-VL optimization.

    Key optimizations over standard Transformers implementation:
    1. Computational Efficiency: Reduces bilinear interpolation multiplications from 4 to 1
       per patch using algebraic simplification (w11=dh*dw; w10=dh-w11; w01=dw-w11; w00=1-dh-w01).
    2. Vectorization: Uses torch.meshgrid to compute indices and weights for the entire
       grid at once, avoiding expensive Python loops.

    Original source: https://github.com/vllm-project/vllm/blob/95c0f92/vllm/model_executor/models/qwen3_vl.py#L470
    """

    num_grid_per_side = self.num_grid_per_side
    m_size = self.spatial_merge_size
    hidden_dim = self.pos_embed.embedding_dim

    outputs = []
    dtype = self.pos_embed.weight.dtype
    for t, h, w in grid_thw:
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, device=self.device, dtype=torch.float64)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, device=self.device, dtype=torch.float64)

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # Create meshgrid view for all h, w vars
        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

        # original computation of weights
        # w00 = (1 - dh_grid) * (1 - dw_grid)
        # w01 = (1 - dh_grid) * dw_grid
        # w10 = dh_grid * (1 - dw_grid)
        # w11 = dh_grid * dw_grid
        # we reuse w11 here to avoid duplicate
        # dh_grid * dw_grid computation
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
        w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
        weights = weights.to(dtype=dtype)

        embeds = self.pos_embed(indices) * weights
        combined = embeds[0] + embeds[1] + embeds[2] + embeds[3]
        combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)

        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
        repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)

        outputs.append(repeated)

    return torch.cat(outputs, dim=0)


@config.override_method(
    "Qwen3_5VisionModel.forward",
    description="Optimized vision forward with Sequence Parallel (SP) support and padded cu_seqlens.",
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
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    # --- Patch.1: Sequence parallel padding and slicing for position embeddings ---
    if get_parallel_state().sp_enabled:
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

    # --- Patch.2: Flatten full-sequence rotary embeddings using the actual total sequence length ---
    # In Sequence Parallelism, hidden_states.size(0) only represents the local shard length.
    # We must use cu_seqlens[-1] (derived from unpadded grid_thw) to flatten the global
    # rotary_pos_emb. This ensures the embeddings cover the entire original sequence
    # before they are padded and sliced in Patch 3 to match the sharded hidden_states.
    total_seq_len = cu_seqlens[-1]
    rotary_pos_emb = rotary_pos_emb.reshape(total_seq_len, -1)
    # --- Patch.2 ---

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    if get_parallel_state().sp_enabled:
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
        pad_seq_len = seq_len * sp_size - total_seq_len.item()
        if pad_seq_len > 0:
            # Append a new entry to cu_seqlens to include the padding tokens as a final segment
            new_cumsum = cu_seqlens[-1] + pad_seq_len
            cu_seqlens = torch.cat([cu_seqlens, new_cumsum.unsqueeze(0)], dim=0)
        # --- Patch.4 ---

    for blk in self.blocks:
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


@config.override_method(
    "Qwen3_5VisionModel.dummy_forward",
    description="Add dummy_forward to prevent FSDP reduce-scatter hang on uneven multimodal batches.",
)
def qwen3_5_vision_model_dummy_forward(self):
    """
    # Run a fake ViT forward so every FSDP rank touches the vision tower.
    # This prevents reduce-scatter hangs when some ranks have no real images/videos.
    """
    if get_parallel_state().sp_enabled:
        sp_size = get_parallel_state().sp_size

        # Fake patch sequence for one local rank:
        # 16 patch tokens, each token flattened from:
        #   3 channels * 2 temporal patches * 16 * 16 spatial patch
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        # grid_thw describes the *global* pre-sharded vision grid, not the local shard.
        # Here:
        #   T = 1
        #   H = 4 * sp_size
        #   W = 4
        # so total global patch tokens = 1 * (4 * sp_size) * 4 = 16 * sp_size.
        grid_thw = torch.tensor([[1, 4 * sp_size, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    else:
        pixel_values = torch.zeros((16, 3 * 2 * 16 * 16), dtype=self.dtype, device=self.device)
        # Non-SP case: a minimal valid 4x4 patch grid.
        # Total patch tokens = 1 * 4 * 4 = 16.
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32, device=self.device)
        dummy_data = {"hidden_states": pixel_values, "grid_thw": grid_thw}
    return self(**dummy_data)


@config.override_method(
    "Qwen3_5Model.forward",
    description=(
        "Optimized multimodal forward supporting Ulysses SP (multimodal scattering), "
        "FSDP-safe dummy vision processing, position_ids shape alignment, and "
        "CPU-NPU sync avoidance via pre-computed metadata."
    ),
)
def qwen3_5_model_forward(
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
) -> tuple | Qwen3_5ModelOutputWithPast:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Token type IDs for multimodal inputs.
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
            input_ids = torch.cat(input_ids_list, dim=0)
        image_mask, video_mask = self.get_placeholder_mask(input_ids)
    # --- Patch.1 ---

    # --- Patch.4: Pop pre-computed Flash Attention kwargs to avoid ViT forward re-computation ---
    # The LM-level flash-attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`) are injected for packed-sequence attention. They must not reach the ViT, which computes its own `cu_seqlens`
    flash_attn_kwargs = {}
    for key in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]:
        if key in kwargs:
            flash_attn_kwargs[key] = kwargs.pop(key)
    # --- Patch.4 ---

    # --- Patch.1: Support Ulysses SP by transposing layout for multimodal scattering ---
    if get_parallel_state().sp_enabled:
        # Transpose from (batch, local_seq, full_hidden) to (batch, full_seq, local_hidden).
        # This gives each rank visibility over the ENTIRE sequence length, which is
        # necessary to scatter vision features into their correct global positions
        # as defined by the global pre-computed masks.
        inputs_embeds = gather_seq_scatter_heads(
            inputs_embeds, seq_dim=1, head_dim=2, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output

        # --- Patch.1: Shard image_embeds for sequence parallel scatter ---
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            image_embeds = gather_seq_scatter_heads(
                image_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        n_image_tokens = image_mask.sum().long().item()
        embeds_image_mask = (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )
        # Slice tensor to drop any padded image tokens
        image_embeds = image_embeds[:n_image_tokens]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_image_mask, image_embeds)

        # sequence parallel patch for image_mask & deepstack_image_embeds
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
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output

        # --- Patch.1: Shard video_embeds for sequence parallel scatter ---
        # sequence parallel patch for video embeds
        if get_parallel_state().sp_enabled:
            # (seq_len // sp_size, hidden_size) to  (seq_len, hidden_size // sp_size)
            video_embeds = gather_seq_scatter_heads(
                video_embeds, seq_dim=0, head_dim=-1, group=get_parallel_state().sp_group
            )
        n_video_tokens = video_mask.sum().long().item()
        embeds_video_mask = (
            video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device, non_blocking=True)
        )

        # Slice tensor to drop any padded video tokens
        video_embeds = video_embeds[:n_video_tokens]
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(embeds_video_mask, video_embeds)

        # sequence parallel patch for video_mask & deepstack_video_embeds
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
        inputs_embeds = gather_heads_scatter_seq(
            inputs_embeds, head_dim=2, seq_dim=1, group=get_parallel_state().sp_group
        )
    # --- Patch.1 ---

    if position_ids is None:
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
        # to (3, batch_size, seq_len) to match the internal requirements of the language model.
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

    return Qwen3_5ModelOutputWithPast(
        **outputs,
        rope_deltas=self.rope_deltas,
    )


config.add_post_import_block("""
def get_position_id(main_func, self, **kwargs):
    # Must be a module-level function for multiprocessing pickle
    position_ids, rope_deltas = main_func(self, **kwargs)
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}
""")


@config.override_method(
    "Qwen3_5ForConditionalGeneration.get_position_id_func",
    description="Expose get_position_id_func to pre-computes position IDs per sample during data preprocessing in worker processes.",
)
def qwen3_5_forconditional_generation_get_position_id_func(self):
    fake_config = copy(self.config)
    fake_config.image_token_id = IMAGE_INPUT_INDEX
    fake_config.video_token_id = VIDEO_INPUT_INDEX
    fake_model = SimpleNamespace(config=fake_config)
    return partial(get_position_id, Qwen3_5Model.get_rope_index, fake_model)  # noqa: F821 already defined via above `add_post_import_block`


@config.override_method(
    "Qwen3_5ForConditionalGeneration.forward",
    description="Support fused cross entropy path in Qwen3_5ForConditionalGeneration.forward",
)
def qwen3_5_forconditional_generation_forward_patched(
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
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5CausalLMOutputWithPast:
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

    return Qwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
